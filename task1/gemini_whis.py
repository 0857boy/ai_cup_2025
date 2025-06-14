import os
import torch
import whisperx
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from opencc import OpenCC
import google.generativeai as genai

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

genai.configure(api_key="your api keys")
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-pro-preview-05-06")

NER_LABELS = [
    "PATIENT", "DOCTOR", "FAMILYNAME", "PERSONALNAME",
    "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL",  "STREET", "CITY",
    "DISTRICT", "COUNTY", "STATE", "COUNTRY", 
    "AGE", "DATE", "TIME", "DURATION", "SET", "PHONE"
]

# 設定日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WhisperXProcessor:
    """使用WhisperX進行語音識別"""

    def __init__(self, model_name="large-v3", device="cuda", compute_type="float16", language=None):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.converter = OpenCC('s2t')

        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA不可用，將使用CPU替代")
            self.device = "cpu"
            self.compute_type = "int8"

        logger.info(f"初始化WhisperX處理器，模型: {model_name}，裝置: {self.device}")
        self.model = self._load_model()
        self.alignment_models = {}

    def _load_model(self):
        try:
            logger.info(f"正在載入WhisperX ASR模型 {self.model_name}...")
            model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            logger.info("WhisperX ASR模型載入成功")
            return model
        except Exception as e:
            logger.error(f"載入WhisperX ASR模型失敗: {e}")
            raise

    def _get_alignment_model(self, language_code):
        if language_code not in self.alignment_models:
            try:
                logger.info(f"正在載入{language_code}語言的對齊模型...")
                model, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self.alignment_models[language_code] = (model, metadata)
                logger.info(f"{language_code}對齊模型載入成功")
            except Exception as e:
                logger.error(f"載入{language_code}對齊模型失敗: {e}")
                return None, None
        return self.alignment_models[language_code]

    def transcribe_audio(self, audio_path):
        try:
            file_id = os.path.splitext(os.path.basename(audio_path))[0]
            logger.info(f"正在處理音頻檔案: {file_id}")
            result = self.model.transcribe(audio_path)
            lang_codes = set()
            for segment in result["segments"]:
                lang_code = segment.get("language", "zh")
                lang_codes.add(lang_code)

            logger.info(f"檢測到的語言: {lang_codes}")
            aligned_segments = []

            for lang_code in lang_codes:
                lang_segments = [s for s in result["segments"] if s.get("language", "zh") == lang_code]
                if not lang_segments:
                    continue

                align_model, metadata = self._get_alignment_model(lang_code)
                if align_model is None:
                    logger.warning(f"無法對{lang_code}語言進行對齊，跳過")
                    aligned_segments.extend(lang_segments)
                    continue

                aligned_result = whisperx.align(
                    lang_segments,
                    align_model,
                    metadata,
                    audio_path,
                    self.device,
                    return_char_alignments=True
                )

                if "segments" in aligned_result:
                    aligned_segments.extend(aligned_result["segments"])
                if "word_segments" in aligned_result and "word_segments" not in result:
                    result["word_segments"] = aligned_result["word_segments"]
                elif "word_segments" in aligned_result:
                    result["word_segments"].extend(aligned_result["word_segments"])

            result["segments"] = aligned_segments
            return file_id, result

        except Exception as e:
            logger.error(f"轉錄音頻檔案 {audio_path} 失敗: {e}")
            return None, None

    def process_directory(self, input_dir, task1_output):
        audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        logger.info(f"在 {input_dir} 中找到 {len(audio_files)} 個音頻檔案")
        output_file = "task2_output.txt"

        with open(task1_output, 'w', encoding='utf-8') as f_task1, \
             open(output_file, "w", encoding="utf-8") as fout:
            for audio_file in tqdm(audio_files, desc="處理音頻檔案"):
                audio_path = os.path.abspath(os.path.join(input_dir, audio_file))
                print("➡️ 嘗試處理檔案路徑：", audio_path)
                cc = OpenCC('s2t')

                file_id, result = self.transcribe_audio(audio_path)

                # 把所有 word_segments 的 word 欄位轉為繁體
                for seg in result.get("word_segments", []):
                    seg["word"] = cc.convert(seg["word"])
                logger.info(f"將簡體中文轉成繁體中文")

                if file_id is None or result is None:
                    logger.warning(f"跳過處理失敗的檔案: {audio_file}")
                    continue
                  
                if "segments" in result:
                    raw_text = " ".join([segment.get("text", "").strip() for segment in result["segments"]])
                    text = self.converter.convert(raw_text)

                    f_task1.write(f"{file_id}\t{text}\n")
                    logger.info(f"Task 1 - 檔案 {file_id} 轉錄完成")

                    gemini_prompt = f"""你是一個中文醫療專用NER標註工具，請根據下列分類從中文句子中提取命名實體。
                    類別如下：
                    {', '.join(NER_LABELS)}
                    輸出格式為：<類別>\\t<實體文字>，每行一個實體。若無實體，請回答 "無實體"。
                    規則說明：
                    -PATIENT, DOCTOR, PERSONALNAME, FAMILYNAME都算是NAME的範疇，而NAME不能是代稱，爸爸、媽媽都不算NAME
                    -名字後加上總、醫師、醫生都代表是DOCTOR
                    -代稱都不能算是NAME的範疇，哥哥、爸爸、大哥、老大都不是NAME
                    -HOSPITAL一定要是醫院名字，只有醫院兩個字或著本院都不是HOSPITAL
                    -對話都是發生在醫院，出現人物除非確認是DOCTOR優先都是PATIENT，如果都出現過才可能是其他種類名字
                    -如果前面出現過的名字，後面就貼一樣的標籤，例如前面出現王小明醫師並標註DOCTOR，後面出現又小明那就是DOCTOR
                    -ROOM為床位資訊，而不是房間名，手術室、急診室都不是ROOM
                    -PATIENT為病人姓名，多為完整人名，有時重複出現同一個人名
                    -TIME為明確時間，如「下午6點」、「兩點半」、「明早」、「早安」等，常與
                    -DATE為特定日期或相對日期，如「今天」、「前天」、「民國107年8月2日」、「現在」。
                    -PROFESSION只能是除了DOCTOR以外的職業,學長不是職業,代表工作名稱
                    -名字部分，出現較少（如「佳佳」、「劉」），可與 PATIENT 或 FAMILYNAME 聯合推測出全名。
                    -DEPARTMENT為醫療單位，如「移植中心」、「器官捐贈小組」，精神科、麻醉科等科不是DEPARTMENT
                    -SET是	重複性時間描述，如 once a week（每週一次）、every day（每天）。
                    -CITY為縣市名稱，可能與 HOSPITAL 一同出現，例:高雄、桃園、台北。
                    -PROFESSION,職業稱謂，如 advisor（顧問）、manager（經理）、teacher（教師）、babysitter（保母）等。
                    -DEPARTMENT指部門，外科、精神科等科別不是DEPARTMENT
                    

                    以下是句子：{text}\n請標註所有實體。
                    """
                    
                    try:
                        response = gemini_model.generate_content(gemini_prompt)
                        ner_result_text = response.text.strip()
                    except Exception as e:
                        logger.error(f"⚠️ {file_id} 發生錯誤：{e}")
                        exit(1)

                    # 處理回應結果
                    if ner_result_text != "無實體":
                        written_entities = set()
                        for row in ner_result_text.split('\n'):
                            if '\t' in row:
                                label, entity = row.split('\t', 1)
                                label = label.strip().strip('<>').upper()
                                entity = entity.strip()

                                ### 詞彙檢查區 ###
                                suffixes_to_strip = ["醫師", "醫生", "老師", "總"]
                                for suffix in suffixes_to_strip:
                                    if entity.endswith(suffix) and len(entity) > len(suffix):
                                        entity = entity[:-len(suffix)]
                                        break

                                pronouns = {"我", "你", "他", "她", "它", "我們", "你們", "他們", "她們", "它們", "別人"}
                                dep = {"ICU", "icu", "病房", "急診室"}
                                if label == "PERSONALNAME" and entity in pronouns:
                                    continue
                                if label == "FAMILYNAME" and entity in {"媽媽", "爸爸", "媽", "爸"}:
                                    continue
                                if label == "PATIENT" and entity in pronouns:
                                    continue
                                if label == "PROFESSION" and entity in {"學長"}:
                                    continue
                                if label == "ROOM" and entity in {"手術房", "產房", "急診室"}:
                                    continue
                                if label == "DEPARTMENT" and entity in dep:
                                    continue
                                if label == "TIME" and entity in {"等一下", "剛剛"}:
                                    continue
                                if label == "DURATION" and entity in {"等一下"}:
                                    continue
                                if label == "HOSPITAL" and entity in {"醫院", "本院"}:
                                    continue
                                if label == "DATE" and entity in {"剛剛", "現在", "目前", "剛才", "之前"}:
                                    continue
                                ### 詞彙檢查區 ###

                                try:
                                    word_segments = result.get("word_segments", [])
                                    # 將所有 token 轉為繁體
                                    for seg in word_segments:
                                        seg["word"] = cc.convert(seg["word"])

                                    # 把 token 接起來做全文字串
                                    full_text = "".join(seg["word"] for seg in word_segments)

                                    entity_positions = []
                                    search_start = 0
                                    while True:
                                        idx = full_text.find(entity, search_start)
                                        if idx == -1:
                                            break
                                        entity_positions.append(idx)
                                        search_start = idx + 1  # 允許重複出現

                                    for idx in entity_positions:
                                        # 找出 token 中第 idx 個字起的 token 編號
                                        char_idx = 0
                                        start_token = None
                                        end_token = None
                                        for i, seg in enumerate(word_segments):
                                            word_len = len(seg["word"])
                                            if start_token is None and char_idx + word_len > idx:
                                                start_token = i
                                            if start_token is not None and char_idx + word_len >= idx + len(entity):
                                                end_token = i
                                                break
                                            char_idx += word_len

                                        if start_token is not None and end_token is not None:
                                            start_time = word_segments[start_token]["start"]
                                            end_time = word_segments[end_token]["end"]
                                            entity_key = (label, round(start_time, 2), round(end_time, 2), entity)
                                            if entity_key not in written_entities:
                                                fout.write(f"{file_id}\t{label}\t{start_time:.2f}\t{end_time:.2f}\t{entity}\n")
                                                written_entities.add(entity_key)
                                except Exception as e:
                                    logger.warning(f"{file_id} 對照時間戳時錯誤: {e}")
            

def main():
    import argparse
    parser = argparse.ArgumentParser(description='使用WhisperX進行語音轉錄')
    parser.add_argument('--input_dir', default="TRAINING_DATASET_2/chinese_audio", help='輸入音頻目錄')
    parser.add_argument('--task1_output', default="whisperx_task1_test.txt", help='Task 1輸出檔案路徑')
    parser.add_argument('--model', default="large-v3", help='WhisperX模型名稱')
    parser.add_argument('--device', default="cuda", help='運算裝置')
    parser.add_argument('--language', default=None, help='辨識語言')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"輸入目錄不存在: {args.input_dir}")
        return

    processor = WhisperXProcessor(
        model_name=args.model, 
        device=args.device, 
        language=args.language
    )
    processor.process_directory(args.input_dir, args.task1_output)
    logger.info(f"處理完成！Task 1結果已保存至 {args.task1_output}")

    

if __name__ == "__main__":
    main()
