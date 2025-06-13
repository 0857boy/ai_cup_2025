#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper-large-v3 轉錄 + Ollama-Qwen NER
Task 1：<id>\t<全文>
Task 2：<id>\t<label>\t<start>\t<end>\t<entity>
"""

import os
import sys
import logging
import shutil
from typing import List, Tuple, Dict, Any

import torch
import whisperx
from opencc import OpenCC
from tqdm import tqdm
import ollama

# ===== 路徑自我檢查 ======================================================
for tool in ("ffmpeg", "ffprobe", "sox"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = shutil.which(tool)

# ===== NER 標籤定義 ======================================================
NER_LABELS = [
    "PATIENT", "DOCTOR", "FAMILYNAME", "PERSONALNAME",
    "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL",
    "STREET", "CITY", "DISTRICT", "COUNTY", "STATE", "COUNTRY",
    "AGE", "DATE", "TIME", "DURATION", "SET", "PHONE"
]

# ===== 日誌設定 ==========================================================
logger = logging.getLogger("whisperx_ner")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ===== WhisperX 處理器 ===================================================
class WhisperXProcessor:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，改用 CPU")
            device, compute_type = "cpu", "int8"

        self.device = device
        self.compute_type = compute_type
        self.converter = OpenCC("s2t")  # 簡→繁
        logger.info(f"載入 WhisperX: {model_name} ({device})")
        self.model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language=language
        )
        self._align_cache: dict[str, Tuple[Any, Any]] = {}

    # ---------- 取得對齊模型 ----------
    def _get_align_model(self, lang: str) -> Tuple[Any, Any]:
        if lang not in self._align_cache:
            logger.info(f"載入 {lang} 對齊模型…")
            self._align_cache[lang] = whisperx.load_align_model(
                language_code=lang, device=self.device
            )
        return self._align_cache[lang]

    # ---------- 轉錄 + 對齊 ----------
    def transcribe(self, wav_path: str) -> Tuple[str, Dict]:
        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        result = self.model.transcribe(wav_path)

        # 多語言段落分別對齊
        langs = {s.get("language", "zh") for s in result["segments"]}
        aligned_segments, word_segments = [], []
        for lang in langs:
            segs = [s for s in result["segments"] if s.get("language", "zh") == lang]
            align_model, metadata = self._get_align_model(lang)
            aligned = whisperx.align(
                segs, align_model, metadata, wav_path, self.device, return_char_alignments=True
            )
            aligned_segments.extend(aligned["segments"])
            word_segments.extend(aligned["word_segments"])

        result["segments"] = aligned_segments
        result["word_segments"] = word_segments
        return file_id, result

# ======== 主要處理流程 ====================================================
def process_dataset(
    input_dir: str,
    task1_path: str,
    whisper_model: str = "large-v3",
    device: str = "cuda",
):
    wp = WhisperXProcessor(model_name=whisper_model, device=device)
    wav_list = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    logger.info(f"共偵測到 {len(wav_list)} 支音檔")

    with open(task1_path, "w", encoding="utf-8") as f_t1, \
         open("task2_output.txt", "w", encoding="utf-8") as f_t2:

        for wav in tqdm(wav_list, desc="轉錄/標註"):
            wav_path = os.path.join(input_dir, wav)
            file_id, asr_result = wp.transcribe(wav_path)

            # 轉繁體全文
            raw = " ".join(s["text"].strip() for s in asr_result["segments"])
            full_text = wp.converter.convert(raw)
            f_t1.write(f"{file_id}\t{full_text}\n")

            # ---------- 呼叫 Ollama 做 NER ----------
            prompt = (
                f"你是一個中文醫療專用 NER 標註工具，請根據下列分類從中文句子中提取命名實體。\n"
                f"類別：{', '.join(NER_LABELS)}\n"
                f"輸出格式：<類別>\\t<實體文字>，每行一個實體；若無實體請回答「無實體」。\n\n"
                f"句子：{full_text}\n請標註所有實體。"
            )

            try:
                ner_resp = ollama.chat(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                ner_text = ner_resp["message"]["content"].strip()
            except Exception as e:
                logger.error(f"{file_id} NER 失敗：{e}")
                continue

            if ner_text == "無實體":
                continue

            # ---------- 時間戳對齊 ----------
            word_segs = asr_result.get("word_segments", [])
            token_text = "".join(OpenCC("s2t").convert(w["word"]) for w in word_segs)

            written: set[Tuple] = set()
            for row in ner_text.splitlines():
                if "\t" not in row:
                    continue
                label, entity = (s.strip() for s in row.split("\t", 1))
                label = label.strip("<>").upper()
                if not entity:
                    continue

                # 在全文中尋找所有出現位置
                idx = token_text.find(entity)
                while idx != -1:
                    # 對應到 token 編號
                    char_ptr, start_tok, end_tok = 0, None, None
                    for i, seg in enumerate(word_segs):
                        nxt = char_ptr + len(seg["word"])
                        if start_tok is None and nxt > idx:
                            start_tok = i
                        if start_tok is not None and nxt >= idx + len(entity):
                            end_tok = i
                            break
                        char_ptr = nxt
                    if start_tok is not None and end_tok is not None:
                        st = word_segs[start_tok]["start"]
                        et = word_segs[end_tok]["end"]
                        key = (label, round(st, 2), round(et, 2), entity)
                        if key not in written:
                            f_t2.write(f"{file_id}\t{label}\t{st:.2f}\t{et:.2f}\t{entity}\n")
                            written.add(key)
                    idx = token_text.find(entity, idx + 1)

    logger.info("全部處理完成！")


# ======== CLI 入口 =======================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="音檔資料夾")
    ap.add_argument("--task1", default="task1_output.txt", help="Task 1 輸出路徑")
    ap.add_argument("--model", default="large-v3", help="WhisperX 模型")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"❌ 找不到資料夾：{args.input_dir}")

    process_dataset(args.input_dir, args.task1, args.model, args.device)
