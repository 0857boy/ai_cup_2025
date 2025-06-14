# Task 1: 語音識別 (ASR) - 醫病語音敏感個人資料辨識競賽

## 📖 任務概述

Task 1 專注於語音識別 (Automatic Speech Recognition, ASR)，將醫療對話語音檔案轉換為文字並輸出轉錄結果。本任務提供了多種實作方案，從基礎的 Whisper 到進階的 WhisperX + Ollama 整合方案。

## 🎯 任務目標

- 將醫療語音對話轉換為準確的文字轉錄
- 提供字符級時間戳對齊功能
- 支援中文（繁簡轉換）語音識別
- 整合 NER 標註功能以識別敏感健康資訊

## 📁 檔案結構

```
task1/
├── README.md                    # 本檔案 - Task 1 詳細說明
├── ollama_qwen_whis.py         # 主要處理腳本 (WhisperX + Ollama NER)
├── whisper_large_v3.py         # 基礎 Whisper 版本
└── Whisperx.ipynb              # WhisperX 進階版本 (Jupyter Notebook)
```

## 🚀 核心功能

### 1. WhisperX + Ollama 整合方案 (`ollama_qwen_whis.py`)

**特點：**
- 使用 WhisperX Large-v3 進行高精度語音識別
- 整合 Ollama Qwen3 進行中文 NER 標註
- 支援字符級時間戳對齊
- 自動簡繁轉換
- 支援多語言檢測

**支援的 NER 標籤：**
```python
NER_LABELS = [
    "PATIENT", "DOCTOR", "FAMILYNAME", "PERSONALNAME",
    "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL", 
    "STREET", "CITY", "DISTRICT", "COUNTY", "STATE", "COUNTRY", 
    "AGE", "DATE", "TIME", "DURATION", "SET", "PHONE"
]
```

### 2. 基礎 Whisper 版本 (`whisper_large_v3.py`)

**特點：**
- 使用 whisper-timestamped 套件
- 支援 Whisper Large-v3 模型
- 基礎語音轉文字功能
- 適合快速原型開發

### 3. WhisperX 進階版本 (`Whisperx.ipynb`)

**特點：**
- Jupyter Notebook 互動式環境
- 支援批次處理多個資料夾
- 提供字符級時間戳索引
- 支援對齊後的精確時間標記

## 🛠️ 環境設置

### 系統需求
- **Python**: 3.8+
- **GPU**: 建議 8GB+ VRAM (CUDA 11.0+)
- **記憶體**: 建議 16GB+ RAM

### 依賴套件安裝

```bash
# 基礎套件
pip install torch torchaudio transformers

# WhisperX 相關
pip install whisperx
pip install whisper-timestamped

# 文字處理
pip install opencc-python-reimplemented
pip install numpy pandas tqdm

# Ollama (本地 LLM)
# 請至 https://ollama.ai 下載並安裝
ollama pull qwen3:8b
```

## 📋 使用說明

### 方案一：WhisperX + Ollama 整合方案（推薦）

```bash
# 基本使用
python ollama_qwen_whis.py

# 指定輸入資料夾和輸出檔案
python ollama_qwen_whis.py --input_dir "audio_files/" --task1_output "asr_results.txt"
```

**輸出格式：**
- `task1_output.txt`: 檔案名稱 + 轉錄文字
- `task2_output.txt`: NER 標註結果

**範例輸出：**
```
1001	患者王小明今天來看診，主訴是胸痛。
1002	醫師建議做心電圖檢查，時間安排在下午2點。
```

### 方案二：基礎 Whisper 版本

```python
import whisper_timestamped as whisper
import os

# 載入模型
model = whisper.load_model("whisper-large-v3", device="cuda")

# 處理單一檔案
audio_path = "audio_file.wav"
audio = whisper.load_audio(audio_path)
result = whisper.transcribe(model, audio, language="zh")

print(result["text"])
```

### 方案三：WhisperX 進階版本

開啟 `Whisperx.ipynb` 並執行以下步驟：

1. **安裝依賴**
```bash
!pip install whisperx
!git lfs install
!git clone https://huggingface.co/Systran/faster-whisper-large-v3
```

2. **載入模型並處理**
```python
import whisperx

device = "cuda"
compute_type = "float16"

model = whisperx.load_model("/content/faster-whisper-large-v3", device, compute_type=compute_type)

# 處理音頻檔案
audio = whisperx.load_audio(wav_file)
result = model.transcribe(audio)
```

3. **時間戳對齊**
```python
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)
```

## ⚙️ 配置選項

### WhisperX 配置

```python
class WhisperXProcessor:
    def __init__(self, 
                 model_name="large-v3",      # 模型版本
                 device="cuda",              # 計算設備
                 compute_type="float16",     # 計算精度
                 language=None):             # 語言設定 (None=自動檢測)
```

### 性能優化設定

```python
# 啟用 TensorFloat-32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 混合精度計算
compute_type = "float16"  # GPU
compute_type = "int8"     # CPU
```

## 📊 性能指標

### 處理速度
- **GPU (RTX 4090)**: 10-20x 實時處理速度
- **CPU**: 0.5-1x 實時處理速度

### 準確度
- **中文語音識別**: WER < 15%
- **英文語音識別**: WER < 10%
- **多語言混合**: WER < 20%

### 記憶體使用
- **WhisperX Large-v3**: ~6GB VRAM
- **基礎 Whisper**: ~4GB VRAM
- **CPU 模式**: ~8GB RAM

## 🔧 故障排除

### 常見問題

1. **CUDA 記憶體不足**
```bash
# 解決方案：使用較小的模型或 CPU 模式
model_name = "medium"  # 或 "small", "base"
device = "cpu"
compute_type = "int8"
```

2. **Ollama 連線失敗**
```bash
# 確認 Ollama 服務運行
ollama list
ollama run qwen3:8b

# 檢查模型是否已下載
ollama show qwen3:8b
```

3. **音頻格式不支援**
```bash
# 轉換為支援格式 (16kHz WAV)
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

4. **簡繁轉換問題**
```python
from opencc import OpenCC
converter = OpenCC('s2t')  # 簡轉繁
text = converter.convert("简体中文")
```

### 錯誤訊息對照

| 錯誤訊息 | 原因 | 解決方案 |
|----------|------|----------|
| `CUDA out of memory` | GPU 記憶體不足 | 減少批次大小或使用 CPU |
| `Model not found` | 模型檔案不存在 | 重新下載模型 |
| `Audio format not supported` | 音頻格式問題 | 轉換為 WAV 格式 |
| `Connection refused` | Ollama 服務未啟動 | 啟動 Ollama 服務 |

## 🎛️ 進階配置

### 自訂 NER 提示詞

```python
ollama_prompt = f"""你是一個中文醫療專用NER標註工具，請根據下列分類從中文句子中提取命名實體。

類別如下：
{', '.join(NER_LABELS)}

輸出格式為：<類別>\\t<實體文字>，每行一個實體。若無實體，請回答 "無實體"。

特殊規則：
- PATIENT, DOCTOR, PERSONALNAME, FAMILYNAME都算是NAME的範疇
- 名字後加上總、醫師、醫生都代表是DOCTOR  
- 代稱都不能算是NAME的範疇，哥哥、爸爸、大哥、老大都不是NAME
- HOSPITAL一定要是醫院名字，只有醫院兩個字或著本院都不是HOSPITAL
- ROOM為床位資訊，而不是房間名，手術室、急診室都不是ROOM

以下是句子：{text}
請標註所有實體。
"""
```

### 批次處理配置

```python
# 處理多個資料夾
folder_paths = [
    '/path/to/folder1',
    '/path/to/folder2',
    '/path/to/folder3'
]

# 並行處理設定
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    # 並行處理音頻檔案
    pass
```

## 📈 評估指標

### 語音識別評估
- **字符錯誤率 (CER)**: 字符級別的識別錯誤率
- **詞錯誤率 (WER)**: 詞級別的識別錯誤率
- **BLEU 分數**: 翻譯品質評估

### NER 標註評估
- **精確率 (Precision)**: 正確識別的實體 / 總識別實體
- **召回率 (Recall)**: 正確識別的實體 / 總真實實體  
- **F1 分數**: 精確率和召回率的調和平均

## 🔗 相關資源

### 模型資源
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Ollama 官網](https://ollama.ai/)

### 技術文檔
- [Whisper 模型介紹](https://openai.com/research/whisper)
- [WhisperX 論文](https://arxiv.org/abs/2303.00747)
- [語音識別評估指標](https://en.wikipedia.org/wiki/Word_error_rate)

## 💡 最佳實踐

1. **音頻預處理**：使用預處理模組進行音頻增強
2. **批次處理**：合理設定批次大小避免記憶體溢出
3. **模型選擇**：根據硬體資源選擇適當的模型大小
4. **結果驗證**：人工檢查關鍵結果確保品質
5. **錯誤處理**：實作完整的異常處理機制

## 📝 更新日誌

- **v1.0.0**: 初始版本，支援基礎 Whisper 語音識別
- **v1.1.0**: 加入 WhisperX 支援，提升識別精度
- **v1.2.0**: 整合 Ollama NER 功能，完整的端到端解決方案
- **v1.3.0**: 加入字符級時間戳對齊，支援多語言檢測
