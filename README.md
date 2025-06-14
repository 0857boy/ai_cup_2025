# AI Cup 2025 - 醫病語音敏感個人資料辨識競賽

## 🏆 競賽概述

本專案為 AI Cup 2025 醫病語音敏感個人資料辨識競賽的完整解決方案，專注於從醫療對話中識別並保護患者隱私資訊，涵蓋語音轉文字 (ASR)、敏感健康資訊識別 (SHI Detection) 以及時間戳對齊等多項先進技術。

### 🎯 任務目標

- **Task 1**: 語音識別 (ASR) - 將醫療對話語音檔案轉換為文字並輸出轉錄結果
- **Task 2**: 敏感健康資訊識別 (SHI Detection) - 從轉錄文字中識別醫療敏感健康資訊並提供時間戳對齊，以保護患者隱私

### 🏅 主要成果

- **語音識別**: 使用 WhisperX Large-v3 實現高精度 ASR
- **敏感健康資訊識別**: 基於 XLM-RoBERTa + CRF + FGM 對抗訓練達到優異的敏感健康資訊識別效能
- **音頻處理**: 先進的 Transformer 神經網絡音頻增強系統
- **隱私保護**: 智能敏感健康資訊檢測與時間戳對齊技術

## 📁 專案架構

```
ai_cup_2025/
├── README.md                    # 專案總覽 (本檔案)
├── preprocess/                  # 音頻預處理模組
│   ├── audio_prepare.py         # 先進音頻處理 (Transformer 增強)
│   ├── audio_prepare(task1).py  # 輕量級音頻處理
│   ├── split_and_check_k_hold_with_test.py  # 數據集分割
│   └── README.md                # 預處理模組說明
├── task1/                       # Task 1: 語音識別
│   ├── README.md                # Task 1 詳細說明
│   ├── gemini_whis.py          # 主要處理腳本 (WhisperX + Gemini)
│   ├── Whisperx.ipynb          # WhisperX 基礎實現筆記本
│   └── config.json             # 音頻檔案路徑配置
│  
└── task2/
    ├── README.md                                      # 本說明文件
    ├── NER_CRF_FGM_BIO.ipynb                          # CRF + FGM 訓練主程式
    ├── predict_all.ipynb                              # 模型預測
    ├── Insert_timestamp.ipynb                         # 時間戳對齊處理
    ├── generate_task2_test_data_index.ipynb           # 產生任務二在句子中的index位置
    ├── config.json                                    # 路徑設定
    └── other_train_method                            # 其他model的訓練檔
        ├── config.json                            # 配置檔案
        ├── NER_BIO.ipynb                          # BIO 基礎模型
        ├── NER_BIOU.ipynb                         # BIOU 基礎模型
        ├── NER_CRF_BIO.ipynb                      # BIO + CRF 模型
        ├── NER_CRF_BIOU.ipynb                     # BIOU + CRF 模型
        ├── FGM_BIO.ipynb                          # BIO + FGM 對抗訓練
        ├── FGM_BIOU.ipynb                         # BIOU + FGM 對抗訓練
        ├── FOCAL_LOSS_BIO.ipynb                   # BIO + Focal Loss
        ├── FOCAL_LOSS_BIOU.ipynb                  # BIOU + Focal Loss
        ├── weight_LOSS_BIO.ipynb                  # BIO + 權重調整
        └── weight_LOSS_BIOU.ipynb                 # BIOU + 權重調整
```

## 🚀 核心技術特點

### 🎵 音頻預處理 (preprocess/)
- **Transformer 神經網絡增強**: 基於深度學習的音頻降噪與增強
- **心理聲學建模**: 基於人類聽覺感知的智能音頻處理
- **GPU 混合精度計算**: 支援 CUDA 加速，大幅提升處理速度
- **K-fold 數據分割**: 使用 MultilabelStratifiedKFold 確保標籤分布平衡

### 🗣️ 語音識別 (task1/)
- **WhisperX Large-v3**: 業界領先的多語言語音識別模型
- **Gemini 2.5 Pro**: Google 先進的LLM模型
- **字符級時間戳對齊**: 精確到字符級別的時間定位
- **智能語言檢測**: 自動檢測和處理多種語言
- **繁簡轉換**: 自動處理繁簡體中文轉換

### 🏷️ 敏感健康資訊識別 (task2/)
- **XLM-RoBERTa Large**: 多語言預訓練模型
- **CRF (條件隨機場)**: 確保序列標註一致性
- **FGM 對抗訓練**: 提升模型魯棒性
- **隱私保護**: 專注於識別醫療對話中的敏感健康資訊 (SHI)

## 🛠️ 環境設置

### 系統需求
- **作業系統**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **GPU**: 建議 8GB+ VRAM (支援 CUDA 11.0+)
- **記憶體**: 建議 16GB+ RAM

### 核心依賴套件

```bash
# 基礎套件
pip install torch torchaudio transformers
pip install numpy pandas tqdm scikit-learn

# 音頻處理
pip install librosa soundfile whisperx
pip install scipy psutil GPUtil

# NER 相關
pip install torchcrf pytorch-crf
pip install datasets opencc-python-reimplemented

# AI 模型服務
pip install google-generativeai  # Gemini 2.5 Pro

# 數據處理
pip install iterstrat  # 多標籤分層分割
```

### 快速安裝

```bash
# 克隆專案
git clone https://github.com/your-repo/ai_cup_2025.git
cd ai_cup_2025

# 安裝依賴 (建議使用虛擬環境)
pip install -r requirements.txt

# 設定 Gemini API Key (請先申請 Google AI Studio API Key)
export GOOGLE_API_KEY="your_api_key_here"
```

## 🏃‍♂️ 快速開始

### 1. 音頻預處理

```bash
cd preprocess
python audio_prepare.py --input_dir "raw_audio/" --output_dir "processed_audio/"
```

### 2. Task 1: 語音識別

```bash
cd task1
# 設定音頻檔案路徑 (編輯 config.json)
python gemini_whis.py --input_dir "audio_files/" --task1_output "task1_output.txt"

# 或使用 Jupyter Notebook
jupyter notebook Whisperx.ipynb
```

### 3. Task 2: 敏感健康資訊識別

```bash
cd task2

# 設定儲存路徑和讀取路徑
config.json

#產生測試資料SHI在句子中的index位置
jupyter notebook generate_task2_test_data_index.ipynb

# 訓練模型
jupyter notebook NER_CRF_FGM_BIO.ipynb

# 預測結果
jupyter notebook predict_all.ipynb

# 插入timestamp到預測結果
jupyter notebook Insert_timestamp.ipynb
```

### 4. 數據集分割

```bash
cd preprocess
python split_and_check_k_hold_with_test.py
```

## 📊 實驗結果與效能

### Task 1 (語音識別)
- **核心模型**: WhisperX Large-v3
- **處理速度**: 10-20x 實時處理速度 (GPU)
- **語言支援**: 中文 (繁/簡)、自動語言檢測
- **輸出格式**: 
  - Task 1: 檔案名稱 + 轉錄文字
  - Task 2: 檔案名稱 + 實體類別 + 實體文字 + 時間戳

### Task 2 (敏感健康資訊識別)
#### 📊 NER 模型組合 F1-score 比較

| 編號 | 標記方式 | 使用技術              | F1-score | 訓練步數 |
|------|----------|------------------------|----------|----------|
| 1    | BIOU     | baseline               | 0.6856   | 812      |
| 2    | BIOU     | + FGM                  | 0.6698   | 1160     |
| 3    | BIOU     | + CRF                  | 0.7084   | 1276     |
| 4    | BIOU     | + focal loss           | 0.6791   | 1856     |
| 5    | BIOU     | + weight loss          | 0.6497   | 3248     |
| 6    | BIO      | baseline               | 0.6729   | 1044     |
| 7    | BIO      | + FGM                  | 0.6801   | 1624     |
| 8    | BIO      | + CRF                  | 0.7184   | 3016     |
| 9    | BIO      | + focal loss           | 0.7063   | 1508     |
| 10   | BIO      | + weight loss          | 0.6581   | 1856     |
| 11   | BIO      | + CRF + FGM            | **0.7256** | 2088     |

### 支援的敏感健康資訊類別 (SHI - 20種)
- **人物資訊**: PATIENT, DOCTOR, FAMILYNAME, PERSONALNAME
- **職業資訊**: PROFESSION
- **地點資訊**: ROOM, DEPARTMENT, HOSPITAL, STREET, CITY, DISTRICT, COUNTY, STATE, COUNTRY
- **時間資訊**: AGE, DATE, TIME, DURATION, SET
- **聯絡資訊**: PHONE

## 🔧 進階配置

### 音頻處理配置

```python
# 自定義音頻處理配置
config = ProcessingConfig(
    sr=22050,                    # 採樣率
    n_fft=2048,                  # FFT 窗口大小
    use_neural_enhancement=True,  # 啟用神經網絡增強
    use_mixed_precision=True,     # 混合精度計算
    max_batch_size=16            # 批次大小
)
```

### NER 模型配置

```python
# 訓練參數配置
training_args = TrainingArguments(
    output_dir="./ner_results",
    learning_rate=3e-5,
    num_train_epochs=40,
    per_device_train_batch_size=4,
    weight_decay=0.03
)
```

## 🔍 故障排除

### 常見問題

1. **CUDA 記憶體不足**
   ```bash
   # 解決方案：減少批次大小
   --per_device_batch_size=2
   # 或啟用梯度檢查點
   --gradient_checkpointing=True
   ```

2. **Gemini API 配置錯誤**
   ```bash
   # 確認 API Key 設定
   export GOOGLE_API_KEY="your_api_key_here"
   # 或在程式中設定
   genai.configure(api_key="your_api_key_here")
   ```

3. **音頻格式不支援**
   ```bash
   # 轉換為支援格式
   ffmpeg -i input.mp3 -ar 16000 output.wav
   ```

### 效能優化建議

- **使用 GPU 加速**: 確保安裝 CUDA 版本的 PyTorch
- **混合精度訓練**: 啟用 AMP 可節省 50% 記憶體
- **批次處理**: 增加批次大小可提升 GPU 利用率
- **模型量化**: 使用 INT8 量化減少推理時間

## 📈 評估指標

### Task 1 評估
- **字符錯誤率 (CER)**: 字符級別的識別錯誤率
- **詞錯誤率 (WER)**: 詞級別的識別錯誤率
- **處理速度**: 相對於實時的處理倍速

### Task 2 評估
- **精確率 (Precision)**: TP / (TP + FP)
- **召回率 (Recall)**: TP / (TP + FN)
- **F1 分數**: 精確率和召回率的調和平均
- **宏平均 F1**: 所有 SHI 類別 F1 的平均值

## 🔗 相關資源

### 官方文檔
- [AI Cup 2025 官網](https://www.aicup.tw/)
- [比賽規則與評分標準](https://www.codabench.org/competitions/4890/)

### 技術文檔
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large)
- [Ollama 官網](https://ollama.ai/)
- [TorchCRF](https://pytorch-crf.readthedocs.io/)

### 學術論文
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning](https://arxiv.org/abs/1911.02116)
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cis_papers/159/)

## 📝 更新日誌

### v2.0.0 (2025-06-12)
- ✨ 新增 Transformer 神經網絡音頻增強
- ✨ 整合 WhisperX + Ollama 完整語音識別流程
- ✨ 實現 CRF + FGM 對抗訓練 SHI 識別模型
- ✨ 加入心理聲學建模與 GPU 混合精度計算
- 🔧 優化數據集分割與 K-fold 交叉驗證
- 📚 完善技術文檔與使用說明

### v1.0.0 (2025-05-01)
- 🎉 專案初始版本
- ✨ 基礎音頻預處理功能
- ✨ Whisper 語音識別整合
- ✨ XLM-RoBERTa SHI 識別模型
- 📚 基礎說明文檔

## 授權

本專案依照 [GNU GPL v3](LICENSE) 條款授權。  
詳見 LICENSE 檔案以取得完整條款。
