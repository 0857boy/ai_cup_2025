# AI Cup 2025 - 醫療語音識別與命名實體識別競賽

## 🏆 競賽概述

本專案為 AI Cup 2025 醫療語音識別與命名實體識別競賽的完整解決方案，涵蓋語音轉文字 (ASR)、命名實體識別 (NER) 以及時間戳對齊等多項先進技術。

### 🎯 任務目標

- **Task 1**: 語音識別 (ASR) - 將醫療語音檔案轉換為文字並輸出轉錄結果
- **Task 2**: 命名實體識別 (NER) - 從轉錄文字中識別醫療相關實體並提供時間戳對齊

### 🏅 主要成果

- **語音識別**: 使用 WhisperX Large-v3 + Ollama Qwen3 實現高精度 ASR
- **實體識別**: 基於 XLM-RoBERTa + CRF + FGM 對抗訓練達到優異效能
- **音頻處理**: 先進的 Transformer 神經網絡音頻增強系統
- **數據管理**: 智能 K-fold 交叉驗證與數據集分割工具

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
│   ├── ollama_qwen_whis.py      # 主要處理腳本 (WhisperX + Ollama)
│   ├── whisper_large_v3.ipynb   # Whisper 基礎版本
│   ├── Whisperx.ipynb           # WhisperX 進階版本
│   └── README.md                # Task 1 詳細說明
└── task2/                       # Task 2: 命名實體識別
    ├── NER_CRF_FGM_BIO.ipynb    # CRF + FGM 訓練主程式
    ├── predict_all.ipynb        # 多模型預測與集成
    ├── Insert_timestamp.ipynb   # 時間戳對齊處理
    ├── inference.py             # 推理腳本
    └── README.md                # Task 2 詳細說明
```

## 🚀 核心技術特點

### 🎵 音頻預處理 (preprocess/)
- **Transformer 神經網絡增強**: 基於深度學習的音頻降噪與增強
- **心理聲學建模**: 基於人類聽覺感知的智能音頻處理
- **GPU 混合精度計算**: 支援 CUDA 加速，大幅提升處理速度
- **K-fold 數據分割**: 使用 MultilabelStratifiedKFold 確保標籤分布平衡

### 🗣️ 語音識別 (task1/)
- **WhisperX Large-v3**: 業界領先的語音識別模型
- **Ollama Qwen3**: 本地部署的中文 NER 模型
- **字符級時間戳**: 精確到字符級別的時間對齊
- **簡繁轉換**: 自動處理繁簡體中文轉換

### 🏷️ 命名實體識別 (task2/)
- **XLM-RoBERTa Large**: 多語言預訓練模型
- **CRF (條件隨機場)**: 確保序列標註一致性
- **FGM 對抗訓練**: 提升模型魯棒性
- **Focal Loss**: 處理類別不平衡問題
- **階層式分類**: Level 1 + Level 2 雙層分類架構

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

# 數據處理
pip install iterstrat  # 多標籤分層分割

# Ollama (本地 LLM)
# 請至 https://ollama.ai 下載並安裝
ollama pull qwen3:8b
```

### 快速安裝

```bash
# 克隆專案
git clone https://github.com/your-repo/ai_cup_2025.git
cd ai_cup_2025

# 安裝依賴 (建議使用虛擬環境)
pip install -r requirements.txt

# 安裝 Ollama 模型
ollama pull qwen3:8b
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
python ollama_qwen_whis.py --input_dir "audio_files/" --task1_output "asr_results.txt"
```

### 3. Task 2: 命名實體識別

```bash
cd task2
# 訓練模型
jupyter notebook NER_CRF_FGM_BIO.ipynb

# 預測結果
python inference.py --model_dir "trained_model/" --input_file "asr_results.txt" --output_file "ner_results.txt"
```

### 4. 數據集分割

```bash
cd preprocess
python split_and_check_k_hold_with_test.py
```

## 📊 實驗結果與效能

### Task 1 (語音識別)
- **模型**: WhisperX Large-v3 + Ollama Qwen3
- **處理速度**: 10-20x 實時處理速度 (GPU)
- **語言支援**: 中文 (繁/簡)、自動語言檢測
- **輸出格式**: 檔案名稱 + 轉錄文字

### Task 2 (命名實體識別)
| 模型架構 | Macro F1 | 訓練時間 | GPU 記憶體 |
|----------|----------|----------|------------|
| XLM-RoBERTa | 0.8520 | 2h | 6GB |
| + CRF | 0.8687 | 2.5h | 7GB |
| + FGM | 0.8756 | 3h | 7GB |
| + Focal Loss | 0.8698 | 2.8h | 7GB |

### 支援的實體類別 (20種)
- **人物**: PATIENT, DOCTOR, FAMILYNAME, PERSONALNAME
- **職業**: PROFESSION
- **地點**: ROOM, DEPARTMENT, HOSPITAL, STREET, CITY, DISTRICT, COUNTY, STATE, COUNTRY
- **時間**: AGE, DATE, TIME, DURATION, SET
- **聯絡**: PHONE

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

2. **Ollama 連線失敗**
   ```bash
   # 確認 Ollama 服務運行
   ollama list
   ollama run qwen3:8b
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
- **宏平均 F1**: 所有實體類別 F1 的平均值

## 🔗 相關資源

### 官方文檔
- [AI Cup 2025 官網](https://aidea-web.tw/)
- [比賽規則與評分標準](https://aidea-web.tw/topic/cbea3c74-d86b-48c8-8c83-957b2e1374f2)

### 技術文檔
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large)
- [Ollama 官網](https://ollama.ai/)
- [TorchCRF](https://pytorch-crf.readthedocs.io/)

### 學術論文
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning](https://arxiv.org/abs/1911.02116)
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cis_papers/159/)

## 👥 貢獻指南

我們歡迎社群貢獻！請參考以下步驟：

1. **Fork 專案**: 建立自己的專案分支
2. **建立功能分支**: `git checkout -b feature/amazing-feature`
3. **提交變更**: `git commit -m 'Add amazing feature'`
4. **推送分支**: `git push origin feature/amazing-feature`
5. **建立 Pull Request**: 詳細描述變更內容

### 開發規範
- 遵循 PEP 8 Python 程式碼風格
- 新功能需附帶測試案例
- 更新相應的文檔說明
- 提交訊息使用英文並描述清楚

## 📝 更新日誌

### v2.0.0 (2025-06-12)
- ✨ 新增 Transformer 神經網絡音頻增強
- ✨ 整合 WhisperX + Ollama 完整語音識別流程
- ✨ 實現 CRF + FGM 對抗訓練 NER 模型
- ✨ 加入心理聲學建模與 GPU 混合精度計算
- 🔧 優化數據集分割與 K-fold 交叉驗證
- 📚 完善技術文檔與使用說明

### v1.0.0 (2025-05-01)
- 🎉 專案初始版本
- ✨ 基礎音頻預處理功能
- ✨ Whisper 語音識別整合
- ✨ XLM-RoBERTa NER 模型
- 📚 基礎說明文檔

## 📄 授權條款

本專案採用 MIT 授權條款。詳見 [LICENSE](LICENSE) 檔案。

## 🙏 致謝

感謝以下開源專案與研究團隊：

- **OpenAI**: Whisper 語音識別模型
- **Hugging Face**: Transformers 與 XLM-RoBERTa 模型
- **Facebook AI**: XLM-RoBERTa 預訓練模型
- **PyTorch 團隊**: 深度學習框架支援
- **Ollama**: 本地 LLM 部署解決方案

## 📧 聯絡資訊

- **專案維護**: AI Cup 2025 團隊
- **問題回報**: GitHub Issues
- **技術討論**: GitHub Discussions
- **電子郵件**: [your-email@domain.com]

---

<div align="center">

**🏆 AI Cup 2025 - 推動醫療 AI 技術發展 🚀**

[⭐ 給我們一顆星](https://github.com/your-repo/ai_cup_2025) | [📚 查看文檔](https://github.com/your-repo/ai_cup_2025/wiki) | [🐛 回報問題](https://github.com/your-repo/ai_cup_2025/issues)

</div>