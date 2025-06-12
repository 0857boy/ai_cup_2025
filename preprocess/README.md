# AI Cup 2025 - 音頻預處理模組

## 模組概述

本模組提供了 AI Cup 2025 比賽中音頻數據預處理的完整解決方案，包含先進的音頻增強、降噪處理，以及數據集管理工具。

## 功能特點

### 🎵 先進音頻處理系統 (`audio_prepare.py`)

這是一個基於深度學習的高級音頻處理系統，具備以下特點：

- **🧠 Transformer 神經網絡增強**: 使用基於 Transformer 架構的深度學習模型進行音頻降噪和增強
- **🎧 心理聲學建模**: 基於人類聽覺感知的智能音頻處理
- **⚡ GPU 混合精度計算**: 支援 CUDA 加速和混合精度計算，大幅提升處理速度
- **🎛️ 自適應參數調整**: 根據音頻特性動態調整處理參數
- **📊 實時品質監控**: 內建音質評估系統，實時監控處理效果
- **🔄 並行批次處理**: 支援多進程並行處理，提高處理效率
- **🎵 保真度音頻重建**: 確保處理後音頻保持高保真度

#### 核心技術

1. **頻譜分析與增強**
   - FFT 窗口: 2048 (高頻率解析度)
   - Mel 濾波器: 128 個
   - 自適應頻譜遮罩

2. **神經網絡架構**
   - Transformer Encoder: 6 層，8 個注意力頭
   - 位置編碼與 GELU 激活函數
   - 混合精度訓練支援

3. **音頻增強技術**
   - 多頻段動態壓縮
   - 軟限制器
   - 智能正規化
   - 心理聲學濾波

### 🚀 輕量級音頻處理 (`audio_prepare(task1).py`)

針對 Task1 優化的簡化版音頻處理器：

- **📏 音頻標準化**: 自動音量正規化
- **🔇 靜音處理**: 智能靜音音量降低
- **🎯 頻譜降噪**: 動態調整的頻譜減法降噪
- **⚡ 快速處理**: 優化的處理流程，適合大批量音頻

#### 處理配置

```python
config = {
    'normalize': True,              # 啟用音頻正規化
    'reduce_silence': True,         # 降低靜音音量
    'silence_threshold': -60,       # 靜音閾值 (dB)
    'silence_reduce_factor': 0.1,   # 靜音音量減少係數
    'denoise': True,                # 啟用降噪
    'denoise_strength': 0.5,        # 降噪強度
    'sr': 16000                     # 目標採樣率
}
```

### 📊 數據集分割工具 (`split_and_check_k_hold_with_test.py`)

智能數據集管理工具，支援：

- **🔄 K-fold 交叉驗證**: 使用 MultilabelStratifiedKFold 確保標籤分布平衡
- **🧪 測試集分離**: 自動分離出測試集，確保所有標籤類別都有覆蓋
- **📋 標籤分布分析**: 生成詳細的標籤分布統計報告
- **📁 自動文件管理**: 自動複製音頻文件到對應目錄

#### 分割策略

1. **測試集生成**
   - 確保所有標籤類別都在測試集中有代表
   - 最小測試集大小為總數據的 10%
   - 隨機採樣保證分布平衡

2. **K-fold 分割**
   - 使用分層採樣保持標籤分布
   - 支援多標籤分類場景
   - 生成平衡的訓練/驗證分割

## 安裝依賴

```bash
pip install torch torchaudio librosa soundfile
pip install scipy scikit-learn pandas numpy
pip install tqdm psutil GPUtil
pip install iterstrat  # 用於多標籤分層分割
```

## 使用方法

### 1. 先進音頻處理

```python
from audio_prepare import AdvancedAudioProcessor, create_optimal_config

# 創建最優配置
config = create_optimal_config()

# 初始化處理器
processor = AdvancedAudioProcessor(config)

# 處理單個文件
processor.process_single_file('input.wav', 'output.wav')

# 批量處理
processor.process_dataset_parallel('input_dir/', 'output_dir/')
```

### 2. 輕量級音頻處理 (Task1)

```python
from audio_prepare_task1 import AudioProcessor

# 初始化處理器
processor = AudioProcessor()

# 處理數據集
processor.process_dataset('input_audio_dir/', 'output_audio_dir/')
```

### 3. 數據集分割

```bash
python split_and_check_k_hold_with_test.py
```

按提示輸入：
- 音頻文件目錄路徑
- task1_answer.txt 路徑
- task2_answer.txt 路徑  
- K-fold 數量
- 輸出目錄路徑

## 輸出結構

### 音頻處理輸出

```
processed_audio_v2/
├── train1/                 # 訓練集1處理結果
├── train2/                 # 訓練集2處理結果
├── validation/             # 驗證集處理結果
├── processing_report.json  # 處理報告
└── global_processing_stats.json  # 全局統計
```

### 數據集分割輸出

```
output_dir/
├── test_set/               # 測試集
│   ├── audio/             # 測試音頻文件
│   ├── task1_answer.txt   # Task1 答案
│   ├── task2_answer.txt   # Task2 答案
│   └── wav_list.json      # 音頻文件列表
├── hold_1/                # 第1折驗證集
├── hold_2/                # 第2折驗證集
├── ...
├── hold_K/                # 第K折驗證集
└── label_distribution_summary.csv  # 標籤分布統計
```

## 性能指標

### 先進音頻處理系統

- **處理速度**: 10-20x 實時處理速度 (GPU)
- **音質分數**: 平均 0.8+ (滿分 1.0)
- **支援格式**: WAV, MP3, FLAC, M4A, AAC, OGG
- **內存優化**: 支援大批量文件處理
- **GPU 加速**: 支援 CUDA 和混合精度計算

### 輕量級處理系統

- **處理速度**: 5-10x 實時處理速度
- **內存佔用**: 低內存模式，適合大規模數據
- **處理穩定性**: 針對語音數據優化的參數

## 配置說明

### ProcessingConfig 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `sr` | 22050 | 目標採樣率 |
| `n_fft` | 2048 | FFT 窗口大小 |
| `hop_length` | 512 | 跳躍長度 |
| `n_mels` | 128 | Mel 濾波器數量 |
| `use_neural_enhancement` | True | 啟用神經網絡增強 |
| `use_mixed_precision` | True | 啟用混合精度計算 |
| `target_lufs` | -23.0 | 目標響度 (LUFS) |
| `max_peak` | -1.0 | 最大峰值 (dB) |

## 日誌與監控

系統會自動生成詳細的處理日誌：

- **處理進度**: 實時顯示處理進度和速度
- **品質監控**: 每個文件的品質分數評估
- **性能統計**: 處理速度、GPU 使用率統計
- **錯誤報告**: 詳細的錯誤信息和失敗文件記錄

## 故障排除

### 常見問題

1. **GPU 內存不足**
   - 減少 `max_batch_size`
   - 啟用 `memory_efficient_mode`

2. **處理速度慢**
   - 確認 GPU 驅動程式正常
   - 檢查 CUDA 版本兼容性

3. **音質下降**
   - 調整 `denoise_strength`
   - 關閉過度的音頻增強選項

## 技術支援

- 🐛 問題回報: 請在 GitHub Issues 中提出
- 📖 技術文檔: 詳見各模組內的 docstring
- 🔧 自定義配置: 參考 `ProcessingConfig` 類別

## 更新日誌

### v2.0
- 新增 Transformer 神經網絡增強
- 支援心理聲學建模
- GPU 混合精度計算優化
- 實時品質評估系統

### v1.0
- 基礎音頻預處理功能
- K-fold 數據集分割
- 頻譜降噪處理