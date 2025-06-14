# Task 2 - 命名實體識別與時間戳對齊 (NER + Timestamp Alignment)

## 📋 任務概述

Task 2 是 AI Cup 2025 的命名實體識別任務，主要目標是：
1. **命名實體識別 (NER)**: 從醫療轉錄文字中識別並標註醫療相關實體
2. **時間戳對齊**: 為識別出的實體提供精確的時間戳資訊
3. **多模型集成**: 使用多種深度學習模型進行高精度NER預測

## 🚀 主要功能

### 1. 命名實體識別 (NER)
- 使用 **XLM-RoBERTa Large** 作為基礎模型
- 加上CRF層和FGM訓練方式
- 支援 20 種醫療相關實體類別：
  - **人物**: PATIENT, DOCTOR, USERNAME, FAMILYNAME, PERSONALNAME
  - **職業**: PROFESSION
  - **地點**: ROOM, DEPARTMENT, HOSPITAL, ORGANIZATION, STREET, CITY, DISTRICT, COUNTY, STATE, COUNTRY, ZIP, LOCATION-OTHER
  - **時間**: AGE, DATE, TIME, DURATION, SET
  - **聯絡**: PHONE, FAX, EMAIL, URL, IPADDRESS
  - **識別**: SOCIAL_SECURITY_NUMBER, MEDICAL_RECORD_NUMBER, HEALTH_PLAN_NUMBER, ACCOUNT_NUMBER, LICENSE_NUMBER, VEHICLE_ID, DEVICE_ID, BIOMETRIC_ID, ID_NUMBER
  - **其他**: OTHER

### 2. 時間戳對齊
- 基於字符級別的精確時間戳計算
- 實體在語音中的起始和結束時間對齊
- 支援 word-level 和 character-level 對齊模式

### 3. 模型架構
- **基礎模型**: XLM-RoBERTa Large (FacebookAI/xlm-roberta-large-finetuned-conll03-english)
- **條件隨機場 (CRF)**: 改善序列標註一致性
- **對抗訓練 (FGM)**: 提升模型魯棒性

## 📁 檔案結構

```
task2/
├── README.md                                  # 本說明文件
├── NER_CRF_FGM_BIO.ipynb                      # CRF + FGM 訓練主程式
├── predict_all.ipynb                          # 模型預測
├── Insert_timestamp.ipynb                     # 時間戳對齊處理
├── generate_task2_test_data_index.ipynb       # 產生任務二在句子中的index位置
├── config.json                                # 路徑設定
```

## 🔧 環境設置

### 必要套件
```bash
pip install torch
pip install transformers
pip install torchcrf
pip install pytorch-crf
pip install datasets
pip install numpy
pip install pandas
pip install tqdm
```

### 模型要求
- GPU 記憶體: 建議 8GB 以上
- CUDA 支援: 建議 CUDA 11.0+

## 💻 使用方法

### 1. config.json 設定說明
config.json 用於集中管理專案中各項模型與資料的路徑設定，避免在程式碼中硬編路徑，讓專案更容易維護與部署。

####📁 設定範例

```json
{
  "huggingface_access_token":  "",
  "model_train_task1_data_path_txt": "",
  "model_train_task2_data_path_txt": "",
  "model_val_task1_data_path_txt": "",
  "model_val_task2_data_path_txt": "",
  "model_save_path": "./NER_model",
  "model_logging_dir":"./ner_logs",
  "answer_val_data_path_txt":"",


  "model_predict_all_result_path_txt": "",
  "whisper_timestamp_word_level_path_json": "",
  "whisper_timestamp_char_level_path_json": "",
  "char_level_timestamp_task2_NER_result_path_txt": "",
  "word_level_timestamp_task2_NER_result_path_txt": "",

  "model_checkpoint_1213" : "",
  "model_checkpoint_1000" : "",
  "model_checkpoint_500" : "",
  "model_test_task1_data_path_txt": "",
  "model_test_task2_data_path_txt": ""

}
```

####📝 欄位說明

```
欄位名稱	                                            說明
huggingface_access_token                            huggingface的access_token
model_train_task1_data_path_txt	                    任務一的訓練集路徑
model_train_task2_data_path_txt                     任務二的訓練集路徑
model_val_task1_data_path_txt                       任務一的驗證集路徑
model_val_task2_data_path_txt                       任務二的驗證集路徑
model_save_path                                     模型儲存資料夾路徑
model_logging_dir                                   模型log的儲存資料夾路徑
answer_val_data_path_txt                            驗證集有index位置的結果路徑

model_predict_all_result_path_txt                   模型預測的結果(有index位置)路徑
whisper_timestamp_word_level_path_json              whisperx的timestamp word_level路徑
whisper_timestamp_char_level_path_json              whisperx的timestamp char_level路徑
char_level_timestamp_task2_NER_result_path_txt      char level timestamp的模型預測結果路徑( 把index換成timestamp)
word_level_timestamp_task2_NER_result_path_txt      word level timestamp的模型預測結果路徑( 把index換成timestamp)

model_checkpoint_{自己命名}                          模型的參數路徑
model_test_task1_data_path_txt                      任務一的測試集路徑
model_test_task2_data_path_txt                      任務二的測試集路徑
```

### 2. 產生task2的SHI在句子中的index位置(可以當作測試) - generate_task2_test_data_index.ipynb

舉例:
```
2505	DURATION	12	22	10 minutes
2505	TIME	115	125	last night
2505	DATE	155	160	today
2505	FAMILYNAME	166	171	James
2943	DATE	225	231	Friday
2943	DATE	248	256	Saturday
2943	DATE	286	293	Tuesday
```

### 3. 模型訓練 - NER_CRF_FGM_BIO.ipynb

主要訓練腳本，整合了多種先進技術：

#### 主要特點：
- **CRF層**: 確保標註序列一致性
- **FGM對抗訓練**: 提升模型泛化能力
- **字符級評估**: 精確的效能評估
- **動態學習率**: 適應性訓練策略

#### 訓練配置：
```python
training_args = TrainingArguments(
    output_dir="./ner_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=40,
    weight_decay=0.03,
)
```

### 4. 模型預測 - predict_all.ipynb

多模型集成預測系統，支援多種模型架構的預測：

#### 支援的模型類型：
- **CRF模型**: `crf`, `crf_FGM`, `crf_BIOU`
- **Focal Loss模型**: `focal`, `focal_loss_BIOU`
- **階層式模型**: `level_1_loss_1`, `level_2_loss_2`
- **權重調整模型**: `weight_class`, `weight_class_BIOU`

#### 預測流程：
```python
# 載入模型
model = XLMRobertaWithCRF.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 執行預測
predictions = get_level2_entities_normal(model, tokenizer, text, label_map)
results = Process_Predict_Ner(predictions)
```

### 5. 時間戳對齊 - Insert_timestamp.ipynb

將 NER 結果與語音時間戳對齊：

#### 對齊方法：
```python
def align_ner_with_whisper(ner_entities, whisper_words):
    for ent in ner_entities:
        ent_start = ent["start"]
        ent_end = ent["end"]
        
        # 尋找重疊的詞彙
        matched_words = []
        for word in whisper_words:
            if not (word["char_end"] <= ent_start or word["char_start"] >= ent_end):
                matched_words.append(word)
        
        # 分配時間戳
        if matched_words:
            ts_start = matched_words[0]['start_time']
            ts_end = matched_words[-1]['end_time']
```

## ⚙️ 技術特點

### CRF (條件隨機場)
- **序列一致性**: 確保 BIO/BIOU 標註規則
- **全局優化**: 考慮整個序列的標註決策
- **轉移矩陣**: 學習標籤間的轉移機率

### FGM 對抗訓練
- **梯度擾動**: 在 embedding 層加入對抗噪聲
- **魯棒性提升**: 增強模型對輸入變化的穩定性
- **正則化效果**: 防止過擬合


## 📊 標註格式

### BIO 格式
- **B-XXX**: 實體開始
- **I-XXX**: 實體內部
- **O**: 非實體

### 範例
```
王小明     B-PATIENT
今天       O
下午       B-TIME
三點       I-TIME
```

## 🔍 評估指標

### 字符級評估
- **精確率 (Precision)**: TP / (TP + FP)
- **召回率 (Recall)**: TP / (TP + FN)
- **F1 分數**: 2 × (Precision × Recall) / (Precision + Recall)
- **宏平均 F1**: 所有類別 F1 的平均值

### 時間重疊評估
```python
def calculate_overlap(pred_start, pred_end, gt_start, gt_end):
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    return max(0, overlap_end - overlap_start)
```

## 🎯 模型效能

### 訓練策略
- **批次大小**: 4-8 (視 GPU 記憶體而定)
- **學習率**: 3e-5 (適合醫療領域微調)
- **訓練輪數**: 40 epochs
- **權重衰減**: 0.03 (防止過擬合)

### 優化技巧
- **梯度累積**: 模擬更大的批次大小
- **學習率調度**: 餘弦退火或線性衰減
- **早停機制**: 防止過擬合
- **模型集成**: 多個模型結果融合

## 🔧 故障排除

### 常見問題
1. **CUDA 記憶體不足**
   - 減少 batch_size
   - 使用梯度檢查點 (gradient checkpointing)
   - 嘗試混合精度訓練

2. **CRF 收斂問題**
   - 調整學習率
   - 檢查標籤對映正確性
   - 確認 mask 設置

3. **FGM 訓練不穩定**
   - 降低 epsilon 值
   - 調整對抗訓練頻率
   - 監控梯度範數

### 效能優化
- **模型量化**: 使用 INT8 量化減少記憶體
- **動態批次**: 根據序列長度調整批次大小
- **快取機制**: 預先計算 tokenizer 結果

## 📈 實驗結果

### 📊 NER 模型組合 F1-score 比較

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

## 🔗 相關連結

- [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large)
- [TorchCRF](https://pytorch-crf.readthedocs.io/)
- [Transformers](https://huggingface.co/transformers/)
- [AI Cup 2025 官網](https://aidea-web.tw/topic/cbea3c74-d86b-48c8-8c83-957b2e1374f2)
