import os
import shutil
import json
import pandas as pd
import numpy as np
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def split_and_check():
    # 使用者輸入
    audio_dir   = input("請輸入包含所有 wav 音檔的資料夾路徑: ").strip()
    task1_path  = input("請輸入 task1_answer.txt 檔案路徑: ").strip()
    task2_path  = input("請輸入 task2_answer.txt 檔案路徑: ").strip()
    K           = int(input("請輸入要切分幾個 fold (K): ").strip())
    output_dir  = input("請輸入輸出資料夾路徑：").strip()
    os.makedirs(output_dir, exist_ok=True)

    # 讀入資料
    df1 = pd.read_csv(task1_path, sep='\t', header=None, dtype=str)
    df2 = pd.read_csv(task2_path, sep='\t', header=None,
                      names=['utt_id','label','start','end','text'], dtype=str)

    # 切出 test set
    multi = (
        df2.groupby(['utt_id', 'label'])
           .size()
           .unstack(fill_value=0)
           .gt(0)
           .astype(int)
    )
    test_utts = set()
    shuffled = multi.sample(frac=1.0, random_state=42)
    for utt_id, row in shuffled.iterrows():
        test_utts.add(utt_id)
        current_labels = multi.loc[list(test_utts)].sum().gt(0)
        if current_labels.all():
            break
    min_test_size = max(int(len(multi) * 0.1), len(test_utts))
    if len(test_utts) < min_test_size:
        additional = set(shuffled.index) - test_utts
        test_utts.update(list(additional)[:min_test_size - len(test_utts)])
    train_utts = list(set(multi.index) - test_utts)
    test_utts = list(test_utts)

    df1_test = df1[df1[0].isin(test_utts)]
    df2_test = df2[df2['utt_id'].isin(test_utts)]
    df1 = df1[df1[0].isin(train_utts)]
    df2 = df2[df2['utt_id'].isin(train_utts)]

    # 輸出 test set
    test_dir = os.path.join(output_dir, "test_set")
    test_audio_dir = os.path.join(test_dir, "audio")
    os.makedirs(test_audio_dir, exist_ok=True)
    df1_test.to_csv(os.path.join(test_dir, "task1_answer.txt"), sep='\t', header=False, index=False)
    df2_test.to_csv(os.path.join(test_dir, "task2_answer.txt"), sep='\t', header=False, index=False)
    for utt in test_utts:
        src = os.path.join(audio_dir, f"{utt}.wav")
        dst = os.path.join(test_audio_dir, f"{utt}.wav")
        if os.path.isfile(src): shutil.copy(src, dst)
    with open(os.path.join(test_dir, 'wav_list.json'), 'w', encoding='utf-8') as jf:
        json.dump([f"{utt}.wav" for utt in test_utts], jf, ensure_ascii=False, indent=2)

    # 建立 K-fold
    multi_k = (
        df2.groupby(['utt_id','label'])
           .size()
           .unstack(fill_value=0)
           .gt(0)
           .astype(int)
    )
    mskf = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    folds = []
    for fold, (_, val_idx) in enumerate(mskf.split(multi_k.index, multi_k.values), start=1):
        for utt in multi_k.index[val_idx]:
            folds.append({'utt_id': utt, 'fold': fold})
    fold_df = pd.DataFrame(folds)

    df1.columns = ['utt_id', 'rest']
    df1 = df1.merge(fold_df, on='utt_id', how='inner')
    df2 = df2.merge(fold_df, on='utt_id', how='inner')

    for fold in range(1, K+1):
        hold_dir = os.path.join(output_dir, f"hold_{fold}")
        audio_out = os.path.join(hold_dir, "audio")
        os.makedirs(audio_out, exist_ok=True)
        utts = fold_df[fold_df['fold']==fold]['utt_id'].tolist()
        for utt in utts:
            src = os.path.join(audio_dir, f"{utt}.wav")
            dst = os.path.join(audio_out, f"{utt}.wav")
            if os.path.isfile(src): shutil.copy(src, dst)
        df1[df1['fold']==fold][['utt_id', 'rest']]             .to_csv(os.path.join(hold_dir, "task1_answer.txt"), sep='\t', header=False, index=False)
        df2[df2['fold']==fold][['utt_id','label','start','end','text']]             .to_csv(os.path.join(hold_dir, "task2_answer.txt"), sep='\t', header=False, index=False)
        with open(os.path.join(hold_dir, "wav_list.json"), 'w', encoding='utf-8') as jf:
            json.dump([f"{utt}.wav" for utt in utts], jf, ensure_ascii=False, indent=2)

    # 驗證並輸出到 CSV
    all_stats = []
    for fold in range(1, K+1):
        df_fold = df2[df2['fold'] == fold]
        counts = Counter(df_fold["label"])
        for label in counts:
            all_stats.append({'split': f"hold_{fold}", 'label': label, 'count': counts[label]})
    test_counts = Counter(df2_test["label"])
    for label in test_counts:
        all_stats.append({'split': 'test_set', 'label': label, 'count': test_counts[label]})

    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.pivot(index='label', columns='split', values='count').fillna(0).astype(int)
    stats_df.to_csv(os.path.join(output_dir, "label_distribution_summary.csv"))
    print("已儲存標籤分布統計至 label_distribution_summary.csv")

if __name__ == "__main__":
    split_and_check()
