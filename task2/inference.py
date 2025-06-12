# predict.py
import argparse, re, json, sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import (
    LukeForTokenClassification,
    LukeTokenizer,
    RobertaTokenizerFast,
    pipeline,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=r"",
                   help="訓練好模型或 HF checkpoint 目錄")
    p.add_argument("--input_file", default=r"",
                   help="來源檔，每行: <ID><tab><text>")
    p.add_argument("--output_file", default=r"",
                   help="輸出路徑")
    p.add_argument("--device", type=int, default=0,
                   help="-1=CPU, 0/1...=CUDA 編號")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu" if args.device == -1 or not torch.cuda.is_available()
                          else f"cuda:{args.device}")

    # --- 載入 tokenizer / model -------------------------------------------------
    tok = RobertaTokenizerFast.from_pretrained(args.model_dir)
    model = LukeForTokenClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tok,
        aggregation_strategy="first",
        device=model.device.index if model.device.type == "cuda" else -1,
    )

    # --- 讀資料並預測 ----------------------------------------------------------
    out_lines = []
    with Path(args.input_file).open() as f:
        rows = [line.rstrip("\n") for line in f if line.strip()]

    for row in tqdm(rows, desc="Predict"):
        try:
            sent_id, text = row.split("\t", 1)
        except ValueError:
            sent_id, text = "UNK", row

        preds = ner(text)

        for ent in preds:
            span = text[ent["start"]: ent["end"]]
            label = ent["entity_group"]
            start = ent["start"]
            end = ent["end"]
            #輸出包含字元 index（start, end）
            out_lines.append(f"{sent_id}\t{label}\t{start}\t{end}\t{span}")

    # --- 輸出 -----------------------------------------------------------------
    Path(args.output_file).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"寫入完成 -> {args.output_file}（共 {len(out_lines)} 條）")

if __name__ == "__main__":
    main()
