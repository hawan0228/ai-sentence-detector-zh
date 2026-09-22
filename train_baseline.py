from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report

from src.baseline_model import save_baseline, train_baseline
from src.preprocessing import load_dataset, split_dataset, summarize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練中文 character TF-IDF baseline")
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/baseline_char_tfidf.joblib")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, cleaning = load_dataset(args.data)
    splits = split_dataset(frame, seed=args.seed)

    print("資料清理：")
    print(json.dumps(cleaning, ensure_ascii=False, indent=2))
    print("資料切分：")
    for name in ("train", "validation", "test"):
        print(f"{name}: {json.dumps(summarize_split(getattr(splits, name)), ensure_ascii=False)}")

    model = train_baseline(splits.train["text"], splits.train["label"], seed=args.seed)
    save_baseline(model, args.output)

    validation_predictions = model.predict(splits.validation["text"])
    print("Validation 結果：")
    print(classification_report(splits.validation["label"], validation_predictions, digits=4))
    print(f"模型已儲存：{args.output}")
    print("Test set 未在訓練階段評估；請執行 python evaluate.py。")


if __name__ == "__main__":
    main()
