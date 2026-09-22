from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def normalize_text(text: object) -> str:
    """做保守的 Unicode 與空白正規化，保留換行和標點。"""
    if text is None or pd.isna(text):
        return ""
    value = unicodedata.normalize("NFC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[^\S\n]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔：{path}")

    raw = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"資料缺少必要欄位：{', '.join(sorted(missing))}")

    frame = raw.loc[:, ["text", "label"]].copy()
    input_rows = len(frame)
    frame["text"] = frame["text"].map(normalize_text)
    empty_rows = int(frame["text"].eq("").sum())
    frame = frame.loc[frame["text"].ne("")].copy()

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError("label 欄位包含無法轉為整數的值。")
    frame["label"] = numeric_labels.astype(int)
    invalid = sorted(set(frame["label"]) - {0, 1})
    if invalid:
        raise ValueError(f"label 僅能是 0 或 1，目前包含：{invalid}")

    conflicts = (
        frame.groupby("text", sort=False)["label"].nunique().loc[lambda values: values > 1]
    )
    if not conflicts.empty:
        raise ValueError(f"發現 {len(conflicts)} 段相同文字具有衝突標籤。")

    duplicate_rows = int(frame.duplicated(subset="text").sum())
    frame = frame.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    report = {
        "input_rows": input_rows,
        "empty_rows_removed": empty_rows,
        "duplicate_rows_removed": duplicate_rows,
        "conflicting_texts": 0,
        "output_rows": len(frame),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
    }
    return frame, report


def split_dataset(frame: pd.DataFrame, seed: int = 42) -> DatasetSplits:
    """建立 70/15/15 的分層切分；輸入應已依正規化文字去重。"""
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=seed,
        stratify=frame["label"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder["label"],
    )
    result = DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
    assert_no_text_overlap(result)
    return result


def assert_no_text_overlap(splits: DatasetSplits) -> None:
    sets = {
        "train": set(splits.train["text"]),
        "validation": set(splits.validation["text"]),
        "test": set(splits.test["text"]),
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {f"{left}/{right}": len(sets[left] & sets[right]) for left, right in pairs}
    if any(overlaps.values()):
        raise ValueError(f"資料切分出現重複文本：{overlaps}")


def summarize_split(frame: pd.DataFrame) -> dict[str, object]:
    counts = frame["label"].value_counts().sort_index()
    return {
        "rows": len(frame),
        "labels": {
            str(label): {
                "count": int(count),
                "ratio": round(float(count / len(frame)), 6),
            }
            for label, count in counts.items()
        },
    }
