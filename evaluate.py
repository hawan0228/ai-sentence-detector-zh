from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.baseline_model import BaselineModel
from src.bert_model import BertModel
from src.preprocessing import load_dataset, split_dataset, summarize_split
from src.text_analysis import LOWER_THRESHOLD, UPPER_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在固定 test set 評估模型")
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--my-data", type=Path, default=Path("data/my_data.csv"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("models/baseline_char_tfidf.joblib")
    )
    parser.add_argument("--bert", type=Path, default=Path("models/bert"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bert", action="store_true")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, ai_probabilities: np.ndarray, latency_ms: float) -> dict[str, Any]:
    predictions = (ai_probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "status": "available",
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, ai_probabilities)),
        "pr_auc": float(average_precision_score(y_true, ai_probabilities)),
        "confusion_matrix": matrix.tolist(),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "average_inference_ms": float(latency_ms),
    }


def evaluate_baseline(model: BaselineModel, frame: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    model.load()
    assert model.pipeline is not None
    classes = list(model.pipeline.named_steps["classifier"].classes_)
    started = perf_counter()
    probabilities = model.pipeline.predict_proba(frame["text"])
    elapsed_ms = (perf_counter() - started) * 1_000 / len(frame)
    ai_probabilities = probabilities[:, classes.index(1)]
    metrics = compute_metrics(frame["label"].to_numpy(), ai_probabilities, elapsed_ms)
    metrics["evaluation_scope"] = "independent_test_split"
    return metrics, ai_probabilities


def evaluate_bert(model: BertModel, frame: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray | None]:
    status = model.status()
    if not status["available"]:
        return {"status": "not_available", "reason": status["reason"]}, None
    probabilities, elapsed_ms = model.predict_many(frame["text"], batch_size=16)
    values = np.asarray(probabilities, dtype=float)
    metrics = compute_metrics(frame["label"].to_numpy(), values, elapsed_ms)
    metrics["evaluation_scope"] = "reference_re_evaluation_not_independent"

    return metrics, values


def load_single_class(path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not path.exists():
        return None, {"status": "not_available", "reason": f"找不到 {path}"}
    frame, cleaning = load_dataset(path)
    if set(frame["label"]) != {1}:
        return frame, {
            "status": "not_available",
            "reason": "my_data.csv 預期只包含 label=1。",
        }
    return frame, cleaning


def single_class_metrics(probabilities: np.ndarray, cleaning: dict[str, Any]) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "status": "available",
        "samples": len(probabilities),
        "cleaning": cleaning,
        "ai_recall_at_0_5": float(predicted.mean()),
        "mean_ai_probability": float(probabilities.mean()),
        "median_ai_probability": float(np.median(probabilities)),
        "uncertain_ratio": float(
            ((probabilities >= LOWER_THRESHOLD) & (probabilities <= UPPER_THRESHOLD)).mean()
        ),
        "human_likely_ratio": float((probabilities < LOWER_THRESHOLD).mean()),
        "ai_likely_ratio": float((probabilities > UPPER_THRESHOLD).mean()),
    }


def save_confusion_plot(metrics: dict[str, Any], output: Path) -> None:
    available = [(name, values) for name, values in metrics.items() if values.get("status") == "available"]
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
    for axis, (name, values) in zip(axes[0], available, strict=True):
        matrix = np.asarray(values["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(title=name, xlabel="Predicted label", ylabel="True label")
        axis.set_xticks([0, 1]); axis.set_yticks([0, 1])
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_roc_plot(y_true: np.ndarray, probabilities: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    for name, values in probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_true, values)
        area = roc_auc_score(y_true, values)
        axis.plot(false_positive, true_positive, label=f"{name} (AUC={area:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def length_buckets(lengths: np.ndarray) -> dict[str, int]:
    """Return aggregate counts without exposing source text."""
    return {
        "<50": int((lengths < 50).sum()),
        "50-99": int(((lengths >= 50) & (lengths < 100)).sum()),
        "100-199": int(((lengths >= 100) & (lengths < 200)).sum()),
        ">=200": int((lengths >= 200).sum()),
    }


def write_error_analysis(
    output: Path,
    test: pd.DataFrame,
    baseline_probabilities: np.ndarray,
    bert_metrics: dict[str, Any],
    bert_probabilities: np.ndarray | None,
    my_data_metrics: dict[str, Any],
) -> None:
    predictions = (baseline_probabilities >= 0.5).astype(int)
    truth = test["label"].to_numpy()
    false_positive = (truth == 0) & (predictions == 1)
    false_negative = (truth == 1) & (predictions == 0)
    wrong = predictions != truth
    lengths = test["text"].str.len().to_numpy()
    lines = [
        "# 錯誤分析",
        "",
        "本報告由 `evaluate.py` 依固定 test set 實際產生。為避免公開原始語料，僅保留彙總統計，不輸出任何文本片段。",
        "",
        "## Traditional",
        "",
        f"- False positives：{int(false_positive.sum())}",
        f"- False negatives：{int(false_negative.sum())}",
        f"- 判對文本平均長度：{lengths[~wrong].mean():.1f}",
        f"- 判錯文本平均長度：{lengths[wrong].mean():.1f}",
        f"- 判錯文本長度中位數：{np.median(lengths[wrong]):.1f}",
        "",
        "判錯文本長度分布：",
    ]
    for bucket, count in length_buckets(lengths[wrong]).items():
        lines.append(f"- {bucket} 字：{count}")
    lines.extend(
        [
            "",
            "## 觀察與限制",
            "",
            "- 少於 5 個中文字的輸入缺乏足夠線索，介面會保留結果但顯示短文本警告。",
            "- Character n-gram 比舊版 word analyzer 適合中文，但仍可能學到固定句型、換行、URL 或資料格式。",
            "- `my_data.csv` 是較口語的單一正類集合，只能檢查 AI recall 與分數分布，不能當作完整 accuracy。",
            f"- `my_data.csv` Traditional AI recall：{my_data_metrics.get('traditional', {}).get('ai_recall_at_0_5', 'not_available')}。",
            f"- `my_data.csv` BERT AI recall：{my_data_metrics.get('bert', {}).get('ai_recall_at_0_5', 'not_available')}。",
            f"- BERT 比較狀態：{bert_metrics.get('status')}。{bert_metrics.get('reason', '')}",
        ]
    )
    if bert_probabilities is not None:
        difference = np.abs(baseline_probabilities - bert_probabilities)
        lines.extend(
            [
                "",
                "## 兩模型分歧彙總",
                "",
                f"- 機率差大於 0.30：{int((difference > 0.30).sum())}",
                f"- 平均絕對機率差：{difference.mean():.4f}",
                f"- 機率差中位數：{np.median(difference):.4f}",
                f"- 最大機率差：{difference.max():.4f}",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    frame, cleaning = load_dataset(args.data)
    splits = split_dataset(frame, seed=args.seed)
    baseline = BaselineModel(args.baseline)
    baseline_metrics, baseline_probabilities = evaluate_baseline(baseline, splits.test)

    bert_model = BertModel(args.bert)
    if args.skip_bert:
        bert_metrics, bert_probabilities = {"status": "not_available", "reason": "使用 --skip-bert 略過。"}, None
    else:
        bert_metrics, bert_probabilities = evaluate_bert(bert_model, splits.test)

    my_frame, my_cleaning = load_single_class(args.my_data)
    if my_frame is None or my_cleaning.get("status") == "not_available":
        my_data_metrics = {"traditional": my_cleaning, "bert": my_cleaning}
    else:
        assert baseline.pipeline is not None
        classes = list(baseline.pipeline.named_steps["classifier"].classes_)
        baseline_my_probabilities = baseline.pipeline.predict_proba(my_frame["text"])[:, classes.index(1)]
        my_data_metrics = {
            "traditional": single_class_metrics(baseline_my_probabilities, my_cleaning)
        }
        if bert_metrics["status"] == "available":
            bert_my_values, _ = bert_model.predict_many(my_frame["text"], batch_size=16)
            my_data_metrics["bert"] = single_class_metrics(np.asarray(bert_my_values), my_cleaning)
        else:
            my_data_metrics["bert"] = {
                "status": "not_available",
                "reason": bert_metrics.get("reason", "BERT 不可用。"),
            }

    model_metrics = {"traditional": baseline_metrics, "bert": bert_metrics}
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "label_mapping": {"0": "human", "1": "ai"},
        "thresholds": {
            "lower": LOWER_THRESHOLD,
            "upper": UPPER_THRESHOLD,
            "method": "conservative_heuristic",
        },
        "data_cleaning": cleaning,
        "splits": {
            name: summarize_split(getattr(splits, name))
            for name in ("train", "validation", "test")
        },
        "models": model_metrics,
        "my_data": my_data_metrics,
    }
    (args.reports / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for name, values in model_metrics.items():
        row = {
            "model": name,
            "status": values["status"],
            "evaluation_scope": values.get("evaluation_scope", "not_available"),
        }
        if values["status"] == "available":
            row.update({key: values[key] for key in ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate", "average_inference_ms")})
        else:
            row["reason"] = values.get("reason", "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.reports / "model_comparison.csv", index=False)
    save_confusion_plot(model_metrics, args.reports / "confusion_matrix.png")
    probability_sets = {"Traditional": baseline_probabilities}
    if bert_probabilities is not None:
        probability_sets["BERT"] = bert_probabilities
    save_roc_plot(splits.test["label"].to_numpy(), probability_sets, args.reports / "roc_curve.png")
    write_error_analysis(
        args.reports / "error_analysis.md",
        splits.test,
        baseline_probabilities,
        bert_metrics,
        bert_probabilities,
        my_data_metrics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"評估報告已寫入：{args.reports}")


if __name__ == "__main__":
    main()
