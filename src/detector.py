from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .baseline_model import BaselineModel
from .bert_model import BertModel
from .preprocessing import normalize_text
from .text_analysis import (
    LOWER_THRESHOLD,
    UPPER_THRESHOLD,
    classify_probability,
    is_analyzable,
    short_text_warning,
    split_sentences,
    text_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "models" / "baseline_char_tfidf.joblib"
DEFAULT_BERT_PATH = PROJECT_ROOT / "models" / "bert"


class TextDetector:
    def __init__(
        self,
        baseline_path: str | Path = DEFAULT_BASELINE_PATH,
        bert_path: str | Path = DEFAULT_BERT_PATH,
        lower_threshold: float = LOWER_THRESHOLD,
        upper_threshold: float = UPPER_THRESHOLD,
    ):
        self.baseline = BaselineModel(baseline_path)
        self.bert = BertModel(bert_path)
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def _format_prediction(
        self,
        prediction: dict[str, Any],
        model_key: str,
        model_name: str,
        text: str,
        include_sentences: bool,
    ) -> dict[str, Any]:
        label, label_zh = classify_probability(
            prediction["ai_probability"], self.lower_threshold, self.upper_threshold
        )
        warnings = list(prediction.get("warnings", []))
        short_warning = short_text_warning(text)
        if short_warning:
            warnings.append(short_warning)

        sentences = []
        if include_sentences:
            predictor = self.baseline if model_key == "baseline" else self.bert
            for sentence in split_sentences(text):
                sentence_prediction = predictor.predict(sentence)
                sentence_label, sentence_label_zh = classify_probability(
                    sentence_prediction["ai_probability"],
                    self.lower_threshold,
                    self.upper_threshold,
                )
                sentence_warnings = list(sentence_prediction.get("warnings", []))
                sentence_short = short_text_warning(sentence)
                if sentence_short:
                    sentence_warnings.append(sentence_short)
                sentences.append(
                    {
                        "sentence": sentence,
                        "ai_probability": sentence_prediction["ai_probability"],
                        "human_probability": sentence_prediction["human_probability"],
                        "label": sentence_label,
                        "label_zh": sentence_label_zh,
                        "model": model_key,
                        "warnings": sentence_warnings,
                    }
                )

        return {
            "available": True,
            "label": label,
            "label_zh": label_zh,
            "ai_probability": prediction["ai_probability"],
            "human_probability": prediction["human_probability"],
            "predicted_class": prediction["predicted_class"],
            "model": model_key,
            "model_name": model_name,
            "latency_ms": prediction["latency_ms"],
            "warnings": warnings,
            "sentences": sentences,
        }

    def analyze_model(
        self, text: object, model: str, include_sentences: bool = True
    ) -> dict[str, Any]:
        value = normalize_text(text)
        if not value:
            raise ValueError("請輸入要分析的中文文字。")
        if not is_analyzable(value):
            raise ValueError("輸入只有空白或標點，沒有可分析的文字內容。")

        if model == "baseline":
            prediction = self.baseline.predict(value)
            result = self._format_prediction(
                prediction,
                "baseline",
                self.baseline.display_name,
                value,
                include_sentences,
            )
            result["features"] = self.baseline.explain(value)
            return result
        if model == "bert":
            prediction = self.bert.predict(value)
            result = self._format_prediction(
                prediction, "bert", self.bert.display_name, value, include_sentences
            )
            result["features"] = None
            return result
        raise ValueError(f"不支援的模型：{model}")

    def analyze(self, text: object, mode: str = "compare") -> dict[str, Any]:
        value = normalize_text(text)
        if not value:
            raise ValueError("請輸入要分析的中文文字。")
        if not is_analyzable(value):
            raise ValueError("輸入只有空白或標點，沒有可分析的文字內容。")

        if mode in {"baseline", "bert"}:
            result = self.analyze_model(value, mode)
            result["statistics"] = text_statistics(value)
            result["comparison"] = [self._comparison_row(result)]
            return result
        if mode != "compare":
            raise ValueError(f"不支援的分析模式：{mode}")

        started = perf_counter()
        baseline_result = self.analyze_model(value, "baseline")
        comparison = [self._comparison_row(baseline_result)]
        warnings = list(baseline_result["warnings"])
        bert_result: dict[str, Any] | None = None
        try:
            bert_result = self.analyze_model(value, "bert")
            comparison.append(self._comparison_row(bert_result))
            difference = abs(
                baseline_result["ai_probability"] - bert_result["ai_probability"]
            )
            if difference > 0.30:
                warnings.append("兩個模型的判定差異較大，建議將結果視為不確定。")
        except Exception as error:
            comparison.append(
                {
                    "model": self.bert.display_name,
                    "ai_probability": None,
                    "label_zh": "not_available",
                    "latency_ms": None,
                    "status": str(error),
                }
            )
            warnings.append(str(error))

        if bert_result is None:
            label = baseline_result["label"]
            label_zh = baseline_result["label_zh"]
        else:
            label = "compare"
            label_zh = "請分別參考兩個模型結果"

        return {
            "available": True,
            "label": label,
            "label_zh": label_zh,
            "ai_probability": None,
            "human_probability": None,
            "model": "compare",
            "model_name": "模型比較",
            "latency_ms": (perf_counter() - started) * 1_000,
            "warnings": warnings,
            "sentences": baseline_result["sentences"],
            "features": baseline_result["features"],
            "statistics": text_statistics(value),
            "comparison": comparison,
            "results": {"baseline": baseline_result, "bert": bert_result},
        }

    @staticmethod
    def _comparison_row(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": result["model_name"],
            "ai_probability": result["ai_probability"],
            "label_zh": result["label_zh"],
            "latency_ms": result["latency_ms"],
            "status": "available",
        }
