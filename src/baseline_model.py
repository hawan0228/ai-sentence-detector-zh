from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import normalize_text


def build_baseline(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def train_baseline(texts, labels, seed: int = 42) -> Pipeline:
    model = build_baseline(seed)
    model.fit(texts, labels)
    return model


def save_baseline(model: Pipeline, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


class BaselineModel:
    display_name = "Character TF-IDF + Logistic Regression"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline: Pipeline | None = None

    def load(self) -> None:
        if self.pipeline is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到傳統模型：{self.model_path}。請先執行 python train_baseline.py。"
            )
        model = joblib.load(self.model_path)
        if not isinstance(model, Pipeline):
            raise TypeError("傳統模型檔不是預期的 scikit-learn Pipeline。")
        if not {"tfidf", "classifier"}.issubset(model.named_steps):
            raise ValueError("傳統模型缺少 tfidf 或 classifier 步驟。")
        classes = list(model.named_steps["classifier"].classes_)
        if set(classes) != {0, 1}:
            raise ValueError(f"傳統模型類別應為 0/1，目前為：{classes}")
        self.pipeline = model

    def predict(self, text: object) -> dict[str, object]:
        self.load()
        assert self.pipeline is not None
        value = normalize_text(text)
        started = perf_counter()
        probabilities = self.pipeline.predict_proba([value])[0]
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        human_probability = float(probabilities[classes.index(0)])
        ai_probability = float(probabilities[classes.index(1)])
        if not all(math.isfinite(value) for value in (human_probability, ai_probability)):
            raise RuntimeError("傳統模型輸出 NaN 或 Inf。")
        return {
            "human_probability": human_probability,
            "ai_probability": ai_probability,
            "predicted_class": int(ai_probability >= 0.5),
            "latency_ms": (perf_counter() - started) * 1_000,
            "warnings": [],
        }

    def explain(self, text: object, limit: int = 8) -> dict[str, object]:
        self.load()
        assert self.pipeline is not None
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        vector = vectorizer.transform([normalize_text(text)])
        if vector.nnz == 0:
            return {
                "ai_features": [],
                "human_features": [],
                "warning": "目前文字沒有模型詞彙表中的有效 n-gram，無法提供特徵線索。",
            }

        feature_names = vectorizer.get_feature_names_out()
        indices = vector.indices
        contributions = vector.data * classifier.coef_[0, indices]
        rows = [
            {"feature": str(feature_names[index]), "contribution": float(score)}
            for index, score in zip(indices, contributions, strict=True)
        ]
        ai_features = sorted(
            (row for row in rows if row["contribution"] > 0),
            key=lambda row: row["contribution"],
            reverse=True,
        )[:limit]
        human_features = sorted(
            (row for row in rows if row["contribution"] < 0),
            key=lambda row: row["contribution"],
        )[:limit]
        return {
            "ai_features": ai_features,
            "human_features": human_features,
            "warning": None,
        }
