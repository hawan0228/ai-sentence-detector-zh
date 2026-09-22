from __future__ import annotations

import re

from .preprocessing import normalize_text

LOWER_THRESHOLD = 0.35
UPPER_THRESHOLD = 0.65


def classify_probability(
    ai_probability: float,
    lower_threshold: float = LOWER_THRESHOLD,
    upper_threshold: float = UPPER_THRESHOLD,
) -> tuple[str, str]:
    if not 0 <= ai_probability <= 1:
        raise ValueError("機率必須介於 0 和 1。")
    if not 0 <= lower_threshold <= upper_threshold <= 1:
        raise ValueError("判定門檻設定錯誤。")
    if ai_probability < lower_threshold:
        return "human_likely", "較偏向人類文本"
    if ai_probability <= upper_threshold:
        return "uncertain", "無法確定"
    return "ai_likely", "較偏向 AI 文本"


def split_sentences(text: object) -> list[str]:
    value = normalize_text(text)
    if not value:
        return []
    pieces = re.split(r"(?<=[。！？!?；;])|\n+", value)
    return [piece.strip() for piece in pieces if piece.strip()]


def content_length(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def is_analyzable(text: object) -> bool:
    value = normalize_text(text)
    return bool(value) and content_length(value) > 0


def short_text_warning(text: str) -> str | None:
    if content_length(text) < 5:
        return "文字過短，結果較不穩定。"
    return None


def text_statistics(text: object) -> dict[str, int | float]:
    value = normalize_text(text)
    sentences = split_sentences(value)
    lengths = [content_length(sentence) for sentence in sentences]
    punctuation_count = len(re.findall(r"[，。！？；：、,.!?;:]", value))
    total_chars = len(re.sub(r"\s", "", value))
    return {
        "總字數": total_chars,
        "句數": len(sentences),
        "平均句長": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "最長句長": max(lengths, default=0),
        "標點數量": punctuation_count,
        "換行數量": value.count("\n"),
    }
