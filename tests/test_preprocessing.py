import pytest

from src.preprocessing import normalize_text
from src.text_analysis import (
    classify_probability,
    is_analyzable,
    short_text_warning,
    split_sentences,
    text_statistics,
)


def test_blank_text_normalization():
    assert normalize_text("  \t\r\n  ") == ""
    assert split_sentences("  ") == []
    assert not is_analyzable("！？；")


def test_sentence_split_keeps_punctuation():
    assert split_sentences("第一句。第二句！\n第三句？還有；最後") == [
        "第一句。", "第二句！", "第三句？", "還有；", "最後"
    ]


def test_short_text_warning_and_statistics():
    assert short_text_warning("你好。") is not None
    stats = text_statistics("你好。\n今天下雨！")
    assert stats["句數"] == 2
    assert stats["換行數量"] == 1
    assert stats["標點數量"] == 2


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "human_likely"), (0.3499, "human_likely"), (0.35, "uncertain"),
     (0.65, "uncertain"), (0.6501, "ai_likely"), (1.0, "ai_likely")],
)
def test_three_way_boundaries(probability, expected):
    assert classify_probability(probability)[0] == expected
