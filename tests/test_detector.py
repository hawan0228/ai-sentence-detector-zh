from pathlib import Path

import joblib
import pytest

from src.baseline_model import BaselineModel, train_baseline
from src.bert_model import BertModel
from src.detector import TextDetector


@pytest.fixture(scope="session")
def baseline_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "baseline.joblib"
    human = [
        "我今天去市場買菜，回家煮了一碗麵。", "昨晚雨下很大，我忘了帶傘。",
        "朋友約我週末爬山，但我有點懶。", "早餐的蛋餅有點焦，老闆還是很親切。",
    ] * 3
    ai = [
        "綜合以上因素，建議從三個面向進行分析。", "總體而言，此問題需要系統性地加以評估。",
        "以下將分別說明主要優點、限制與建議。", "此外，應注意資料品質可能影響最終結果。",
    ] * 3
    model = train_baseline(human + ai, [0] * len(human) + [1] * len(ai))
    joblib.dump(model, path)
    return path


def test_baseline_load_and_probabilities(baseline_path):
    result = BaselineModel(baseline_path).predict("今天下班後想去吃麵。")
    assert 0 <= result["ai_probability"] <= 1
    assert 0 <= result["human_probability"] <= 1
    assert result["human_probability"] + result["ai_probability"] == pytest.approx(1)


def test_detector_is_deterministic_and_warns_for_short_text(baseline_path, tmp_path):
    detector = TextDetector(baseline_path=baseline_path, bert_path=tmp_path / "missing")
    first = detector.analyze("你好。", "baseline")
    second = detector.analyze("你好。", "baseline")
    assert first["ai_probability"] == pytest.approx(second["ai_probability"])
    assert any("過短" in warning for warning in first["warnings"])


def test_long_text_does_not_fail(baseline_path, tmp_path):
    detector = TextDetector(baseline_path=baseline_path, bert_path=tmp_path / "missing")
    result = detector.analyze("今天下雨，所以我留在家裡看書。" * 500, "baseline")
    assert len(result["sentences"]) == 500
    assert 0 <= result["ai_probability"] <= 1


def test_blank_and_punctuation_only_are_rejected(baseline_path, tmp_path):
    detector = TextDetector(baseline_path=baseline_path, bert_path=tmp_path / "missing")
    with pytest.raises(ValueError, match="請輸入"):
        detector.analyze("  ", "baseline")
    with pytest.raises(ValueError, match="只有空白或標點"):
        detector.analyze("！？；", "baseline")


def test_bert_unavailable_is_graceful(tmp_path):
    status = BertModel(tmp_path / "missing").status()
    assert status["available"] is False
    assert "找不到" in status["reason"]


def test_compare_keeps_baseline_when_bert_unavailable(baseline_path, tmp_path):
    detector = TextDetector(baseline_path=baseline_path, bert_path=tmp_path / "missing")
    result = detector.analyze("今天下班後想去吃麵。", "compare")
    assert result["available"] is True
    assert len(result["comparison"]) == 2
    assert result["comparison"][1]["label_zh"] == "not_available"
