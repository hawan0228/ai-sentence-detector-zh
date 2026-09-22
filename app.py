from __future__ import annotations

from typing import Any

import pandas as pd

from src.detector import TextDetector

MODEL_CHOICES = {
    "傳統模型": "baseline",
    "BERT": "bert",
    "模型比較": "compare",
}

detector = TextDetector()


def _probability_text(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _summary(result: dict[str, Any]) -> str:
    lines = [f"### {result['label_zh']}"]
    if result["ai_probability"] is not None:
        lines.extend(
            [
                f"- AI 風格傾向：{_probability_text(result['ai_probability'])}",
                f"- Human：{_probability_text(result['human_probability'])}",
            ]
        )
    lines.extend(
        [
            f"- 使用模式：{result['model_name']}",
            f"- 總處理時間：{result['latency_ms']:.2f} ms",
        ]
    )
    if result["warnings"]:
        lines.append("\n**提醒**")
        lines.extend(f"- {warning}" for warning in dict.fromkeys(result["warnings"]))
    return "\n".join(lines)


def _comparison_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in result.get("comparison", []):
        rows.append(
            {
                "模型": row["model"],
                "AI 風格傾向": _probability_text(row["ai_probability"]),
                "判定": row["label_zh"],
                "推論時間 (ms)": (
                    round(row["latency_ms"], 2) if row["latency_ms"] is not None else "—"
                ),
                "狀態": row["status"],
            }
        )
    return pd.DataFrame(rows)


def _sentence_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "句子": row["sentence"],
                "AI 風格傾向": f"{row['ai_probability'] * 100:.2f}%",
                "判定": row["label_zh"],
                "模型": row["model"],
                "提醒": "；".join(row["warnings"]),
            }
            for row in result.get("sentences", [])
        ]
    )


def _feature_frame(result: dict[str, Any]) -> pd.DataFrame:
    features = result.get("features")
    if not features:
        return pd.DataFrame(columns=["傾向", "文字特徵", "貢獻值"])
    rows = []
    for direction, key in (("AI", "ai_features"), ("Human", "human_features")):
        rows.extend(
            {
                "傾向": direction,
                "文字特徵": item["feature"],
                "貢獻值": round(item["contribution"], 6),
            }
            for item in features[key]
        )
    if features.get("warning"):
        rows.append({"傾向": "提醒", "文字特徵": features["warning"], "貢獻值": "—"})
    return pd.DataFrame(rows)


def analyze_for_ui(text: str, model_choice: str):
    try:
        mode = MODEL_CHOICES.get(model_choice, "compare")
        result = detector.analyze(text, mode)
        return (
            _summary(result),
            _comparison_frame(result),
            _sentence_frame(result),
            _feature_frame(result),
            result["statistics"],
        )
    except Exception as error:
        return (
            f"### 無法分析\n\n{error}",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )


def build_app():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("尚未安裝 Gradio，請執行 pip install -r requirements.txt。") from error

    with gr.Blocks(title="中文 AI 語句分析") as demo:
        gr.Markdown(
            "# 中文 AI 語句分析\n"
            "比較 character TF-IDF 與本地 BERT 對中文文字風格的估計。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="中文文字",
                    lines=12,
                    placeholder="請輸入一段中文文字……",
                )
                model_choice = gr.Radio(
                    list(MODEL_CHOICES), value="模型比較", label="分析模式"
                )
                with gr.Row():
                    analyze_button = gr.Button("開始分析", variant="primary")
                    clear_button = gr.Button("清除")
                gr.Examples(
                    examples=[
                        ["下班後我去市場買了兩顆番茄，回家才發現忘記買蛋。", "模型比較"],
                        ["綜合以上因素，可將問題分成資料、方法與評估三個面向進行說明。", "傳統模型"],
                        ["你好。", "模型比較"],
                        ["第一段說明背景。第二段整理方法；最後提出仍需驗證的限制。", "模型比較"],
                    ],
                    inputs=[text_input, model_choice],
                )
            with gr.Column(scale=2):
                summary = gr.Markdown("尚未分析。")
                comparison = gr.Dataframe(label="模型比較", interactive=False)
                sentences = gr.Dataframe(label="逐句分析", interactive=False)
                features = gr.Dataframe(label="傳統模型判斷線索", interactive=False)
                statistics = gr.JSON(label="文字統計")

        outputs = [summary, comparison, sentences, features, statistics]
        analyze_button.click(analyze_for_ui, [text_input, model_choice], outputs)
        text_input.submit(analyze_for_ui, [text_input, model_choice], outputs)
        clear_button.click(
            lambda: ("", "模型比較", "尚未分析。", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}),
            outputs=[text_input, model_choice, *outputs],
        )
        gr.Markdown(
            "---\n"
            "本工具僅提供模型估計的文字風格傾向，結果不能證明作者是否使用 AI，"
            "也不應單獨用於學術處分或其他高風險決策。"
        )
    return demo


if __name__ == "__main__":
    build_app().launch(share=False)
