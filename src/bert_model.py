from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .preprocessing import normalize_text


class BertModel:
    """延遲載入本地二元分類模型；0=Human、1=AI 由訓練 notebook 確認。"""

    display_name = "Chinese RoBERTa fine-tuned classifier"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None
        self.device = "cpu"
        self.load_error: str | None = None
        self.loading_warnings: list[str] = []

    @property
    def available(self) -> bool:
        return self.model is not None and self.load_error is None

    def load(self) -> None:
        if self.available:
            return
        if self.load_error:
            raise RuntimeError(self.load_error)
        if not self.model_path.exists():
            self.load_error = f"找不到 BERT 模型目錄：{self.model_path}"
            raise FileNotFoundError(self.load_error)

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            self.load_error = (
                "BERT 依賴尚未安裝。請執行 pip install -r requirements-bert.txt。"
            )
            raise RuntimeError(self.load_error) from error

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            loaded = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
                output_loading_info=True,
            )
            model, loading_info = loaded
            missing = loading_info.get("missing_keys", [])
            unexpected = loading_info.get("unexpected_keys", [])
            mismatched = loading_info.get("mismatched_keys", [])
            if missing or unexpected or mismatched:
                raise RuntimeError(
                    "BERT 權重未完整對應："
                    f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
                )
            if int(model.config.num_labels) != 2:
                raise RuntimeError(f"BERT 模型不是二元分類器：num_labels={model.config.num_labels}")

            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all().item():
                    raise RuntimeError(f"BERT 權重 {name} 含 NaN 或 Inf。")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self.torch = torch
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
        except Exception as error:
            self.load_error = f"BERT 模型目前無法載入：{error}"
            raise RuntimeError(self.load_error) from error

    def status(self) -> dict[str, object]:
        try:
            self.load()
            return {
                "available": True,
                "device": self.device,
                "warnings": list(self.loading_warnings),
            }
        except Exception as error:
            return {"available": False, "reason": str(error), "warnings": []}

    def _chunk_token_ids(self, text: str) -> list[list[int]]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        max_positions = int(getattr(self.model.config, "max_position_embeddings", 512))
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        chunk_size = max(1, max_positions - special_count)
        return [token_ids[index : index + chunk_size] for index in range(0, len(token_ids), chunk_size)]

    def predict(self, text: object) -> dict[str, object]:
        self.load()
        value = normalize_text(text)
        started = perf_counter()
        chunks = self._chunk_token_ids(value)
        if not chunks:
            raise ValueError("沒有可供 BERT 分析的文字內容。")

        weighted_ai = 0.0
        total_weight = 0
        with self.torch.inference_mode():
            for token_ids in chunks:
                input_ids = [self.tokenizer.cls_token_id, *token_ids, self.tokenizer.sep_token_id]
                inputs = {
                    "input_ids": self.torch.tensor([input_ids], device=self.device),
                    "attention_mask": self.torch.ones(
                        (1, len(input_ids)), dtype=self.torch.long, device=self.device
                    ),
                    "token_type_ids": self.torch.zeros(
                        (1, len(input_ids)), dtype=self.torch.long, device=self.device
                    ),
                }
                logits = self.model(**inputs).logits
                if not self.torch.isfinite(logits).all().item():
                    raise RuntimeError("BERT 推論輸出 NaN 或 Inf。")
                probabilities = self.torch.softmax(logits, dim=-1)[0]
                # 訓練 notebook 將資料標籤 0/1 原樣交給模型：0=Human、1=AI。
                ai_probability = float(probabilities[1].item())
                weight = max(1, len(token_ids))
                weighted_ai += ai_probability * weight
                total_weight += weight

        ai_probability = weighted_ai / total_weight
        warnings = list(self.loading_warnings)
        if len(chunks) > 1:
            warnings.append(
                f"全文超過單次模型長度，已分成 {len(chunks)} 個 token 區塊並依長度加權。"
            )
        if not math.isfinite(ai_probability):
            raise RuntimeError("BERT 加權分數不是有限值。")
        return {
            "human_probability": 1.0 - ai_probability,
            "ai_probability": ai_probability,
            "predicted_class": int(ai_probability >= 0.5),
            "latency_ms": (perf_counter() - started) * 1_000,
            "warnings": warnings,
            "chunks": len(chunks),
        }

    def predict_many(self, texts, batch_size: int = 16) -> tuple[list[float], float]:
        """批次推論供評估使用；長文本仍按 token chunk 長度加權。"""
        self.load()
        all_chunks: list[tuple[int, list[int]]] = []
        text_count = 0
        for text_count, text in enumerate(texts, start=1):
            chunks = self._chunk_token_ids(normalize_text(text))
            if not chunks:
                raise ValueError(f"第 {text_count} 筆沒有可供 BERT 分析的內容。")
            all_chunks.extend((text_count - 1, chunk) for chunk in chunks)

        # 相近長度放在同一批可減少 padding；text_index 仍保留原始輸出順序。
        all_chunks.sort(key=lambda item: len(item[1]))

        weighted_sums = [0.0] * text_count
        total_weights = [0] * text_count
        started = perf_counter()
        with self.torch.inference_mode():
            for start in range(0, len(all_chunks), batch_size):
                batch = all_chunks[start : start + batch_size]
                sequences = [
                    [self.tokenizer.cls_token_id, *tokens, self.tokenizer.sep_token_id]
                    for _, tokens in batch
                ]
                max_length = max(len(sequence) for sequence in sequences)
                input_rows = []
                attention_rows = []
                for sequence in sequences:
                    padding = max_length - len(sequence)
                    input_rows.append(sequence + [self.tokenizer.pad_token_id] * padding)
                    attention_rows.append([1] * len(sequence) + [0] * padding)
                inputs = {
                    "input_ids": self.torch.tensor(input_rows, device=self.device),
                    "attention_mask": self.torch.tensor(attention_rows, device=self.device),
                    "token_type_ids": self.torch.zeros(
                        (len(batch), max_length), dtype=self.torch.long, device=self.device
                    ),
                }
                logits = self.model(**inputs).logits
                if not self.torch.isfinite(logits).all().item():
                    raise RuntimeError("BERT 批次推論輸出 NaN 或 Inf。")
                probabilities = self.torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                for (text_index, tokens), probability in zip(batch, probabilities, strict=True):
                    weight = max(1, len(tokens))
                    weighted_sums[text_index] += float(probability) * weight
                    total_weights[text_index] += weight

        values = [total / weight for total, weight in zip(weighted_sums, total_weights, strict=True)]
        return values, (perf_counter() - started) * 1_000 / max(1, text_count)
