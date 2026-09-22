# 模型檔案

模型權重不納入 Git，請在本機放置於以下位置：

- `models/baseline_char_tfidf.joblib`：執行 `python train_baseline.py` 產生的 character TF-IDF + Logistic Regression Pipeline。
- `models/bert/`：Hugging Face 格式的本地二元分類模型，需包含 config、tokenizer 與權重檔。

既有的 `baseline_lr_model.pkl` 與 `tfidf_vectorizer.pkl` 是早期 word-level baseline 產物；新程式不會覆寫它們。

專案目前沒有可公開驗證的模型下載網址，因此不提供虛構連結。BERT 由 notebook 中的 `hfl/chinese-roberta-wwm-ext` 微調而來，標籤為 `0 = Human`、`1 = AI`。
