# 實驗設計與結果

本文件補充 README 中省略的資料處理、模型設定與評估方式。

## 資料建構

資料由本人從不同平台蒐集，並自行透過多種 AI 模型與 prompt engineering 產出及整理。使用過的模型系列包括 GPT-4o、GPT-4.5、GPT-4 Turbo、Gemini 1.0 Ultra、Gemini 1.5 Pro、Claude 3.5、Llama 3.1 與 Gemma。

資料設計同時納入：

- 一般人類文本。
- 一般 AI 回答。
- 經 prompt engineering 產生、較口語且「像人類」的 AI 文本。
- 文獻、報導、史料、參考書等具有專業表達、較「像 AI」的人類文本。

蒐集設計比例為易 50%、中 35%、難 15%。

## 清理與切分

`load_dataset` 執行：

1. 檢查 `text,label` 欄位。
2. 以 NFC 統一 Unicode。
3. 統一換行並收斂非換行空白。
4. 移除空文字。
5. 將 label 轉成整數並限制為 0／1。
6. 檢查同文異標。
7. 依正規化全文去重。

實際清理：

| 項目 | 數值 |
|---|---:|
| 原始筆數 | 40,532 |
| 空文字移除 | 0 |
| 重複文本移除 | 2 |
| 衝突文本 | 0 |
| 清理後筆數 | 40,530 |

Traditional 使用 `train_test_split` 和 `stratify=label`，固定 seed 42，先切出 30%，再等分為 validation 與 test。三份集合經實際檢查沒有完全相同文本重疊。

## Traditional

模型：

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        random_state=42,
    )),
])
```

Validation classification report：

| 類別 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Human | 95.45% | 97.96% | 96.69% | 3,339 |
| AI | 97.44% | 94.31% | 95.85% | 2,740 |
| Accuracy |  |  | 96.32% | 6,079 |
| Macro average | 96.44% | 96.14% | 96.27% | 6,079 |

獨立 test 結果：

| 指標 | 數值 |
|---|---:|
| Accuracy | 95.6579% |
| AI Precision | 97.3242% |
| AI Recall | 92.9197% |
| AI F1 | 95.0709% |
| Macro-F1 | 95.5954% |
| ROC-AUC | 99.0477% |
| PR-AUC | 99.0098% |
| False positive rate | 2.0958% |
| False negative rate | 7.0803% |
| 平均批次推論 | 0.3612 ms/筆 |

## BERT

既有 notebook 使用：

- Base model：`hfl/chinese-roberta-wwm-ext`
- Max length：400
- Learning rate：`2e-5`
- Train batch：16
- Evaluation batch：16
- Epochs：3
- Weight decay：0.01
- Best model at end：啟用
- Notebook 資料切分：80%／20%，GroupShuffleSplit，seed 42

Notebook 保存的 20% holdout report 約為：

| 類別 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Human | 1.00 | 0.95 | 0.97 | 4,476 |
| AI | 0.94 | 1.00 | 0.97 | 3,630 |
| Accuracy |  |  | 約 0.97 | 8,106 |

Notebook 沒有保存精確預測陣列或未四捨五入的混淆矩陣。Trainer 回載最佳 checkpoint 時曾顯示 LayerNorm `beta/gamma` 與 `weight/bias` 鍵名警告。

本次以 PyTorch 2.14.0 CPU、Transformers 5.17.0 載入最終模型時：

- 201/201 張量成功載入。
- missing keys：0。
- unexpected keys：0。
- mismatched keys：0。
- 未發現 NaN／Inf 權重。

### 切分重疊限制

目前 `evaluate.py` 的 test 是新的 70/15/15 切分，但 BERT 已經用 notebook 的舊 80% train 訓練。實際交集：

| 目前集合 | 筆數 | 位於舊 BERT train | 位於舊 BERT holdout |
|---|---:|---:|---:|
| Train | 28,371 | 22,769（80.25%） | 5,602（19.75%） |
| Validation | 6,079 | 4,810（79.12%） | 1,269（20.88%） |
| Test | 6,080 | 4,845（79.69%） | 1,235（20.31%） |

因此 `evaluate.py` 得到的 BERT 97.93% 只能用來確認模型載入、推論與輸出管線，不能當作獨立泛化指標，也不適合與 Traditional 的獨立 test 結果做嚴格排名。

重跑參考數值：

| 指標 | 數值 |
|---|---:|
| Accuracy | 97.9276% |
| AI Precision | 95.6993% |
| AI Recall | 99.8905% |
| AI F1 | 97.7500% |
| Macro-F1 | 97.9146% |
| ROC-AUC | 99.9116% |
| PR-AUC | 99.9032% |
| False positive rate | 3.6826% |
| False negative rate | 0.1095% |
| 平均批次推論 | 75.5465 ms/筆 |

若要公平比較兩模型，應保存一份固定 split manifest，並只用同一份 train 重新微調 BERT，validation 選擇 checkpoint，test 僅在最後使用一次。

## 單一正類資料 `my_data.csv`

原始 300 筆均為 AI，去重後 286 筆。這份資料只能回答「多少 AI 樣本被辨識出來」，不能提供完整 accuracy、specificity、ROC-AUC 或 false positive rate。

| 模型 | AI recall @ 0.5 | 平均 AI 分數 | 中位數 | Uncertain | Human likely |
|---|---:|---:|---:|---:|---:|
| Traditional | 23.08% | 38.40% | 35.98% | 44.41% | 47.20% |
| BERT | 100.00% | 99.94% | 99.95% | 0.00% | 0.00% |

Traditional 在這批口語 AI 文本上出現明顯分布偏移；BERT 在此集合表現很高，但仍需 Human 對照組與來源隔離，才能評估完整的錯判風險。

## 評估結果

- [`reports/metrics.json`](../reports/metrics.json)：機器可讀完整結果。
- [`reports/model_comparison.csv`](../reports/model_comparison.csv)：模型指標摘要。
- [`reports/confusion_matrix.png`](../reports/confusion_matrix.png)：混淆矩陣。
- [`reports/roc_curve.png`](../reports/roc_curve.png)：ROC 曲線。
- [`reports/error_analysis.md`](../reports/error_analysis.md)：不含原始文本的錯誤與模型分歧彙總。
