# 安裝與操作

本文件整理完整的安裝、訓練、評估與 Demo 操作方式

## 環境

建議 Python 3.11。本次驗證環境：

- Python 3.11.7
- pandas 3.0.5
- NumPy 2.4.6
- scikit-learn 1.9.1
- Gradio 6.27.0
- PyTorch 2.14.0 CPU
- Transformers 5.17.0

## 建立虛擬環境

Windows PowerShell：

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux／macOS：

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 足以使用 Traditional、評估圖表、測試與 Gradio。若需使用本地 BERT：

```bash
python -m pip install -r requirements-bert.txt
```

## 本地資料與模型

必要資料：

```text
data/data.csv
```

選用的單一正類測試資料：

```text
data/my_data.csv
```

BERT 應以 Hugging Face 格式放在：

```text
models/bert/config.json
models/bert/model.safetensors
models/bert/tokenizer.json
models/bert/tokenizer_config.json
```

資料與大型模型由 `.gitignore` 排除。專案目前沒有經驗證的公開下載網址。

## 訓練 Traditional

使用預設路徑與 seed：

```bash
python train_baseline.py
```

指定輸入、輸出和 seed：

```bash
python train_baseline.py \
  --data data/data.csv \
  --output models/baseline_char_tfidf.joblib \
  --seed 42
```

Windows PowerShell 可寫成一行，或使用反引號取代反斜線續行。

程式會輸出：

- 清理前後筆數。
- 重複與衝突標籤檢查。
- Train／Validation／Test 筆數及類別比例。
- Validation classification report。
- 完整 scikit-learn Pipeline 模型。

訓練階段不讀取 test 指標。

## 評估

評估 Traditional 及可用的 BERT：

```bash
python evaluate.py
```

只評估 Traditional：

```bash
python evaluate.py --skip-bert
```

自訂路徑：

```bash
python evaluate.py \
  --data data/data.csv \
  --my-data data/my_data.csv \
  --baseline models/baseline_char_tfidf.joblib \
  --bert models/bert \
  --reports reports \
  --seed 42
```

輸出：

```text
reports/metrics.json
reports/model_comparison.csv
reports/confusion_matrix.png
reports/roc_curve.png
reports/error_analysis.md
```

BERT 在 CPU 上可執行，但完整資料評估明顯較慢。評估程式會按 token 長度排列批次以降低 padding；輸出會依原始樣本順序還原。

## 啟動 Gradio

```bash
python app.py
```

瀏覽器開啟：

```text
http://127.0.0.1:7860
```

介面提供：

- Traditional
- BERT
- 模型比較
- 全文判定與 Human／AI 分數
- 逐句分析
- Traditional n-gram 判斷線索
- 文字統計
- 模型警告與分歧提示

BERT 採延遲載入，因此第一次選用 BERT 時會比之後的推論慢。若缺少 BERT 套件、模型或權重載入失敗，Compare 模式仍會顯示 Traditional 結果。

## 判定門檻

目前使用未正式校準的保守經驗門檻：

| AI 分數 | 顯示結果 |
|---:|---|
| `< 0.35` | 較偏向人類文本 |
| `0.35 ～ 0.65` | 無法確定 |
| `> 0.65` | 較偏向 AI 文本 |

模型原始預測類別仍以 0.5 為二元分類門檻；三段式區間主要用於 Demo 呈現不確定性。

## 測試

```bash
python -m pytest -q
```

目前輕量測試涵蓋：

- 空白及純標點輸入。
- 中文句子切分與標點保留。
- 短文本警告。
- 長文本不崩潰。
- Traditional 模型載入與機率範圍。
- Human／AI 機率總和。
- 三段式門檻邊界。
- 相同輸入結果一致。
- BERT 不可用時的安全降級。
