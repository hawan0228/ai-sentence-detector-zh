# 資料說明

訓練與評估預期使用 `data/data.csv`，欄位為 `text,label`。依目前 notebook 與資料內容，標籤定義為 `0 = Human`、`1 = AI`。

## 蒐集設計

資料由專案作者從不同平台蒐集，並自行透過 prompt 調適與 prompt engineering 產生及整理 AI 文本。AI 語料使用過 GPT-4o、GPT-4.5、GPT-4 Turbo、Gemini 1.0 Ultra、Gemini 1.5 Pro、Claude 3.5、Llama 3.1 與 Gemma 等模型系列。

為增加辨識難度，AI 類別包含刻意生成的「像人類的 AI 文本」；Human 類別則包含文獻、報導、史料、參考書等較正式、專業且可能「像 AI」的文本。蒐集時規劃的難度比例為易 50%、中 35%、難 15%。

## 實際統計

| 類別 | 原始筆數 | 去重後筆數 |
|---|---:|---:|
| Human（0） | 22,263 | 22,263 |
| AI（1） | 18,269 | 18,267 |
| 合計 | 40,532 | 40,530 |


`my_data.csv` 只有標籤 1，僅適合回報 AI recall 與分數分布，不能計算完整分類 accuracy。

`samples.csv` 只提供人工撰寫的格式示例，不是正式訓練或評估資料。
