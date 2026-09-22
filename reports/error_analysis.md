# 錯誤分析

本報告由 `evaluate.py` 依固定 test set 實際產生。為避免公開原始語料，僅保留彙總統計，不輸出任何文本片段。

## Traditional

- False positives：70
- False negatives：194
- 判對文本平均長度：177.7
- 判錯文本平均長度：130.1
- 判錯文本長度中位數：114.0

判錯文本長度分布：

- <50 字：26
- 50-99 字：87
- 100-199 字：114
- >=200 字：37

## 觀察與限制

- 少於 5 個中文字的輸入缺乏足夠線索，介面會保留結果但顯示短文本警告。
- Character n-gram 比舊版 word analyzer 適合中文，但仍可能學到固定句型、換行、URL 或資料格式，而非作者身分。
- `my_data.csv` 是較口語的單一正類集合，只能檢查 AI recall 與分數分布，不能當作完整 accuracy。
- `my_data.csv` Traditional AI recall：0.23076923076923078。
- `my_data.csv` BERT AI recall：1.0。
- BERT 比較狀態：available。

## 兩模型分歧

為避免反推出或重製原始資料，本次公開報告不列出高分歧案例文字。重新執行新版 `evaluate.py` 時，只會寫入分歧筆數與機率差的彙總統計。
