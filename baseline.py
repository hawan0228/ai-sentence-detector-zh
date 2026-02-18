import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 讀資料
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# TFIDF 向量化
vectorizer = TfidfVectorizer(max_features=5000)     # 頻率最高的前 5000 個 term 作為特徵
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 訓練 Logistic Regressionr 進行分類
clf = LogisticRegression(max_iter=1000)    # 迭代 1000 次
clf.fit(X_train_vec, y_train)

# 預測
y_pred = clf.predict(X_test_vec)

# 評估
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 保存模型和vectorizer
import joblib
joblib.dump(clf, "models/baseline_lr_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("模型與向量器已保存於 models/ 目錄")
