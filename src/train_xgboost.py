"""
train_xgboost.py
----------------
本程式用於訓練電信流失預測模型，包含：
1. 初始模型訓練
2. XGBoost 特徵重要性計算（以 gain 為指標）
3. 篩選 Gain 累積貢獻達 80% 的特徵
4. 以該特徵集重新訓練模型並進行測試集評估
"""

# 📦 資料處理與視覺化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ⚙️ 特徵處理與建模流程
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 📈 模型與訓練
from xgboost import XGBClassifier

# 🧪 評估指標
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 匯入訓練集
X_train = pd.read_csv("data/x_origin_train.csv")
y_train = pd.read_csv("data/y_label_訓練.csv")
# 匯入驗證集
X_val = pd.read_csv("data/x_origin_val.csv")
y_val = pd.read_csv("data/y_label_驗證.csv")

# 如果 y_train 是 DataFrame，轉為 Series
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]

if isinstance(y_val, pd.DataFrame):
    y_val = y_val.iloc[:, 0]

# 🔹 區分欄位類型
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 🔹 建立前處理器（OrdinalEncoder for 類別欄位）
preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
], remainder='passthrough',
force_int_remainder_cols=False  # 🛡️ 避免 future warning
                                 )

# 🔹 建立 pipeline（XGBoost 模型）
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=100, learning_rate=0.1, max_depth=5,
      eval_metric='aucpr',scale_pos_weight=3
    ))
])

# 🔹 訓練模型
pipeline.fit(X_train, y_train)

# 🔹 預測驗證集
y_val_pred = pipeline.predict(X_val)
y_val_prob = pipeline.predict_proba(X_val)[:, 1]  # 機率分數用於計算 AUC

# 🔹 計算評估指標
val_accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_prob)
train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))

# 🔹 印出結果
print(f"Train Accuracy: {train_accuracy:.10f}")
print(f"\nValidation Accuracy: {val_accuracy:.10f}")
print(f"ROC AUC: {roc_auc:.10f}\n")

print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# 取得訓練後的 XGBoost 模型
booster = pipeline.named_steps['clf'].get_booster()

# 取得經過 OrdinalEncoder 後的所有欄位名稱
# 類別欄位名稱
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
feature_names = cat_cols + num_cols  # OrdinalEncoder 不會產生新欄位名稱

# 抓出 feature importance（by gain）
score_dict = booster.get_score(importance_type='gain')

# 轉為 DataFrame（注意：XGBoost 的欄位名是 f0, f1, f2,...）
# 把 f0 ➝ feature_names[0]，這樣就能顯示人類可讀的特徵名稱。
importance_df = pd.DataFrame([
    {"feature": feature_names[int(f[1:])], "gain": score}
    for f, score in score_dict.items()
])

# 排序 + 計算累積貢獻比例
importance_df = importance_df.sort_values(by="gain", ascending=False).reset_index(drop=True)
importance_df["gain_pct"] = importance_df["gain"] / importance_df["gain"].sum()
importance_df["gain_cumsum"] = importance_df["gain_pct"].cumsum()

#（選配）畫圖前 20 名
plt.figure(figsize=(10, 6))
plt.barh(importance_df.head(20)['feature'][::-1], importance_df.head(20)['gain'][::-1])
plt.xlabel("Gain")
plt.title("Top 20 Features by Gain")
plt.tight_layout()
plt.show()

# 選出 Gain 80% 的特徵
# 取得 Gain 80% 特徵
# 建立 Gain 80% 的特徵清單
selected_features_80 = importance_df[importance_df["gain_cumsum"] <= 0.80]["feature"].tolist()

print(f"🎯 Gain 80% 特徵數量：{len(selected_features_80)}")
print(selected_features_80)

# 🔁 retrain 函數（以任意特徵清單）
def retrain_and_evaluate(feature_subset, subset_name):
    print(f"\n🔁 Retrain Model using {subset_name} ({len(feature_subset)} features)\n")

    # 🔹 篩選特徵子集
    X_train_sel = X_train[feature_subset].copy()
    X_val_sel   = X_val[feature_subset].copy()

    # 🔹 區分欄位類型
    cat_cols = X_train_sel.select_dtypes(include='object').columns.tolist()

    # 🔹 前處理器：Ordinal 編碼
    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough',force_int_remainder_cols=False)

    # 🔹 Pipeline（含 XGBoost 模型）
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=3
        ))
    ])

    # 🔹 訓練模型
    pipeline.fit(X_train_sel, y_train)

    # 🔹 預測與評估
    y_val_pred = pipeline.predict(X_val_sel)
    y_val_prob = pipeline.predict_proba(X_val_sel)[:, 1]
    y_train_pred = pipeline.predict(X_train_sel)

    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.10f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.10f}")
    print(f"ROC AUC: {roc_auc_score(y_val, y_val_prob):.10f}\n")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    return pipeline, X_train_sel, X_val_sel

# ✅ 執行 Gain 80% 特徵版本
pipeline_80, X_train_sel, X_val_sel = retrain_and_evaluate(selected_features_80, "Gain 80%")

# 測試集結果
X_test = pd.read_csv("data/x_origin_test.csv")
y_test = pd.read_csv("data/y_label_test.csv")  # 如果有標籤的話

# 如果 y_test 是 DataFrame，轉為 Series
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# 選擇 Gain 80% 特徵
X_test_sel = X_test[selected_features_80].copy()

# 機率預測
y_test_proba = pipeline_80.predict_proba(X_test_sel)[:, 1]

# 自訂 threshold 的預測
y_test_pred_custom = (y_test_proba >= 0.5).astype(int)

print(f"✅ Test Accuracy: {accuracy_score(y_test, y_test_pred_custom):.4f}")
print(f"✅ ROC AUC: {roc_auc_score(y_test, y_test_pred_custom):.4f}")
print(f"📌 Precision (class=1): {precision_score(y_test, y_test_pred_custom):.4f}")
print(f"📌 Recall (class=1): {recall_score(y_test, y_test_pred_custom):.4f}")
print(f"📌 F1-score (class=1): {f1_score(y_test, y_test_pred_custom):.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred_custom))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_custom))

import pickle

# 📌 儲存最終模型（pipeline_80）為 pkl 檔
with open("model/xgb_pipeline_gain80.pkl", "wb") as f:
    pickle.dump(pipeline_80, f)
