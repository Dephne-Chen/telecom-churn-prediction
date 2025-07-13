"""
split_data.py
-------------
將清理後的資料集切分為訓練集、驗證集與測試集，
並將目標欄位 Customer Status 進行標籤編碼（Churned=1, Stayed=0）。
"""


import pandas as pd
from sklearn.model_selection import train_test_split

# 📥 載入原始資料
df = pd.read_csv('data/cleaned_telecom_final.csv')

# 🔎 設定特徵與目標欄位
X_raw = df.drop(columns=['Customer Status'])  # 特徵欄位
y_raw = df['Customer Status']                # 目標欄位

# 📊 Step 1：切出 20% 測試集
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=42
)

# 📊 Step 2：將剩餘資料切為訓練集（64%）與驗證集（16%）
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42
)

# Label Encode y（Churned=1, Stayed=0）
label_mapping = {'Churned': 1, 'Stayed': 0}

def encode_target_y_safe(y, label_mapping):
    y_series = pd.Series(y).astype(str).str.strip().str.title()
    y_mapped = y_series.map(label_mapping)

    if y_mapped.isna().any():
        missing = y_series[y_mapped.isna()].unique()
        raise ValueError(f"❌ 下列類別無法對應：{missing.tolist()}")
    return y_mapped.astype(int)

y_train = encode_target_y_safe(y_train, label_mapping)
y_val   = encode_target_y_safe(y_val, label_mapping)
y_test  = encode_target_y_safe(y_test, label_mapping)

# 📁 輸出切分後的資料（存放於 data/）
X_train.to_csv("data/x_origin_train.csv", index=False)
X_val.to_csv("data/x_origin_val.csv", index=False)
X_test.to_csv("data/x_origin_test.csv", index=False)
y_train.to_csv("data/y_label_訓練.csv", index=False)
y_val.to_csv("data/y_label_驗證.csv", index=False)
y_test.to_csv("data/y_label_test.csv", index=False)

# ✅ 結果摘要
print("✅ y 標籤編碼完成：Churned = 1, Stayed = 0")
print("📊 Train shape:", X_train.shape)
print("📊 Validation shape:", X_val.shape)
print("📊 Test shape:", X_test.shape)
print("✅ 切分結果已儲存至 data/")
