"""
preprocessing.py
----------------
電信流失預測專案的資料清洗與預處理腳本。
步驟包括欄位篩選、缺失值處理、異常值排除與資料重分類。
"""


import numpy as np
import pandas as pd

# 讀取資料
df = pd.read_csv('data/telecom_customer_churn.csv')

# 刪掉無實質預測意義的欄位
df = df.drop(columns=['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude'])

# 刪除資料洩漏欄位
df = df.drop(columns=['Total Revenue', 'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Churn Category', 'Churn Reason'])

# 列出全部欄位的缺失值數量
df.isnull().sum()

# 處理沒有網路服務的欄位
# 因為「沒有網路服務」，不是資料缺漏 -> 類別型欄位補「No」，數值型欄位補 0
cat_cols = ['Online Security', 'Online Backup', 'Device Protection Plan',
            'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data']

df[cat_cols] = df[cat_cols].fillna("No")
df['Internet Type'] = df['Internet Type'].fillna("No Internet")
df["Avg Monthly GB Download"] = df["Avg Monthly GB Download"].fillna(0)

# 處理沒有家庭電話的欄位
# 因為「沒有家庭電話」，不是資料缺漏 -> 類別型欄位補「No」，數值型欄位補 0
df['Multiple Lines'] = df['Multiple Lines'].fillna('No')
df['Avg Monthly Long Distance Charges'] = df['Avg Monthly Long Distance Charges'].fillna(0)

# Offer空值->未使用優惠方案->若有空值補 "None"
df["Offer"] = df["Offer"].fillna("None")

# 快速瀏覽所有欄位的「內容樣本」，查看有無異常值
for col in df.columns:
    print(f"{col}: {df[col].unique()[:10]}")

# 上面發現'Monthly Charge'有負值
# 查看有幾筆異常值
neg_count = (df['Monthly Charge'] < 0).sum()
print("Monthly Charge < 0 的資料筆數：", neg_count)

# 只有 120 筆，由於不是關鍵業務場景，刪除對模型、分析結果影響極低
# 只保留'Monthly Charge'>= 0 的資料
df = df[df['Monthly Charge'] >= 0]

# 刪除「Join」中留存一個月且為'Month-to-Month'
df = df[~((df['Customer Status'] == 'Joined') &
          (df['Contract'] == 'Month-to-Month') &
          (df['Tenure in Months'] == 1))]

# 將剩餘 Joined 轉換為 Stayed 
df.loc[df['Customer Status'] == 'Joined', 'Customer Status'] = 'Stayed'

df.to_csv('data/cleaned_telecom_final.csv', index=False)
print("✅ 資料清理完成，輸出至：data/cleaned_telecom_final.csv")
