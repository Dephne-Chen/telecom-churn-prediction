"""
shap_analysis.py
----------------
載入已訓練好的 XGBoost 模型（pipeline），對驗證集進行 SHAP 分析，
包含 SHAP Summary Plot 與特定特徵之 Dependence Plot。
圖檔將輸出至 output/ 目錄。
"""

import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import os

# 📁 建立輸出資料夾（如不存在）
os.makedirs("output", exist_ok=True)

# 🔹 載入模型（pipeline_80）
with open("model/xgb_pipeline_gain80.pkl", "rb") as f:
    saved_data = pickle.load(f)
    pipeline = saved_data["pipeline"]
    selected_features = saved_data["selected_features"]
    
# 🔹 載入驗證集
X_val = pd.read_csv("data/x_origin_val.csv")

# 🔹 從模型中取得特徵名稱（如 cat__Contract）
feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()

# 🔹 從驗證集中選擇對應特徵（去除 __ 前綴）
X_val_sel = X_val[[col.split("__")[-1] for col in feature_names]].copy()

# 🔹 前處理後資料（SHAP 需要數值格式）
X_val_proc = pipeline.named_steps['preprocess'].transform(X_val_sel)

# 🔹 抽出 XGBoost 模型本體
xgb_model = pipeline.named_steps['clf']

# ✅ 建立 Explainer（tree-based explainer 適用於 XGBoost）
explainer = shap.Explainer(xgb_model, X_val_proc)

# 🔍 計算 SHAP 值
shap_values = explainer(X_val_proc)

# 📊 Summary Plot
shap.summary_plot(
    shap_values,
    features=X_val_proc,
    feature_names=feature_names,
    show=False
)
plt.savefig("output/shap_summary.png", bbox_inches="tight")
plt.close()

# 📍 Dependence Plots（選擇三個重點變數）
for col in ["cat__Contract", "remainder__Tenure in Months", "remainder__Monthly Charge"]:
    shap.dependence_plot(
        ind=col,
        shap_values=shap_values.values,
        features=X_val_proc,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(f"output/shap_dependence_{col.replace('__', '_')}.png", bbox_inches="tight")
    plt.close()

print("✅ SHAP 分析完成，圖檔已儲存至 output/")

