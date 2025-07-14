# 📊 Telecom Customer Churn Prediction

本專題旨在建構一套電信客戶流失預測模型，透過特徵篩選與模型優化策略，協助企業有效辨識高流失風險客戶，並提升留存率。

---

## 🧠 專案亮點
- **資料處理**：缺失值補全、類別轉換、標準化
- **特徵篩選**：依據 XGBoost 的 gain 值排序，選擇累積貢獻達 **80%** 的特徵作為子集
- **模型訓練**：使用 XGBoost，結合手動調參
- **模型解釋**：使用 SHAP 分析視覺化模型邏輯與重要特徵

---

## 🗂️ 專案結構
| 資料夾 | 說明 |
|--------|------|
| `src/` | Python 程式碼（資料前處理、模型訓練、SHAP 分析） |
| `data/` | 資料樣本 |
| `output/` | 模型特徵圖表、SHAP summary 等圖片 |
| `report/` | 專題簡報 |

---
## 🚀 執行方式
```bash
# 1️⃣ 安裝必要套件（建議先建立虛擬環境）
pip install -r requirements.txt

# 2️⃣ 資料前處理：清洗欄位、補值、排除異常與資料洩漏欄位
python src/preprocessing.py

# 3️⃣ 資料切分：將清洗後資料切為訓練／驗證／測試集，並對y進行 Label Encoding
python src/split_data.py

# 4️⃣ 模型訓練：使用 XGBoost，搭配手動調參與 Gain 80% 特徵子集
python src/train_xgboost.py

# 5️⃣ 模型解釋：透過 SHAP 視覺化模型邏輯與關鍵特徵影響力
python src/shap_analysis.py

```
---

## 📈 模型成效

- 模型：XGBoost（手動調參）
- 特徵版本：Gain 累積貢獻 80%
- 測試集成效（Class 0）：
  - Recall：0.88
  - Precision：0.75
  - F1-score：0.65

💡 根據商業情境，企業更關注「流失客戶」（Class 0）的辨識，因此模型以提升該類別的召回率為主要優化目標。

---

## 🖥️ 使用者介面（Streamlit）

本專案搭配 Streamlit 建立互動式操作介面，主要展示以下功能：

- 預測指定客戶是否會流失
- 顯示預測結果與策略方案
- 呈現 SHAP 圖解釋模型判斷邏輯

📁 Streamlit 原始碼位於 `app/` 資料夾中。

---

## 👥 團隊成員
陳晏綾、林靖軒、孫敏嘉、吳建儒    
(每個部分皆由組員共同參與)

## 📜 License
本專案僅供學術展示與學習用途，禁止未經授權商業使用。
