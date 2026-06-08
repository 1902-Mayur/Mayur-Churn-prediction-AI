# ⚡ ChurnShield AI — Customer Churn Prediction System

> An advanced full-stack ML web application that predicts telecom customer churn using ensemble machine learning, with an interactive dark-themed dashboard, batch CSV processing, and AI-generated retention recommendations.

---

## 📸 Features

| Feature | Description |
|---|---|
| 🔮 Single Prediction | Enter customer details and get instant churn probability with risk gauge |
| 📂 Batch Processing | Upload CSV to score hundreds of customers at once |
| 📊 Analytics Dashboard | 7 auto-generated charts covering EDA, model comparison, feature importance, BI |
| 💡 Retention Tips | Personalised retention strategy for each prediction |
| 🔌 REST API | JSON endpoint for programmatic integration |
| 🏆 Auto Model Selection | Best model auto-picked by ROC-AUC across 6 classifiers |

---

## 🤖 ML Pipeline

```
Raw CSV → Clean → Feature Engineer → SMOTE Balance → Train 6 Models → Evaluate → Select Best → Serve
```

### Models Trained
- Logistic Regression (baseline)
- Decision Tree
- Random Forest (200 trees)
- Gradient Boosting
- XGBoost
- Voting Ensemble (RF + XGB + GB)

### Engineered Features (8 new)
- `engagement_score` — weighted composite of services, tenure, demographics
- `num_services` — count of active add-on services
- `avg_monthly_spend` — TotalCharges / tenure
- `charge_per_service` — MonthlyCharges / active services
- `tenure_group` — bucketed tenure (0–3)
- `is_long_term` — tenure > 24 months flag
- `high_spender` — top quartile monthly charges
- `low_support_risk` — no tech support AND no online security

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── app.py                    # Flask web application
├── requirements.txt
├── README.md
├── dataset/
│   ├── telecom_churn.csv     # 7,043 customer records
│   └── generate_dataset.py
├── notebooks/
│   └── churn_pipeline.py     # Full ML pipeline
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── feature_names.pkl
│   └── metadata.json
├── templates/
│   ├── base.html
│   ├── index.html            # Overview + model comparison table
│   ├── predict.html          # Single prediction form
│   ├── batch.html            # CSV batch upload
│   └── dashboard.html        # Full visual analytics
├── static/
│   ├── css/main.css          # Dark-themed design system
│   └── js/main.js
└── visuals/
    ├── 01_eda_overview.png
    ├── 02_correlation_heatmap.png
    ├── 03_engineered_features.png
    ├── 04_model_comparison.png
    ├── 05_best_model_evaluation.png
    ├── 06_feature_importance.png
    └── 07_business_dashboard.png
```

---

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset and train models
python dataset/generate_dataset.py
python notebooks/churn_pipeline.py

# 5. Start the web app
python app.py
```

Open **http://localhost:5050** in your browser.

---

## 🔌 REST API

```bash
curl -X POST http://localhost:5050/api/quick_predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 6,
    "monthly_charges": 95,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "online_security": "No",
    "payment_method": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn": true,
  "probability": 0.847,
  "confidence": "84.7%",
  "risk_segment": "High"
}
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.79 | ~0.66 | ~0.55 | ~0.60 | ~0.84 |
| Decision Tree | ~0.77 | ~0.62 | ~0.60 | ~0.61 | ~0.74 |
| Random Forest | ~0.82 | ~0.71 | ~0.61 | ~0.65 | ~0.87 |
| Gradient Boosting | ~0.82 | ~0.71 | ~0.62 | ~0.66 | ~0.87 |
| **XGBoost** | **~0.83** | **~0.72** | **~0.63** | **~0.67** | **~0.88** |
| Voting Ensemble | ~0.83 | ~0.72 | ~0.63 | ~0.67 | ~0.88 |

---

## 💡 Key Business Insights

- **Month-to-month** contract customers churn **3× more** than 2-year customers
- **Electronic check** payment correlates with highest churn rate (~45%)
- **Fiber optic** users without security add-ons are at significantly higher risk
- Customers with **tenure < 12 months** have the highest churn probability
- Adding **Tech Support** reduces churn probability by ~40%

---

## 🛠 Tech Stack

- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3 (custom dark design system), JavaScript
- **Visualisation:** Matplotlib, Seaborn
- **Data:** Pandas, NumPy

---

## 📝 Resume Description

> *Developed a Customer Churn Prediction System using Python, Scikit-learn, Random Forest, and XGBoost to identify high-risk telecom customers with ~88% ROC-AUC. Built a full-stack Flask dashboard with batch CSV scoring, explainable AI feature importance, SMOTE class balancing, and personalised retention strategy recommendations.*

---

## 🔮 Future Improvements

- [ ] SHAP values for per-customer feature explanations
- [ ] Real-time streaming predictions with Kafka
- [ ] Automated retraining pipeline with MLflow tracking
- [ ] Deploy to Render / Streamlit Cloud
- [ ] PDF report generation for batch predictions
- [ ] Email alerts for high-risk customer segments
