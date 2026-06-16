# 📡 Telecom Customer Churn Prediction

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-FF9A00?style=for-the-badge)](https://huggingface.co/spaces/Simosro/Cust-Churn-Prediction)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-yellow?style=for-the-badge)](https://catboost.ai)

> AI-powered customer retention intelligence — trained on **1,000,000** telecom customer records.
> Deployed as an interactive Gradio app on HuggingFace Spaces.

---

## 🚀 Live Demo

**👉 [Try the app on HuggingFace](https://huggingface.co/spaces/Simosro/Cust-Churn-Prediction)**

![App Screenshot](assets/app_screenshot.png)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.6861 |
| PR-AUC | 0.2066 |
| 5-Fold CV PR-AUC | 0.2061 ± 0.0023 |
| Recall | 100% |
| Capture Rate @ Top 20% | 40.6% |

> **5-Fold CV std = 0.0023** — excellent stability, zero overfitting confirmed.

---

## 🏗️ Project Structure

```
telecom-customer-churn-prediction/
├── app.py                  # Gradio UI (HuggingFace deployment)
├── inference.py            # Production inference pipeline
├── requirements.txt        # Dependencies
├── cust-churn.ipynb        # Full ML pipeline 
└── README.md
```

---

## 🔬 ML Pipeline

### Dataset
- **1,000,000** synthetic telecom customer records
- **Churn rate:** 9.92% (1:9.1 imbalance)
- **32 raw features** → **46 total features** after engineering

### Feature Engineering (17 engineered features)
Key engineered features confirmed by SHAP analysis:
- `service_vulnerability` — no tech support × high service calls (SHAP rank 8)
- `complaint_trend` — complaint acceleration over tenure (lift 1.50×)
- `contract_sat_risk` — contract type × satisfaction interaction (SHAP rank 10)

### Imbalance Strategy
Class Weights (9:1) — gives **Recall=100%** vs **0.5%** without weights (63× improvement) at negligible PR-AUC cost.

### Models Compared
| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| **CatBoost** ⭐ | **0.2066** | **0.6861** |
| Random Forest | 0.2007 | 0.6818 |
| XGBoost | 0.1997 | 0.6803 |
| LightGBM | 0.1894 | 0.6759 |

### Hyperparameter Tuning
- **Optuna** — 75 trials, 3-fold CV
- Best: `depth=3`, `lr=0.027`, `iterations=789`, `l2=10.66`
- Depth=3 confirmed optimal across 5 independent runs → feature ceiling established

### Calibration
- Isotonic regression calibration
- Brier score: `0.2229 → 0.0851` (62% improvement)

---

## 📈 Business Output

### 4-Quadrant Customer Segmentation

| Segment | Customers | True Churn Rate | Action |
|---------|-----------|-----------------|--------|
| **A: Priority Save** | 18,503 | **21%** | Immediate outreach — contract upgrade |
| B: Loyalty Reward | 81,523 | 7% | Loyalty program enrollment |
| C: Let Go / Monitor | 21,497 | 19% | Monitor only |
| D: Upsell Target | 78,477 | 8% | Pitch additional services |

### Retention ROI
| Metric | Value |
|--------|-------|
| Segment A customers | 18,503 |
| Revenue saved (24mo LTV) | **$11,235,264** |
| Campaign cost | $647,600 |
| **Net ROI** | **$10,587,664** |
| **ROI %** | **1,635%** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| ML | CatBoost, XGBoost, LightGBM, Scikit-learn |
| Tuning | Optuna (75 trials) |
| Explainability | SHAP |
| Imbalance | Class Weights 9:1 |
| UI | Gradio 4.44.0 |
| Deployment | HuggingFace Spaces |
| Compute | Kaggle (T4 GPU) |

---

## 🔍 Top SHAP Features

```
1. contract              SHAP = 0.4379  ← single biggest driver
2. payment_risk_score    SHAP = 0.1942
3. customer_satisfaction SHAP = 0.1524
4. has_online_security   SHAP = 0.1013
5. has_tech_support      SHAP = 0.0987
6. satisfaction_risk     SHAP = 0.0980
7. num_services          SHAP = 0.0627
8. service_vulnerability SHAP = 0.0572  ⭐ engineered
9. charge_per_service    SHAP = 0.0452
10. contract_sat_risk    SHAP = 0.0276  ⭐ engineered
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/Simos-ro-hub/Telecom-Customer-Churn-Prediction
cd telecom-customer-churn-prediction
pip install -r requirements.txt
# Add model_artifacts/ folder from HuggingFace Space
python app.py
```

---

## 👤 Author

**Simosro** — Final-year CSE student, North South University, Bangladesh

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Simosro-FF9A00?logo=huggingface)](https://huggingface.co/Simosro)

---

## 📄 License

MIT License — free to use, modify and distribute.
