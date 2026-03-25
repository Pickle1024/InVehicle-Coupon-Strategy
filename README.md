# In-Vehicle Coupon Recommendation Strategy

Predicting whether a driver will accept an in-vehicle coupon using machine learning, with cluster-based persona segmentation and actionable business strategy.

## Problem Statement

When a driver receives a coupon on their phone while driving, will they accept it? This project builds a **binary classification pipeline** to predict coupon acceptance (Y = 1) or rejection (Y = 0), then translates model insights into a targeted marketing strategy.

## Dataset

- **Source**: [UCI Machine Learning Repository — In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)
- **Size**: 12,684 records, 25 features
- **Coupon types**: Bar, Coffee House, Carry Out, Restaurant (<$20), Restaurant ($20–$50)
- **Features**: driving context (destination, weather, time), demographics (age, income, education), and behavioral frequency (bar visits, restaurant visits, etc.)

## Project Structure

```
├── data/
│   └── in-vehicle-coupon-recommendation.csv   # Raw dataset
├── notebooks/
│   └── Group9_InVehicle_Coupon_Strategy.ipynb  # Full analysis notebook
├── docs/
│   ├── Group9_InVehicle_Coupon_Strategy_Report.pdf
│   └── Group9_InVehicle_Coupon_Strategy_Presentation.pdf
├── requirements.txt
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis
- Target distribution (56.8% acceptance rate)
- Acceptance rate breakdowns by coupon type, demographics, and behavioral features
- Missing data handling: dropped `car` column (99% missing), imputed behavioral columns with "never"

### 2. Preprocessing
- Mapped ordinal categories to numeric midpoints (age, income, visit frequency)
- One-hot encoded remaining categorical features
- Built a scikit-learn `ColumnTransformer` pipeline for reproducibility

### 3. Clustering & Feature Augmentation
- K-Means clustering (K=5) on 8 demographic + behavioral features
- Identified 5 distinct user personas (e.g., Young Singles, Budget Families, Premium Diners)
- Cluster labels appended as features to the classification dataset
- **Leakage-safe**: scaler and KMeans fit on train set only

### 4. Model Training & Evaluation

Each model trained **twice** — baseline (no clusters) vs. enhanced (with cluster features):

| Model | Baseline AUC | Enhanced AUC |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

- Hyperparameter tuning via `RandomizedSearchCV` (5-fold stratified CV)
- Evaluation: ROC-AUC, F1, confusion matrix, feature importance
- SHAP analysis for XGBoost interpretability

### 5. Business Strategy
- **Cluster × Coupon targeting matrix**: which coupon to send to which persona
- **Timing analysis**: optimal send times by coupon type and persona
- **ROI scenario analysis**: targeted delivery vs. blanket delivery

## Getting Started

```bash
# Clone the repo
git clone https://github.com/PickLens/InVehicle-Coupon-Strategy.git
cd InVehicle-Coupon-Strategy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/Group9_InVehicle_Coupon_Strategy.ipynb
```

## Tech Stack

- **Python 3.10+**
- **Data**: pandas, NumPy, SciPy
- **Visualization**: matplotlib, seaborn
- **ML**: scikit-learn, XGBoost, SHAP
