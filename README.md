# In-Vehicle Coupon Strategy: How Persona Segmentation Cuts Coupon Waste by 72%

> Analyzing 12,684 driving scenarios revealed that coupon type, timing, and user persona are the three levers that determine whether a driver accepts or ignores an in-vehicle coupon

**[View Full Presentation (PDF)](docs/Group9_InVehicle_Coupon_Strategy_Presentation.pdf)** | [Project Report (PDF)](docs/Group9_InVehicle_Coupon_Strategy_Report.pdf)

---

## Background

When a driver receives a coupon notification while on the road, the decision to accept or ignore happens in seconds. Blanket coupon delivery — sending every coupon to every driver — results in a 43% rejection rate, wasting marketing budget and degrading user experience.

We analyzed 12,684 coupon delivery scenarios from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation), covering 5 coupon types, 25 contextual/behavioral/demographic features, and built a prediction + segmentation pipeline to answer:

1. **What makes a driver accept a coupon?**
2. **Can we identify distinct driver personas with different coupon preferences?**
3. **How much waste can targeted delivery eliminate vs. blanket distribution?**

---

## Three Key Findings

### Finding #1: Coupon Type and Context Are the Dominant Drivers — Not Demographics

**Bottom line**: Carry-out and cheap restaurant coupons in low-urgency contexts with longer expiry windows are most likely to be accepted. Bar coupons and short-window offers during work commutes are most likely to be rejected.

#### Evidence

Logistic Regression coefficient analysis reveals the feature hierarchy:

<img width="1589" height="690" alt="image" src="https://github.com/user-attachments/assets/72d4d8ae-03ba-4752-963f-bc22898e4cf6" />


| Feature | Direction | Interpretation |
|---|---|---|
| Carry out & Take away coupon | Strong positive | Highest acceptance signal |
| Restaurant(<$20) coupon | Strong positive | Low-cost = low-friction |
| Bar coupon | Strong negative | Strongest rejection signal |
| Destination: No Urgent Place | Positive | Relaxed context = more receptive |
| Expiration: 1 day | Positive | More time to decide = more acceptance |
| Expiration: 2 hours | Negative | Urgency backfires for coupons |
| Destination: Work | Negative | Commuters reject distractions |

**Key insight**: Giving drivers more urgency (shorter expiration) actually _lowers_ acceptance rate. This is counterintuitive — in traditional marketing, urgency drives action. But in a driving context, a 2-hour window feels like pressure, while a 1-day window feels like an option.

Occupation also plays a surprisingly large role: healthcare workers consistently accept, while production/legal occupations consistently reject.

---

### Finding #2: Five Distinct Driver Personas Exist — And They Want Different Things

**Bottom line**: K-Means clustering (K=5) on 8 demographic + behavioral features identified five driver personas with acceptance rates ranging from 51% to 66%. Targeting by persona improves model AUC by up to +0.95%.

#### The Five Personas

<img width="1183" height="490" alt="image" src="https://github.com/user-attachments/assets/602b9bc0-8dc5-49c4-8d34-1365e5749236" />

| Persona | Size | Accept Rate | Profile |
|---|---|---|---|
| **Young Singles** | 3,972 (31%) | 57% | Age ~26, no children, low income, moderate coffee/takeout |
| **Social Food Lovers** | 926 (7%) | 66% | Age ~31, high frequency across _all_ venue types |
| **Family Takeout** | 2,562 (20%) | 58% | Age ~34, has children, heavy takeout & coffee, avoids bars |
| **Conservative Families** | 4,085 (32%) | 51% | Age ~42, highest income, has children, low activity across all venues |
| **Young Bar Hoppers** | 1,139 (9%) | 62% | Age ~26, highest bar frequency, moderate other venues |

#### What Each Persona Wants

<img width="1033" height="489" alt="image" src="https://github.com/user-attachments/assets/aa23820a-17f8-43ba-abb0-5a47f762cbcb" />

- **Social Food Lovers** (66% accept) respond to almost everything — they're the easy wins
- **Conservative Families** (51% accept) are the hardest to convert — they reject Bar and Coffee House coupons but accept Carry-out
- **Young Bar Hoppers** (62% accept) are highly receptive to Bar coupons but ignore Restaurant($20–50)
- **Family Takeout** prefers Carry-out and Restaurant(<$20) — low-cost, family-friendly options
- **Young Singles** are moderate across the board, best reached via Coffee House coupons

#### Integration with Prediction

Adding cluster labels + cluster-distance features as inputs to classification models:

| Model | Baseline AUC | + Clustering AUC | Delta |
|---|---|---|---|
| Logistic Regression | 0.7228 | 0.7243 | +0.0015 |
| Random Forest | 0.8061 | 0.8156 | +0.0095 |
| **XGBoost** | **0.8223** | **0.8318** | **+0.0095** |

Clustering adds negligible value for linear models (LR already captures the variance), but provides meaningful lift for tree-based models that can exploit non-linear persona-feature interactions.

**Leakage prevention**: Scaler and KMeans fit on train set only, applied to test set — no data leakage.

---

### Finding #3: Who you send to matters more than what you send

The same coupon can produce dramatically different outcomes:

<img width="1390" height="528" alt="image" src="https://github.com/user-attachments/assets/e1a0ed45-87e8-496a-8094-8568ea8b8aa4" />

- Bar coupon → Conservative Families: 26% acceptance
- Bar coupon → Young Bar Hoppers: 80% acceptance

By moving from blanket strategy to persona-targeted delivery, average acceptance improves from 57% → 70% (+13%).


**Bottom line**: Using the XGBoost + Clustering model to select _which_ coupons to send to _which_ personas, we can reduce wasted coupons from 59,000 to 16,280 per 100,000 deliveries — a 72% reduction.

#### Scenario: Bar Coupon Campaign

<img width="889" height="491" alt="image" src="https://github.com/user-attachments/assets/48c68579-2529-48f4-91f9-ce76c5450059" />

| Strategy | Coupons Sent | Wasted (Rejected) | Waste Rate |
|---|---|---|---|
| Blanket (send to everyone) | 100,000 | 59,000 | 59% |
| Targeted (persona-based) | 16,280 | 4,500 | 28% |

Targeted delivery means: only send Bar coupons to **Young Bar Hoppers** and **Social Food Lovers** during evening hours (6PM/10PM) with 1-day expiration. Skip Conservative Families and work commuters entirely.

#### Timing Matters

Acceptance rate varies significantly by time of day and persona:

- **2PM and 6PM** are peak acceptance windows across all personas
- **7AM** is the worst time — commuters reject everything
- **Young Bar Hoppers** at **10PM** have the highest single-cell acceptance rate
- **1-day expiration** consistently outperforms **2-hour expiration** across all clusters

---

## Model Performance Summary

Three models trained twice each — baseline (raw features) vs. enhanced (with cluster features):

| Model | Test AUC | Test F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.723 | 0.730 | Well-calibrated but limited discriminative power |
| LR + Clustering | 0.724 | 0.731 | Negligible gain, clustering doesn't help linear models |
| Random Forest | 0.806 | 0.786 | Strong non-linear capture |
| RF + Clustering | 0.816 | 0.794 | Meaningful +1% AUC lift |
| XGBoost | 0.822 | 0.787 | Best baseline performance |
| **XGBoost + Clustering** | **0.832** | **0.795** | **Best overall — selected model** |

**XGBoost hyperparameters** tuned via `RandomizedSearchCV` with 5-fold stratified CV (scoring = ROC-AUC).

**Calibration**: Logistic Regression's probability outputs are well-calibrated — a predicted 70% acceptance reflects ~70% actual acceptance. Useful for ranking/scoring use cases.

**SHAP analysis** on the XGBoost model confirms that coupon type, expiration, and behavioral frequency features dominate predictions, with cluster features providing secondary lift.

---

## Actionable Recommendations

Based on the three findings above:

| Action | Expected Impact |
|---|---|
| Stop sending Bar coupons to Conservative Families and Family Takeout personas | Eliminate ~30% of Bar coupon waste |
| Use 1-day expiration instead of 2-hour for all coupon types | +8–12% acceptance rate lift |
| Concentrate delivery at 2PM and 6PM, avoid 7AM | +5–10% acceptance rate lift |
| Deploy XGBoost + Clustering model for real-time scoring | Predict acceptance with 0.83 AUC |
| Send Coffee House coupons to Young Singles, Carry-out to Families | Match coupon type to persona preference |

---

## About This Analysis

### Data

- **Source**: [UCI ML Repository — In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)
- **Size**: 12,684 records, 25 features (after dropping `car` column with 99% missing)
- **Target**: Binary — coupon accepted (Y=1) or rejected (Y=0)
- **Baseline acceptance rate**: 56.8%

### Methods

- **EDA**: Target distribution, acceptance by coupon type, behavioral/demographic breakdowns
- **Preprocessing**: Ordinal-to-numeric mapping, one-hot encoding via `ColumnTransformer`
- **Clustering**: K-Means (K=5) on 8 demographic + behavioral features, leakage-safe pipeline
- **Models**: Logistic Regression, Random Forest, XGBoost with `RandomizedSearchCV`
- **Interpretability**: Feature coefficients, feature importance, SHAP beeswarm plots
- **Prescriptive**: Cluster x Coupon targeting matrix, timing analysis, ROI scenario comparison

### Tech Stack

Python 3.10+ — pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn

### Reproduce

```bash
git clone https://github.com/Pickle1024/InVehicle-Coupon-Strategy.git
cd InVehicle-Coupon-Strategy
pip install -r requirements.txt
jupyter notebook notebooks/Group9_InVehicle_Coupon_Strategy.ipynb
```

---

<details>
<summary><b>Project Structure</b></summary>

```
InVehicle-Coupon-Strategy/
├── README.md
├── requirements.txt
├── data/
│   └── in-vehicle-coupon-recommendation.csv    # Raw dataset (12,684 rows)
├── notebooks/
│   └── Group9_InVehicle_Coupon_Strategy.ipynb   # Full analysis pipeline
└── docs/
    ├── Group9_InVehicle_Coupon_Strategy_Report.pdf
    └── Group9_InVehicle_Coupon_Strategy_Presentation.pdf
```

</details>
