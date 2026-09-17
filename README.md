# Telco Customer Churn Prediction

End-to-end data science project that predicts whether a telecom customer will
churn, using the IBM Telco Customer Churn dataset. The project covers the full
workflow — data understanding, cleaning, EDA, feature engineering, model
training/evaluation, interpretation — and serves the final model through a
FastAPI REST API.

## 1. Business Problem

Customer churn — a customer cancelling their phone/internet service — is
costly for a telecom company: acquiring a new customer is more expensive than
retaining an existing one, and churn also means lost recurring monthly
revenue. If customers likely to churn can be identified **before** they leave,
the retention team can proactively intervene (targeted offers, service
improvements, contract incentives) instead of reacting after the fact.

## 2. Objective

Build a classification model that, given a customer's account and service
attributes (contract type, tenure, billing method, services subscribed,
charges, etc.), predicts whether that customer is likely to churn, so the
retention team can rank and prioritize outreach to at-risk customers.

**Target variable:** `Churn` — `Yes` (customer left) / `No` (customer stayed).

## 3. Dataset Description

- **Source:** `data/TelcoCustomerChurn.csv` (with `data/TelcoCustomerChurn - Data Dictionary.csv` describing each column).
- **Size:** 7,043 customer records, 21 columns.
- **Feature groups:** demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account info (`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`), subscribed services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`), identifier (`customerID`), and the target (`Churn`).

## 4. Project Structure

```
telco-customer-churn/
├── data/
│   ├── TelcoCustomerChurn.csv                  # Raw dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb   # Full Day 1 analysis & modeling notebook
├── model/
│   └── churn_model.pkl        # Saved, fitted pipeline (preprocessing + Decision Tree)
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py # Shared TotalServices/tenure_group logic (used by notebook AND API)
│   └── schemas.py             # Pydantic request/response models for the API
├── app.py                     # FastAPI app: GET /health, POST /predict
├── requirements.txt
├── sample_request.json        # Example /predict request body
└── README.md
```

## 5. Data Understanding and Cleaning

- `df.dtypes` showed most columns as text/`object`, with `TotalCharges` unexpectedly stored as text rather than numeric.
- **Missing values:** `df.isnull().sum()` reported 0 nulls, but this was misleading — `TotalCharges` contained **11 blank/whitespace strings** (invisible to a plain null check). All 11 belong to customers with `tenure == 0` (brand-new signups, `Churn = No`) who have no billing history yet.
- **Duplicates:** 0 fully duplicated rows, 0 duplicate `customerID` values.
- **Redundant categories:** 7 columns (`MultipleLines` and the 6 internet-dependent add-on columns) contain a `"No phone/internet service"` category that is fully redundant with `PhoneService`/`InternetService`.
- **Suspicious values check:** no negative or out-of-range values found for `tenure`, `MonthlyCharges`, or `SeniorCitizen`.
- **Target distribution:** `Churn = No` for 5,174 customers (73.46%), `Churn = Yes` for 1,869 customers (26.54%) — a moderate ~3:1 class imbalance.
- **Cleaning applied:**
  - `TotalCharges` converted to numeric (`pd.to_numeric(..., errors="coerce")`); the resulting 11 `NaN`s imputed with `0` (a domain-derived constant, not a statistic — safe to apply before the train/test split).
  - `SeniorCitizen` re-mapped from `0`/`1` to `"No"`/`"Yes"` to match the other binary categorical flags.
  - `customerID` retained in the dataframe but excluded from the modeling feature set (not a data-quality issue — it's simply not predictive).

## 6. EDA Summary

Key churn-rate findings from the exploratory analysis:

| Factor | Finding |
|---|---|
| Contract type | Month-to-month 42.7% churn vs. One year 11.3% vs. Two year 2.8% |
| Internet service | Fiber optic 41.9% churn vs. DSL 19.0% vs. No internet 7.4% |
| Tenure | Falls from 47.4% (0-12 months) to 6.6% (61-72 months) |
| Monthly charges | Median $79.65 (churned) vs. $64.43 (stayed) |
| Payment method | Electronic check 45.3% churn vs. 15-19% for other methods |
| Add-on services (Security/Backup/Protection/Support) | Drops from 38.9% (1 add-on) to 5.3% (4 add-ons) |
| Senior citizen | 41.7% vs. 23.6% for non-seniors |
| Paperless billing | 33.6% vs. 16.3% for paper billing |

**Overall:** the clearest churn drivers are short-term contracts, fiber optic internet combined with higher monthly charges, manual/electronic-check payment, few add-on services, and senior citizen status. Churn risk is heavily concentrated in the first 12 months of tenure.

## 7. Feature Engineering

Two features were engineered (in `src/feature_engineering.py`, shared by both the notebook and the API so training and serving stay consistent):

1. **`TotalServices`** — count of `"Yes"` across the 8 phone/add-on service columns, plus 1 if the customer has any internet service. Range 0-9. Captures overall engagement/switching cost.
2. **`tenure_group`** — `tenure` bucketed into 6 ranges (`0-12`, `13-24`, `25-36`, `37-48`, `49-60`, `61-72`), formalizing the strongest single relationship found in the EDA.

Both are computed only from pre-existing predictor columns (no leakage from `Churn`) and are fully computable for a brand-new customer at signup.

## 8. Data Preprocessing

A `ColumnTransformer` with three branches, fit only on the training set:

- **Numeric** (`tenure`, `MonthlyCharges`, `TotalCharges`, `TotalServices`): median imputation. No scaling — a Decision Tree splits on raw thresholds and is invariant to monotonic transforms.
- **Nominal categorical** (16 columns — gender, Contract, PaymentMethod, etc.): most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`, so an unseen category at prediction time is encoded as all-zeros instead of erroring.
- **Ordinal categorical** (`tenure_group` only): most-frequent imputation + `OrdinalEncoder` with the bucket order fixed explicitly, preserving `0-12 < 13-24 < ... < 61-72`.

This produces 48 numeric output features. The whole preprocessor is wrapped in a `Pipeline` alongside the classifier, so the exact same fitted object is reused for training, testing, and every future API request.

## 9. Train/Test Split

70:30 split, stratified on `Churn`, `random_state=42`:

- `X_train`: 4,930 rows
- `X_test`: 2,113 rows
- Churn ratio preserved almost exactly: 73.47%/26.53% (train) vs. 73.45%/26.55% (test), matching the full dataset's 73.46%/26.54%.

## 10. Decision Tree Configurations

Two configurations were compared, varying `max_depth`, `min_samples_split`, and `min_samples_leaf` together:

- **Model A (shallow, regularized):** `max_depth=4`, `min_samples_split=50`, `min_samples_leaf=25`.
- **Model B (deeper, flexible):** `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5`.

Both use `random_state=42`; `criterion` and `class_weight` were left at defaults for both.

## 11. Model Comparison

| Metric | Model A (shallow) | Model B (deeper) |
|---|---|---|
| Accuracy | 0.7918 | 0.7653 |
| Precision (Churn=Yes) | 0.6580 | 0.5619 |
| Recall (Churn=Yes) | 0.4492 | 0.5258 |
| F1 (Churn=Yes) | 0.5339 | 0.5433 |
| Train − Test accuracy gap | 0.0060 | 0.0945 |

Model A has higher accuracy/precision; Model B has higher recall and a marginally higher F1. Model B's larger train/test gap indicates more overfitting.

## 12. Final Model Selection

**Model B (deeper, flexible)** was selected, despite its lower accuracy and larger overfitting gap, because in a retention context a missed churner (false negative) is more costly than an unnecessary retention offer (false positive) — so higher recall on the churn class is prioritized. Model A remains a valid, more conservative alternative if stability/interpretability were prioritized instead.

## 13-17. Final Model Evaluation Metrics

Evaluated on the 2,113-row test set:

| Metric | Value |
|---|---|
| Accuracy | 0.7653 |
| Precision (Churn=Yes) | 0.5619 |
| Recall (Churn=Yes) | 0.5258 |
| F1 (Churn=Yes) | 0.5433 |

## 18. Confusion Matrix Interpretation

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | TN = 1,322 | FP = 230 |
| **Actual Yes** | FN = 266 | TP = 295 |

- **True Positives (295):** correctly flagged churners — actionable for retention.
- **True Negatives (1,322):** correctly identified as staying — no wasted effort.
- **False Positives (230):** flagged as at-risk but stayed — a low-cost "wasted" retention offer.
- **False Negatives (266):** actual churners missed entirely — a lost customer with no chance to intervene.

Of the 561 customers who actually churned, the model catches 295 (52.6% recall) and misses 266 (47.4%). Of the 525 customers flagged as likely to churn, 295 (56.2% precision) actually churned.

## 19. Precision vs. Recall Business Reasoning

A false negative (missed churner) represents fully lost recurring revenue with no chance to intervene, while a false positive (unnecessary retention offer) only costs a relatively cheap gesture to a customer who was staying anyway. This asymmetry means **recall should generally be prioritized over precision** for churn identification — the reason Model B (higher recall) was chosen over Model A (higher precision) despite Model A's better accuracy.

## 20-21. Feature Importance / Top Features

The top 5 features account for ~78% of total importance:

1. **`Contract_Month-to-month` (0.3164)** — by far the strongest driver; matches the EDA's contract-type finding.
2. **`tenure` (0.1599)** — newer customers churn more, consistent with the EDA.
3. **`TotalCharges` (0.1112)** — correlated with tenure; likely captures overlapping "customer longevity" signal.
4. **`MonthlyCharges` (0.0967)** — higher bills associated with higher churn.
5. **`InternetService_Fiber optic` (0.0948)** — matches the EDA's fiber-optic churn finding.

## 22. Model Interpretation

The tree's root split is `Contract_Month-to-month`, confirming it as the single most decisive factor: non-month-to-month customers churn at 6.7% vs. 42.9% for month-to-month customers. Among month-to-month customers, the next split is `InternetService_Fiber optic`: those with fiber already tip into majority-churn territory (54.8%). The top-3-level tree visualization mirrors this reasoning as a human-readable flowchart, corroborating the EDA rather than contradicting it.

**Known limitations of this interpretation** (see notebook Section 12.3 for full detail): feature importance shows no direction of effect on its own, numeric features are structurally favored over categorical ones, correlated features (`tenure`/`TotalCharges`) split credit, importances can be unstable across configurations, and interactions between features aren't captured by importance scores alone.

## 23. Saved Model/Pipeline

The entire fitted `Pipeline` (preprocessing `ColumnTransformer` + `DecisionTreeClassifier`, i.e. Model B) is saved with `joblib` to `model/churn_model.pkl`. Saving the whole pipeline — not just the classifier — ensures the exact fitted imputers/encoders travel with the model, so a separate process (the API) can score raw customer records without reimplementing preprocessing and risking train/serve skew.

> **Note:** `model/churn_model.pkl` currently holds the bonus `Tuned + class_weight='balanced'` pipeline (same `preprocessor`/`classifier` step names), not Model B — see [Bonus: Hyperparameter Tuning & Class Imbalance Handling](#bonus-hyperparameter-tuning--class-imbalance-handling) below for how and why it was superseded.

## 24-25. FastAPI API — POST /predict

`app.py` exposes:

- `GET /health` — liveness check, returns `{"status": "ok", "model_loaded": true}`.
- `POST /predict` — accepts a raw customer record (19 fields, validated by the `CustomerData` Pydantic model in `src/schemas.py`), then:
  1. Converts it to a single-row DataFrame.
  2. Applies `engineer_features()` from `src/feature_engineering.py` (the same function used in the notebook) to compute `TotalServices`/`tenure_group`.
  3. Passes the result through the saved pipeline's `predict()`/`predict_proba()`.
  4. Returns `prediction` (`"No"`/`"Yes"`) and `churn_probability` (probability of the `"Yes"` class, read from the model's actual `classes_` order rather than an assumed index).

The model is loaded once at process startup, never retrained per-request. Invalid input (missing fields, wrong types, invalid categorical values, out-of-range numbers) is rejected by Pydantic validation with an HTTP 422 response before reaching the model.

## 26. Setup Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 27. How to Run the Notebook

Before running any cells, make sure the notebook is using the `.venv` environment created in Setup Instructions, not a different global/system Python:

1. Open `notebook/churn_analysis.ipynb` in VS Code.
2. Click the kernel picker in the top-right of the notebook (or run **Notebook: Select Notebook Kernel**) and choose the Python environment at `.venv/bin/python`.
3. Verify the correct environment is active by running `import sys; print(sys.executable)` in a cell — the printed path should point inside the project's `.venv` folder.

Alternatively, run from the terminal after activating the environment:

```bash
source .venv/bin/activate
jupyter notebook notebook/churn_analysis.ipynb
```

Run all cells top to bottom; the final cells save the fitted pipeline to `model/churn_model.pkl`, which the API depends on.

## 28. How to Run the API

```bash
uvicorn app:app --reload
```

The API becomes available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`. `model/churn_model.pkl` must already exist (produced by running the notebook) before starting the API.

## 29-30. Sample Request and Response

`sample_request.json` (a real customer record from `data/TelcoCustomerChurn.csv`):

```json
{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 28,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 104.80,
  "TotalCharges": 3046.05
}
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

Response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.788
}
```

This matches the dataset's actual recorded `Churn = Yes` label for this customer. (Probability reflects the bonus `Tuned + class_weight='balanced'` model currently saved in `model/churn_model.pkl` — see the Bonus section below; it differs from the original Model B's probability for the same request.)

## 31. Limitations

- **Recall is 52.6%** — nearly half of actual churners are still missed by the final model.
- **Model B overfits more than Model A** (train/test accuracy gap of 0.0945 vs. 0.0060), suggesting some of its recall advantage could be improved further through tuning rather than accepted as-is.
- **Feature importance limitations:** no direction-of-effect information, bias toward numeric/high-cardinality features, correlated features (`tenure`/`TotalCharges`) splitting credit, and instability across different tree configurations.
- **No hyperparameter tuning (grid search/cross-validation)** was performed on Model A/B — only two hand-picked configurations were compared. *(Addressed as a bonus — see [Bonus: Hyperparameter Tuning & Class Imbalance Handling](#bonus-hyperparameter-tuning--class-imbalance-handling) below, which supersedes Model B as the saved/served model.)*
- **Class imbalance** (73.46%/26.54%) was originally handled only via stratified splitting, not via resampling or `class_weight`. *(Also addressed in the Bonus section below.)*
- The API's broad `except Exception` around prediction always returns HTTP 400, which doesn't distinguish genuine server-side bugs from bad input.

## 32. Future Improvements

- Add cross-field validation and stricter error-status separation (e.g. 500 for genuine server errors vs. 400 for business-rule prediction failures) to the API.
- Try alternative model families (e.g. Random Forest, Gradient Boosting) for comparison against the single Decision Tree.
- Explore SMOTE or other resampling techniques as an alternative/complement to `class_weight="balanced"` (see Bonus section below).
- Expose the deployed model's config/version via the API response or a `/model-info` endpoint, so callers can tell which underlying model produced a prediction.

## Bonus: Hyperparameter Tuning & Class Imbalance Handling

This section documents an additional bonus task built on top of the Section 1-32 assignment above — same `X_train`/`X_test` split (`random_state=42`), same preprocessing pipeline, no changes to the original EDA/feature engineering/baseline work. Implemented in notebook Section 14.

**1. Hyperparameter tuning (`GridSearchCV`):** searched `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion` using 5-fold `StratifiedKFold` cross-validation, scored on F1 (Churn=Yes), fit only on `X_train`/`y_train` (`X_test` untouched). Best parameters found: `max_depth=6, min_samples_leaf=5, min_samples_split=50, criterion="entropy"` (best CV F1 = 0.5815).

**2. Class imbalance handling:** `class_weight="balanced"` was added on top of those same tuned hyperparameters, isolating the effect of imbalance handling from the hyperparameter choice. No SMOTE/resampling was used — the ~2.77:1 imbalance is moderate, not extreme.

**3. Model comparison — all four models evaluated fairly on the same untouched test set:**

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) |
|---|---|---|---|---|
| Model A (baseline) | 0.7918 | 0.6580 | 0.4492 | 0.5339 |
| Model B (baseline) | 0.7653 | 0.5619 | 0.5258 | 0.5433 |
| Tuned (GridSearchCV) | 0.7856 | 0.6076 | 0.5437 | 0.5738 |
| Tuned + `class_weight='balanced'` | 0.7397 | 0.5064 | 0.7772 | 0.6132 |

**4. Final model selection is leakage-safe:** the winner among the four candidates is chosen using 5-fold cross-validated F1 computed on `X_train`/`y_train` only (the same CV setup as tuning) — **not** the test-set table above. Selecting a model by comparing test-set scores across candidates would itself be a form of data leakage through model selection, even though no row is used for fitting.

| Model | CV F1 mean (Churn=Yes) | CV F1 std |
|---|---|---|
| Model A | 0.5378 | 0.0205 |
| Model B | 0.5519 | 0.0239 |
| Tuned | 0.5815 | 0.0327 |
| Tuned + `class_weight='balanced'` | 0.5978 | 0.0165 |

`Tuned + class_weight='balanced'` wins on cross-validated F1, so it is selected as the final model. It is evaluated on the untouched test set exactly once afterward (the metrics in row 4 of the table above — Accuracy 0.7397, Precision 0.5064, Recall 0.7772, F1 0.6132, confusion matrix TN 1127 / FP 425 / FN 125 / TP 436), and the fitted pipeline is saved to `model/churn_model.pkl`, replacing the original Section 12 Model B artifact.

**Impact vs. the original Model B baseline:** recall improves from 0.5258 to 0.7772 (+0.25) and F1 from 0.5433 to 0.6132 (+0.07), at the cost of accuracy (-0.026) and precision (-0.056) — consistent with the recall-first business framing already established in Sections 10-11 (a missed churner costs more than an unnecessary retention offer).

**Limitations of the bonus work:**

- Only Decision Tree hyperparameters were tuned — no other model families (Random Forest, Gradient Boosting, etc.) were compared.
- `class_weight="balanced"` was the only imbalance-handling technique tried; SMOTE/other resampling was not explored.
- The `GridSearchCV` grid (`max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`) is not exhaustive; a finer or wider search could find a different optimum.
- The balanced model's precision (0.5064) means close to half of its positive predictions are false alarms — a real operational cost for the retention team, even though it's cheaper than a missed churner.
- `app.py`/`model/churn_model.pkl` required no code changes (the API only calls `predict()`/`predict_proba()`/`named_steps["classifier"].classes_`), but the API still doesn't expose which model config is deployed — a caller can't tell from the response alone.
