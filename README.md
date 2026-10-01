# ATE Retest-Benefit Prediction AI

Supervised ML system for semiconductor ATE (Automatic Test Equipment) failure events.

When an ATE test fails, the system uses **only information available before a physical retest**, predicts the probability that a retest would be beneficial, applies a separately defined decision policy, and later scores that recommendation against the actual outcome.

```text
ATE FAIL
    → Pre-retest parameters
    → ML model
    → P(RETEST_BENEFICIAL)
    → DOCX-reference 30% decision policy
    → RETEST / DON'T RETEST
    → Actual outcome later
    → Validation
    → KPI cards
```

> The ML model is **not** an LLM. XGBoost / Gradient Boosting / Logistic Regression are candidate classifiers. The application behaves as an AI agent because it runs **prediction → recommendation → validation** as one workflow.

---

## Business questions

The system answers two different questions without mixing them:

| # | Question | Output |
|---|---|---|
| 1 | Should this failed event be sent for retest? | `P(RETEST_BENEFICIAL)` + isolated `RETEST` / `DON'T RETEST` recommendation |
| 2 | How well has the AI been making those decisions? | Post-outcome comparison of recommendation vs `Ground_Truth` (TP / FP / FN / TN + KPIs) |

---

## Quick start

### Prerequisites

- Python 3.10+
- pip

### Install

```powershell
cd ATE-Retest-Benefit-Prediction-AI
pip install -r retest_ai/requirements.txt
```

### Run Streamlit UI

From the **project root**:

```powershell
streamlit run retest_ai/app.py
```

Opens at http://localhost:8501

### Run FastAPI

From the **project root**:

```powershell
uvicorn retest_ai.api.main:app --reload
```

| Resource | URL |
|---|---|
| Service | http://127.0.0.1:8000 |
| Swagger docs | http://127.0.0.1:8000/docs |
| Health | `GET /health` |
| Model info | `GET /model/info` |
| Single predict | `POST /predict` |
| Batch predict | `POST /predict/batch` |

### Run tests

From the **project root**:

```powershell
python -m unittest discover -s retest_ai/tests -p "test_*.py"
```

---

## Repository layout

```text
ATE-Retest-Benefit-Prediction-AI/
├── README.md
├── LICENSE
├── PROJECT_SPECIFICATION.md
├── inspect_datasets.py
├── ATE_Retest_50_Devices_Month_0_Historical.xlsx
├── ATE_Retest_50_Devices_Month_6_Historical.xlsx
├── ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx
├── ATE_Retest_50_Devices_AI_Dataset.xlsx
├── RETEST~2.docx                          # Reference report (DOCX baseline KPIs)
└── retest_ai/
    ├── app.py                             # Streamlit UI
    ├── requirements.txt
    ├── artifacts/                         # Persisted model artifacts
    ├── api/                               # FastAPI (routes, schemas)
    ├── config/                            # Feature whitelist, paths, DOCX KPI baselines
    ├── data/                              # Ingestion, preprocessing, validation
    ├── models/                            # Training, evaluation, calibration, online learning
    ├── decision/                          # Isolated 30% DOCX-reference policy
    ├── explainability/                    # SHAP contributions
    ├── kpis/                              # Decision quality + business impact
    ├── validation/                        # Post-outcome recommendation scoring
    └── tests/
```

---

## Dataset structure

| File | Role |
|---|---|
| `ATE_Retest_50_Devices_Month_0_Historical.xlsx` | Labeled history. **Train** for temporal validation. |
| `ATE_Retest_50_Devices_Month_6_Historical.xlsx` | Labeled history. **Temporal holdout** (not mixed randomly with Month 0). |
| `ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx` | Unseen events. **Inference only**. No ground truth required. |
| `ATE_Retest_50_Devices_AI_Dataset.xlsx` | Historical snapshot used to **recompute** the supplied DOCX reference KPIs. |
| `Month_12_PRIVATE_VALIDATION_ONLY.xlsx` | Optional later outcomes. **Never** used as prediction input. |

**Deployment flow**

```text
Month 0 historical  +  Month 6 historical
            ↓
         LEARN (supervised)
            ↓
     Month 12 new data
            ↓
   P(RETEST_BENEFICIAL)
            ↓
   RETEST / DON'T RETEST
            ↓
 Explanation + KPI + impact
```

Final deployment model is trained on Month 0 + Month 6, then used for Month 12 inference.

Prototype scale: ~50 devices, ~125 events per month file (synthetic / not plant-validated).

---

## Pre-retest feature whitelist

Only fields observable **before** a retest enter the model:

| Type | Features |
|---|---|
| Categorical | `Wafer_ID`, `ATE_Site`, `Fail_Test`, `Fail_Bin`, `First_Result` |
| Numerical | `Voltage_V`, `Temperature_C`, `First_Test_Time_sec`, `Test_Month` |

Identifiers kept for tracking / UI only (never in feature matrix `X`):

- `Device_ID`
- `Failure_Event`

---

## Leakage prevention

These fields must **never** enter the prediction feature matrix `X`:

- `Ground_Truth`, `Retest_Result`, `Final_Result`, `Retest_Count`
- `True_Retest_Pass_Probability`, `AI_Retest_Probability`, `AI_Recommendation`
- `Retest_Time_sec` (actual retest execution time)

The API rejects them as inputs. Training workbooks may contain labels; those labels are used only as the supervised target or for later validation — not as features.

---

## Training & model selection

### Target

Supervised target: `Ground_Truth`

```text
RETEST_BENEFICIAL  →  1
PERSISTENT_FAILURE →  0
```

Model output: `P(RETEST_BENEFICIAL)` in `[0, 1]`. Probabilities are not rewritten to look better in the UI.

### Temporal validation

```text
Month 0  →  training
Month 6  →  temporal holdout evaluation
Month 12 →  final unseen inference
```

This is historical temporal validation, **not** Month 12 performance.

### Candidate models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Gradient Boosting | Challenger |
| XGBoost | Primary candidate for larger tabular datasets |

Selection is evidence-based on Month 6 holdout (ROC-AUC, then Brier Score). XGBoost is **not** declared best unless measured results support it.

Calibration (Brier, Log Loss, reliability buckets) is evaluated, not used to invent a new probability.

A **0.5 evaluation/reporting cutoff** may appear in model-comparison expanders only. It does **not** produce operational RETEST / DON'T RETEST recommendations.

---

## Decision policy (isolated)

Defined in `retest_ai/decision/decision_policy.py` — separate from the ML model.

```text
if P(RETEST_BENEFICIAL) >= 0.30  →  RETEST
if P(RETEST_BENEFICIAL) <  0.30  →  DON'T RETEST
```

**Label:** Reference / DOCX decision policy — subject to validation

This is the 30% logic referenced in `RETEST~2.docx`. It is **not** a scientifically proven or permanently approved production threshold. Changing the policy does **not** require retraining the model. The probability is not modified when the recommendation is applied.

There is no 50% operational threshold and no threshold slider for production recommendations.

---

## Post-outcome validation

After the actual outcome is known:

```text
AI recommendation + actual Ground_Truth
    → TP  Correct Retest
    → FP  Unnecessary Retest
    → FN  Missed Opportunity
    → TN  Correct Skip
```

Month 12 outcome KPIs appear only after outcomes are loaded separately (`Month_12_PRIVATE_VALIDATION_ONLY.xlsx`).

---

## KPI calculations

All dynamic (not hard-coded from the DOCX as live performance):

| KPI | Formula |
|---|---|
| Accuracy | `(TP + TN) / Total` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| Specificity | `TN / (TN + FP)` |
| Unnecessary Retests | FP count and `FP / Total` |
| Missed Opportunities | FN count and `FN / Total` |

DOCX figures (e.g. 70.4% accuracy on 125 events) are **historical report reference values**. They are listed for audit and recomputed from the AI dataset. They are **not** shown as current Month 12 performance.

Default ATE cost rate for impact KPIs: **$1800 / hour** (configurable; not a measured plant rate).

---

## Streamlit UI pages

| Page | What it shows |
|---|---|
| **Single Event** | Event info, `P(RETEST_BENEFICIAL)`, DOCX-reference recommendation, SHAP contributions |
| **Month 12** | Event-level probability + recommendation, filters, CSV / Excel export |
| **Overview** | Month 12 prediction KPIs; clickable outcome KPIs and 2×2 matrix on labeled Month 6 data |
| **Historical Temporal Validation** | Model comparison (Month 0 train → Month 6 holdout) |
| **Reference Report** | DOCX listed vs recomputed historical KPIs |

SHAP explains **model feature contribution**, not physical root cause.

---

## FastAPI contract (summary)

### `POST /predict`

Accepts **only** approved pre-retest fields. Returns:

- `P(RETEST_BENEFICIAL)` (and percent)
- `recommendation` (`RETEST` / `DON'T RETEST`)
- policy threshold + policy label
- optional explanation fields

### `POST /predict/batch`

Same prediction contract for a list of events.

### `GET /model/info`

Active model name/version, feature whitelist, training metadata.

Leakage / forbidden columns in a request → **HTTP 422**.

---

## Dependencies

See `retest_ai/requirements.txt`:

- `pandas`, `numpy`, `scikit-learn`, `xgboost`
- `shap`, `openpyxl`
- `streamlit`, `plotly`
- `fastapi`, `uvicorn`, `httpx`, `pydantic`

---

## Known limitations

- Prototype / synthetic-scale ATE dataset (50 devices, ~125 events per month file).
- The 30% recommendation rule is a **DOCX-reference policy subject to validation**, not a certified production threshold.
- Month 12 accuracy cannot be claimed until actual Month 12 outcomes are provided separately.
- SHAP explains model contribution, not physical root cause.
- Temporal holdout uses 125 Month 6 events; calibration buckets are small and should be treated as directional.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Further reading

- [`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md) — full product / engineering specification
- [`retest_ai/README.md`](retest_ai/README.md) — package-level notes
