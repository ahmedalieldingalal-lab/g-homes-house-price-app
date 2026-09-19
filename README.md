# G-Homes House Price Intelligence

A Streamlit app that takes a house's details, classifies it into a physical/price
segment, predicts its sale price with a calibrated confidence interval, and explains
the result in plain English (optionally narrated by an LLM).

This app is a graduation-project deliverable. Every real computation — validation,
classification, price + interval, SHAP attribution, comparable-sales lookup, confidence
tiering, and narrative generation — lives in the `step6_*.py` / `consolidated_pipeline.py`
modules; `app.py` is a thin Streamlit UI layer on top of them.

## Data & modeling summary

- **Dataset:** Kaggle's [`shree1992/housedata`](https://www.kaggle.com/datasets/shree1992/housedata)
  (not the more common King County `kc_house_data` file). It lacks `grade`/`lat`/`long`,
  which is disclosed and worked around with a condition-based premium-outlook score and
  city/zip target encoding.
- **Model A (physical clustering):** MinMaxScaler → PCA (≥80% variance) → KMeans,
  4 physical clusters ("Untouched Classics", "Refreshed & Finished Below", "Untouched,
  Finished Below", "Refreshed Classics").
- **Model B (price bands):** nested quantile bands within each physical cluster.
- **Classifier A / B:** recover physical cluster / price band from observable features.
  The deployed price-band classifier is a Random Forest (CV macro-F1 0.9051, test
  macro-F1 0.8903, test accuracy 90.75%), chosen over CatBoost (statistically tied)
  for a lighter deployment footprint.
- **Price regression:** tuned GBM with a calibrated 90% interval via split conformal
  prediction.
- **Confidence tiering:** every prediction is tagged with a data-driven confidence tier
  (well-supported / moderately-supported / edge-case) based on comparable-sales
  evidence and how far the input sits from the training distribution.
- **Testing:** an automated suite of 3,033 checks (`test_full_app_audit.py`) plus 45
  additional checks (`test_beyond_training_range.py`) covering 160 representative
  houses across all 4 physical clusters, 0 failures.

See the project's Model Card and Study Guide (delivered separately) for full detail,
methodology, and limitations.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app needs **no API key to run** — with no Gemini key configured it renders a
deterministic, plain-English "Machine-Statistics (generated without AI)" narrative,
which is the app's designed default path and fully satisfies the project's deployment
requirement on its own.

To additionally enable the optional LLM-narrated explanation layer on top of that
(via Google's Gemini API), set an environment variable before running:

```bash
export GEMINI_API_KEY=your-key-here     # macOS/Linux
setx GEMINI_API_KEY your-key-here       # Windows (new shell after)
```

## Deploying on Streamlit Community Cloud

1. Point Streamlit Community Cloud at this repository, branch `main`, entry point
   `app.py`.
2. **Do not commit an API key to this repository.** To enable the optional LLM
   narrative layer on the deployed app, add it instead through the app's own
   **Settings → Secrets** panel on Streamlit Community Cloud, as:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

   Streamlit Community Cloud exposes root-level secrets as regular environment
   variables inside the running app, so no code change is needed — `app.py` already
   reads `GEMINI_API_KEY` via `os.environ`. Leaving this secret unset is fine: the app
   runs perfectly well without it, using its deterministic narrative path.

## Repository contents

- `app.py` — the Streamlit UI entry point.
- `consolidated_pipeline.py`, `step6_*.py` — the shared inference pipeline (cleaning,
  feature engineering, classification, regression, confidence, SHAP, narrative).
- `saved_models/` — trained model artifacts (classifiers, conformal regressors,
  cluster/classification metadata) used at inference time. Nothing here is retrained
  by the app; everything is loaded read-only.
- `consolidated_output/enriched_dataset.csv` — the cleaned, feature-engineered
  training/test dataset, used for the comparable-sales lookup shown in the app.
- `test_full_app_audit.py`, `test_beyond_training_range.py`, `test_yr_built.py`,
  `test_yr_renovated.py` — the project's automated test suite (run with
  `python3 <file>.py`; not part of the deployed app itself).
- `measure_confidence_distribution.py` — the analysis script used to produce the
  published confidence-tier distribution.

## Disclaimer

This is an educational/graduation project, not a production valuation tool. Prices are
statistical estimates from a fixed historical dataset and should not be used for real
real-estate decisions.
