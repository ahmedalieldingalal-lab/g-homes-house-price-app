"""Phase A, step 7 (part 1) -- the shipped 90% CQR (conformalized quantile
regression) interval for one new house, using the SAME three saved models
and Mondrian band-correction the training pipeline built
(`regression_pipeline.build_conformal_branch_cv`), reading its exact
parameters (`cutpoints_dollar`, `q_hat_by_band`) from
`saved_models/regressors/manifest.json` rather than re-deriving them.

AUDIT FIX 2026-09-09 (SB12, final audit): this docstring used to
attribute the shipped interval to `build_conformal_branch` (the
superseded SPLIT-conformal function, `regression_pipeline.py:357`) --
the same cross- vs. split-conformal prose-confusion class as Ledger C's
D7 / Ledger B's N6, surviving in this layer too. Verified directly:
`manifest.json["conformal_interval"]["method"]` is
`"cross_conformal_pooled_oof"`, which only `build_conformal_branch_cv`
(`regression_pipeline.py:511`, manifest write at `:681`) writes --
`build_conformal_branch` never runs today. No numeric consequence from
this mislabeling: both functions share the identical digitize-then-
quantile-adjust formula and the 0=low/1=mid/2=high band-index
convention (confirmed by reading both bodies), so the formula
transcribed below was always correct -- only the function name and line
numbers attributed to it were wrong.

Formula (copied verbatim from `build_conformal_branch_cv`, lines ~649-654 --
the same digitize-then-quantile-adjust pattern the superseded
`build_conformal_branch` also has at ~470-475, since `_cv` repeats it once
for out-of-fold calibration and once for final test-set scoring):
    predicted_price = expm1(point_model.predict(X))
    band = digitize(predicted_price, cutpoints_dollar)   # 0=low,1=mid,2=high
    q_hat = q_hat_by_band[band]
    lo_log = quantile_lo_model.predict(X) - q_hat
    hi_log = quantile_hi_model.predict(X) + q_hat
    cqr_lo, cqr_hi = expm1(lo_log), expm1(hi_log)

Step 7 also confirms (and records here, so it is never rediscovered) that
`conformal_cqr/point_gbm.joblib` and `regressors/gradient_boosting_gbm.
joblib` are the SAME model -- already independently verified by an Opus
audit agent (identical params, $0 prediction difference across all 908
test rows). This module always loads the `conformal_cqr` copy, since it's
the one that also has its quantile-model siblings alongside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# AUDIT FIX 2026-09-09 (SB4, final audit): anchored to this file's own
# location instead of the process cwd -- see step6_classify_house.py's
# identical fix for the full rationale.
REPO_ROOT = Path(__file__).resolve().parent
REGRESSOR_MANIFEST_PATH = REPO_ROOT / "saved_models" / "regressors" / "manifest.json"
POINT_MODEL_PATH = REPO_ROOT / "saved_models" / "regressors" / "conformal_cqr" / "point_gbm.joblib"
QUANTILE_LO_PATH = REPO_ROOT / "saved_models" / "regressors" / "conformal_cqr" / "quantile_lo_gbm.joblib"
QUANTILE_HI_PATH = REPO_ROOT / "saved_models" / "regressors" / "conformal_cqr" / "quantile_hi_gbm.joblib"

_MODELS_CACHE: dict | None = None


def _check_feature_contract(model, expected_features: list[str], model_name: str) -> None:
    """AUDIT FIX 2026-09-09 (SB9, final audit): matches step6_classify_
    house.py's identical fix -- runtime model loading here was entirely
    trust-based (a bare `joblib.load`), with nothing checking that these 3
    saved GBM models still agree with `manifest["feature_order"]` (the
    same 35-name order `step6_inference_contract.build_feature_row`
    assembles by hand into a plain numpy array, bypassing sklearn's own
    incidental column-name protection). Raises loudly and specifically at
    load time instead of silently scoring with columns in the wrong slots."""
    if not hasattr(model, "feature_names_in_"):
        return
    actual = list(model.feature_names_in_)
    if actual != expected_features:
        raise RuntimeError(
            f"{model_name}'s saved feature contract no longer matches manifest.json's feature_order -- "
            f"refusing to load, since silently continuing would score houses with values in the wrong "
            f"feature slots.\n"
            f"  Model's actual feature_names_in_: {actual}\n"
            f"  manifest.json's feature_order:    {expected_features}\n"
            f"This means the shipped model was retrained without updating (or regenerating) "
            f"manifest.json -- rebuild it, then re-verify."
        )


def load_cqr_models() -> dict:
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        with open(REGRESSOR_MANIFEST_PATH) as f:
            manifest = json.load(f)
        conformal = manifest["conformal_interval"]
        feature_order = manifest["feature_order"]
        point_model = joblib.load(POINT_MODEL_PATH)
        q_lo_model = joblib.load(QUANTILE_LO_PATH)
        q_hi_model = joblib.load(QUANTILE_HI_PATH)
        _check_feature_contract(point_model, feature_order, "conformal_cqr/point_gbm")
        _check_feature_contract(q_lo_model, feature_order, "conformal_cqr/quantile_lo_gbm")
        _check_feature_contract(q_hi_model, feature_order, "conformal_cqr/quantile_hi_gbm")
        _MODELS_CACHE = {
            "point_model": point_model,
            "q_lo_model": q_lo_model,
            "q_hi_model": q_hi_model,
            "cutpoints_dollar": conformal["cutpoints_dollar"],
            "q_hat_by_band": conformal["q_hat_by_band"],  # keys "low"/"mid"/"high"
        }
    return _MODELS_CACHE


def predict_with_interval(X_row: pd.DataFrame, models: dict | None = None) -> dict:
    """`X_row` is a single-row DataFrame in the regressor's exact
    `feature_order` (e.g. from `step6_inference_contract.build_feature_row`).
    Returns predicted_price plus the 90% CQR interval, all in dollars.

    AUDIT FIX 2026-09-09 (SB22, final audit): `regression_pipeline.py:657`
    (`build_conformal_branch_cv`) clamps its point prediction with
    `np.maximum(np.expm1(test_pred_log), 0.0)`; this function used to do a
    bare `np.expm1` with no clamp on `predicted_price`, `cqr_lo`, or
    `cqr_hi`. Added the same clamp here for consistency with the training
    pipeline, plus a runtime assertion that the interval and point
    estimate stay correctly ordered (`cqr_lo <= predicted_price <=
    cqr_hi`) -- training-time verification (500 real houses + 384
    Tier-A-grid corner houses, including the cheapest reachable corner,
    min `cqr_lo` = $21,299.89) found zero quantile-crossing or point-
    outside-interval cases and zero negative values, so neither this
    assertion nor the clamp are expected to ever fire in practice; both
    exist so a genuine violation (e.g. from a future retrain) is caught
    loudly here instead of silently reaching the UI as, for example, a
    negative dollar amount or a visually broken interval bar."""
    models = models if models is not None else load_cqr_models()

    pred_log = float(models["point_model"].predict(X_row)[0])
    predicted_price = max(float(np.expm1(pred_log)), 0.0)

    band_idx = int(np.digitize([predicted_price], models["cutpoints_dollar"])[0])  # 0, 1, or 2
    band_name = ["low", "mid", "high"][band_idx]
    q_hat = models["q_hat_by_band"][band_name]

    lo_log = float(models["q_lo_model"].predict(X_row)[0]) - q_hat
    hi_log = float(models["q_hi_model"].predict(X_row)[0]) + q_hat
    cqr_lo = max(float(np.expm1(lo_log)), 0.0)
    cqr_hi = max(float(np.expm1(hi_log)), 0.0)

    if not (cqr_lo <= predicted_price <= cqr_hi):
        raise AssertionError(
            f"CQR interval ordering violated for this house: cqr_lo={cqr_lo}, "
            f"predicted_price={predicted_price}, cqr_hi={cqr_hi} -- refusing to report "
            f"an interval that doesn't contain its own point estimate."
        )

    return {
        "predicted_price": predicted_price,
        "cqr_lo": cqr_lo,
        "cqr_hi": cqr_hi,
        "price_tercile_band": band_name,
        "relative_width": (cqr_hi - cqr_lo) / predicted_price if predicted_price else float("nan"),
    }
