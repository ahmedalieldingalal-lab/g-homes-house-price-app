"""Phase A, step 5 (Step 6 -- LLM-based Model Interpretation, see
PROJECT_TODO.md "FINAL ACTION PLAN").

TreeSHAP explanations for the shipped point-GBM regressor, expressed as
MULTIPLICATIVE percentage effects in DOLLAR space -- not raw log-space SHAP
values, and not additive dollar contributions (both were considered and
rejected; see Decision 1/2 and the "SHAP percentage math correction" entry
in PROJECT_TODO.md).

Why multiplicative: the regression target is `log1p(price)`, so TreeSHAP's
native output is additive only in LOG space:

    base_value + sum(shap_i) == model.predict(X_row)   (exact, by construction)

Exponentiating that sum to get back to dollars turns the SUM into a
PRODUCT, not a sum of dollar amounts:

    predicted_price + 1 = exp(base_value + sum(shap_i))
                         = exp(base_value) * prod(exp(shap_i))

Defining each feature's percentage effect as `p_i = exp(shap_i) - 1` (i.e.
"this feature multiplies the price by (1 + p_i)") makes that identity
directly, exactly checkable:

    prod(1 + p_i) * exp(base_value) == predicted_price + 1

This holds for the FULL feature set, and -- because it's a product of
independent per-feature factors -- it *still* holds after bucketing the
smaller features into a single "everything else" catch-all whose own
percentage effect is `exp(sum of those features' shap_i) - 1`. That is
exactly Decision 2's "top 6-8 features + one catch-all" display contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def build_explainer(model) -> shap.TreeExplainer:
    """tree_path_dependent needs no background dataset and GUARANTEES
    exact additivity in log-space (base_value + sum(shap) == prediction)
    -- the training-data path counts are already stored in the trees
    themselves, so there's no approximation/background-sample choice to
    make or get wrong."""
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")


def explain_house(X_row: pd.DataFrame, model, explainer: shap.TreeExplainer | None = None,
                   top_k: int = 8) -> dict:
    """`X_row` is a single-row DataFrame in the model's exact feature
    order (e.g. from `step6_inference_contract.build_feature_row`).

    Returns a dict with the raw log-space SHAP decomposition, the
    per-feature multiplicative percentage effects, the top-k + catch-all
    grouping, and an `identity_holds` bool the caller can assert on before
    ever handing anything to the LLM (per Decision 2/step 5's requirement:
    "assert it holds before any LLM call is made").
    """
    if len(X_row) != 1:
        raise ValueError(f"explain_house expects exactly one row, got {len(X_row)}")
    explainer = explainer if explainer is not None else build_explainer(model)

    shap_values = explainer.shap_values(X_row)
    shap_row = np.asarray(shap_values)[0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    pred_log = float(model.predict(X_row)[0])
    additive_log_check = abs((base_value + shap_row.sum()) - pred_log)

    predicted_price = float(np.expm1(pred_log))

    feature_names = list(X_row.columns)
    pct_effects = {name: float(np.exp(sv) - 1.0) for name, sv in zip(feature_names, shap_row)}

    # Full-feature-set multiplicative identity check.
    full_product = np.exp(base_value) * np.prod([1.0 + p for p in pct_effects.values()])
    full_identity_diff = abs(full_product - (predicted_price + 1.0))

    # Top-k by |shap value| (log-space magnitude -- ranks features by their
    # actual effect on the prediction, not by the size of the percentage
    # number, which would double-count the base-rate distortion for large
    # base values).
    order = np.argsort(-np.abs(shap_row))
    top_idx = order[:top_k]
    rest_idx = order[top_k:]

    top_features = [
        {"feature": feature_names[i], "shap_log": float(shap_row[i]), "pct_effect": pct_effects[feature_names[i]],
         "value": _native(X_row.iloc[0][feature_names[i]])}
        for i in top_idx
    ]
    catchall_shap_sum = float(shap_row[rest_idx].sum()) if len(rest_idx) else 0.0
    catchall_pct_effect = float(np.exp(catchall_shap_sum) - 1.0)

    grouped_product = np.exp(base_value) * np.prod([1.0 + f["pct_effect"] for f in top_features]) * (1.0 + catchall_pct_effect)
    grouped_identity_diff = abs(grouped_product - (predicted_price + 1.0))

    # AUDIT FIX 2026-09-09 (SB14, verification pass): `full_identity_diff`/
    # `grouped_identity_diff` are computed in DOLLAR space (they compare
    # against `predicted_price + 1.0`), so their floating-point roundoff
    # scales with the house's price -- a fixed absolute `1e-6` tolerance
    # that's comfortably tight for a typical house shrinks toward zero
    # headroom as price grows (measured: the worst of 40 expensive houses
    # already used 1/20.5 of the available margin at $3.8M; a sufficiently
    # expensive house, or a future model retrain with slightly different
    # rounding behavior, could exceed it and report a FALSE identity
    # failure on a numerically-fine prediction). `additive_log_check` stays
    # on the original fixed `1e-6` -- it's computed in LOG space
    # (`pred_log` is O(13-15) regardless of price), where roundoff does not
    # scale with the house's dollar price, so no relative tolerance is
    # needed there. The two dollar-space checks instead use a tolerance
    # relative to the magnitude actually being compared.
    _dollar_tol = 1e-6 * max(1.0, predicted_price + 1.0)
    identity_holds = (
        additive_log_check < 1e-6
        and full_identity_diff < _dollar_tol
        and grouped_identity_diff < _dollar_tol
    )

    return {
        "base_value_log": base_value,
        "predicted_log": pred_log,
        "predicted_price": predicted_price,
        "shap_values_log": {name: float(sv) for name, sv in zip(feature_names, shap_row)},
        "pct_effects": pct_effects,
        "top_features": top_features,
        "catchall_n_features": int(len(rest_idx)),
        "catchall_pct_effect": catchall_pct_effect,
        "additive_log_check_diff": additive_log_check,
        "full_identity_diff": full_identity_diff,
        "grouped_identity_diff": grouped_identity_diff,
        "identity_holds": identity_holds,
    }


def _native(v):
    """json/LLM-prompt-safe scalar (numpy types aren't JSON-serializable)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
