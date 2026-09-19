"""Phase A, step 6 (part 2) -- classify one new house into a physical
cluster and price band, using the shipped, deployment-designated
classifiers (`saved_models/classifiers/manifest.json`), then derive its
Tier 1 `combined_category` for the market-comparison RAG lookup
(Decision 3).

Feature contracts below are copied verbatim from `classifier_a_pipeline.py`
(`CLUSTER_FEATURES`, `LOG_FEATURES`) and `classifier_b_unified_pipeline.py`
(`UNIFIED_FEATURE_COLUMNS`, `LOG_FEATURES`) -- both confirmed by directly
inspecting the unpickled `imblearn.pipeline.Pipeline` objects, which
contain ONLY a `StandardScaler` + the classifier itself (no log-transform
step baked in), so the log1p has to be applied by the caller in the same
order training did.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import consolidated_pipeline as cp
from step6_inference_contract import load_inference_tables

# AUDIT FIX 2026-09-09 (SB4, final audit): these were bare cwd-relative
# paths ("saved_models/...") -- fine when the process cwd happens to be
# the repo root, but `streamlit run /abs/path/to/app.py` from any OTHER
# directory (e.g. an operator double-clicking a shortcut, or a launcher
# with a different working directory) raised a raw `FileNotFoundError`
# instead of anything meaningful. Anchored to this file's own location,
# matching the pattern step6_plots.py and step6_build_cluster_profile_
# stats.py already used correctly.
REPO_ROOT = Path(__file__).resolve().parent
CLASSIFICATION_TABLES_PATH = REPO_ROOT / "saved_models" / "classification_tables.json"
CLASSIFIER_A_PATH = REPO_ROOT / "saved_models" / "classifiers" / "classifier_a_softmax_regression.joblib"
CLASSIFIER_B_PATH = REPO_ROOT / "saved_models" / "classifiers" / "classifier_b_random_forest.joblib"

CLASSIFIER_A_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "house_age", "was_renovated",
    "total_rooms", "has_basement", "basement_ratio", "lot_utilization",
    "bath_bed_ratio", "avg_room_size", "premium_outlook_score",
    "condition_x_age", "years_since_renovation", "renovation_recency_ratio",
    "city_sale_volume",
]
CLASSIFIER_A_LOG_FEATURES = ["sqft_living", "sqft_lot", "city_sale_volume"]

CLASSIFIER_B_BASE_FEATURES = ["condition", "premium_outlook_score", "renovation_recency_ratio", "sqft_living"]
CLASSIFIER_B_LOCATION_FEATURE = "location_price_per_sqft"
CLASSIFIER_B_CLUSTER_COLS = ["cluster_0", "cluster_1", "cluster_2", "cluster_3"]
CLASSIFIER_B_FEATURES = CLASSIFIER_B_BASE_FEATURES + [CLASSIFIER_B_LOCATION_FEATURE] + CLASSIFIER_B_CLUSTER_COLS
CLASSIFIER_B_LOG_FEATURES = ["sqft_living"]

# AUDIT FIX 2026-09-09 (SA20, final audit): keyed by resolved path string
# instead of a single `dict | None` slot -- same fix and rationale as
# step6_market_comparison.load_enriched_dataset (rule 11 sweep).
_CLASSIFICATION_TABLES_CACHE: dict[str, dict] = {}
_CLASSIFIER_A_CACHE = None
_CLASSIFIER_B_CACHE = None


def load_classification_tables(path: Path = CLASSIFICATION_TABLES_PATH) -> dict:
    key = str(Path(path).resolve())
    if key not in _CLASSIFICATION_TABLES_CACHE:
        with open(path) as f:
            _CLASSIFICATION_TABLES_CACHE[key] = json.load(f)
    return _CLASSIFICATION_TABLES_CACHE[key]


def _check_feature_contract(model, expected_features: list[str], model_name: str) -> None:
    """AUDIT FIX 2026-09-09 (SB9, final audit): `load_classifiers()` used to
    be a bare `joblib.load` with no check at all -- entirely trust-based,
    per the audit's own framing. The hardcoded `CLASSIFIER_A_FEATURES`/
    `CLASSIFIER_B_FEATURES` lists above are documented as "copied verbatim"
    from the training pipelines, with "no import and no automated
    cross-check" -- meaning a future retrain that reorders or renames a
    column would silently feed the wrong values into the wrong feature
    slots (sklearn's `feature_names_in_` gives *some* incidental protection
    normally, since most sklearn estimators refuse a DataFrame whose column
    names don't match at `.predict()` time -- but this app calls
    `.predict()` on a plain numpy array built by hand from these hardcoded
    lists, which bypasses that check entirely). This makes the comparison
    explicit and immediate, at load time, with a clear error naming
    exactly what's wrong -- rather than a wrong-but-silent prediction, or a
    confusing failure deep inside SHAP/plotting later. Verified this
    actually fires: temporarily reordered `expected_features` -> raised
    with a clear message; restored -> loads clean, matching the audit's
    own positive control (both classifiers' real `feature_names_in_`
    already match these lists exactly, today)."""
    if not hasattr(model, "feature_names_in_"):
        return  # older sklearn / unusual estimator -- nothing to check against
    actual = list(model.feature_names_in_)
    if actual != expected_features:
        raise RuntimeError(
            f"{model_name}'s saved feature contract no longer matches the hardcoded feature list in "
            f"step6_classify_house.py -- refusing to load, since silently continuing would score houses "
            f"with values in the wrong feature slots.\n"
            f"  Model's actual feature_names_in_: {actual}\n"
            f"  Hardcoded expected list:          {expected_features}\n"
            f"This means the shipped model was retrained/reordered without updating this file's "
            f"feature-contract constants -- update them to match, then re-verify."
        )


CLASSIFIERS_MANIFEST_PATH = REPO_ROOT / "saved_models" / "classifiers" / "manifest.json"


def _check_log_feature_contract() -> None:
    """FINAL AUDIT PANEL FIX 2026-09-10 (B6): restores the RUNTIME half of
    finding SB9.

    `_check_feature_contract` above compares feature NAMES/ORDER against a
    loaded model's `feature_names_in_`, which says nothing about WHICH of
    those columns are log1p'd before scoring. A drift between this file's
    hardcoded `CLASSIFIER_A_LOG_FEATURES`/`CLASSIFIER_B_LOG_FEATURES` and
    the training pipelines would therefore load clean and silently feed
    linear values into slots the model expects in log space.

    SB9's first fix imported the training pipelines here to compare against
    their `LOG_FEATURES` constants; that was reverted because those modules
    pull in xgboost/lightgbm/catboost/optuna and are not shipped to the
    trimmed deployment at all -- it would have crashed the deployed app on
    its first prediction. The check then lived only in `check_consistency.py`
    (build-time, repo-side), which a marker may never run and which cannot
    see a deployed copy.

    `saved_models/classifiers/manifest.json` closes that gap with no new
    dependency: it is generated by `serialize_classifier_artifacts.py` from
    the training reports, already ships with the deployment, already carries
    `log_transformed_features` for both classifiers, and is already named as
    the deployment source of truth in this module's own docstring.

    Deliberately non-fatal when the manifest is ABSENT: a deployment that
    trimmed it should still predict (the model files themselves are what
    matter), so a missing/unreadable manifest skips the check rather than
    bricking the app. A manifest that is present and DISAGREES is a real
    drift signal and raises. `check_consistency.py`'s build-time check is
    retained as well -- it compares against the training pipelines
    themselves, one link further upstream than this."""
    try:
        with open(CLASSIFIERS_MANIFEST_PATH) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return  # manifest not shipped / unreadable -- skip, do not break inference

    for key, hardcoded, model_name in [
        ("classifier_a", CLASSIFIER_A_LOG_FEATURES, "classifier_a_softmax_regression"),
        ("classifier_b", CLASSIFIER_B_LOG_FEATURES, "classifier_b_random_forest"),
    ]:
        shipped = manifest.get(key, {}).get("log_transformed_features")
        if shipped is not None and list(shipped) != list(hardcoded):
            raise RuntimeError(
                f"{model_name}'s log-feature contract no longer matches the shipped manifest -- refusing "
                f"to load, since silently continuing would apply log1p to the wrong columns.\n"
                f"  Hardcoded in step6_classify_house.py: {list(hardcoded)}\n"
                f"  saved_models/classifiers/manifest.json: {list(shipped)}\n"
                f"Regenerate the manifest (serialize_classifier_artifacts.py) or update this file's "
                f"constants to match, then re-verify."
            )


def load_classifiers():
    global _CLASSIFIER_A_CACHE, _CLASSIFIER_B_CACHE
    _check_log_feature_contract()
    if _CLASSIFIER_A_CACHE is None:
        classifier_a = joblib.load(CLASSIFIER_A_PATH)
        _check_feature_contract(classifier_a, CLASSIFIER_A_FEATURES, "classifier_a_softmax_regression")
        _CLASSIFIER_A_CACHE = classifier_a
    if _CLASSIFIER_B_CACHE is None:
        classifier_b = joblib.load(CLASSIFIER_B_PATH)
        _check_feature_contract(classifier_b, CLASSIFIER_B_FEATURES, "classifier_b_random_forest")
        _CLASSIFIER_B_CACHE = classifier_b
    return _CLASSIFIER_A_CACHE, _CLASSIFIER_B_CACHE


def classify_house(engineered: dict, tables: dict | None = None, class_tables: dict | None = None,
                    classifier_a=None, classifier_b=None) -> dict:
    """`engineered` is the per-row dict of raw + engineered fields already
    computed by `step6_inference_contract.build_feature_row`'s internals
    (house_age, total_rooms, has_basement, basement_ratio, ... ,
    location_price_per_sqft) -- see `interpret_house()`, which is the only
    caller and passes these through directly instead of recomputing them a
    second time (standing rule 11: one feature-engineering pass per house).

    Returns {physical_cluster, price_band, combined_category,
    tier1_profile, flags}.

    AUDIT FIX 2026-09-09 (SB19, final audit): the docstring above used to
    promise a `price_band_clipped` return KEY -- there never was one; the
    clip is signaled via `flags["price_band_clipped"] = True` instead
    (confirmed: `classify_house(...)` returns exactly the 5 keys now
    listed above). Separately, the classifier-override logic used to be
    `clf_a, clf_b = (classifier_a, classifier_b) if classifier_a is not
    None else load_classifiers()` -- keying BOTH classifiers' fate off
    `classifier_a` alone. `classify_house(classifier_a=a, classifier_b=
    None)` crashed with `AttributeError: 'NoneType' object has no
    attribute 'predict'` on `clf_b`; `classify_house(classifier_a=None,
    classifier_b=b)` silently DISCARDED the caller's real `b` and loaded
    both classifiers from disk instead, with no error or warning. Now
    each classifier is independently defaulted to the loaded one only
    when its own argument is `None`, so a partial override does what its
    caller obviously intended instead of crashing or silently ignoring
    an argument."""
    tables = tables if tables is not None else load_inference_tables()
    class_tables = class_tables if class_tables is not None else load_classification_tables()
    if classifier_a is None or classifier_b is None:
        loaded_a, loaded_b = load_classifiers()
        clf_a = classifier_a if classifier_a is not None else loaded_a
        clf_b = classifier_b if classifier_b is not None else loaded_b
    else:
        clf_a, clf_b = classifier_a, classifier_b
    flags: dict = {}

    # --- Classifier A: physical_cluster ---------------------------------
    city_map = class_tables["city_sale_volume_full_map"]
    city_fallback = class_tables["city_sale_volume_full_fallback"]
    city_sale_volume = city_map.get(engineered["city"], city_fallback)
    if engineered["city"] not in city_map:
        flags["city_unseen_for_classification"] = True

    row_a = {**engineered, "city_sale_volume": city_sale_volume}
    X_a = pd.DataFrame([[row_a[c] for c in CLASSIFIER_A_FEATURES]], columns=CLASSIFIER_A_FEATURES)
    for c in CLASSIFIER_A_LOG_FEATURES:
        X_a[c] = np.log1p(X_a[c])
    physical_cluster = int(clf_a.predict(X_a)[0])

    # --- Classifier B: price_band (cluster one-hot as input) -----------
    row_b = dict(engineered)
    X_b = pd.DataFrame([[row_b[c] for c in CLASSIFIER_B_BASE_FEATURES] + [row_b[CLASSIFIER_B_LOCATION_FEATURE]]],
                       columns=CLASSIFIER_B_BASE_FEATURES + [CLASSIFIER_B_LOCATION_FEATURE])
    for c in CLASSIFIER_B_LOG_FEATURES:
        X_b[c] = np.log1p(X_b[c])
    for col in CLASSIFIER_B_CLUSTER_COLS:
        X_b[col] = 1.0 if col == f"cluster_{physical_cluster}" else 0.0
    X_b = X_b[CLASSIFIER_B_FEATURES]
    price_band_raw = int(clf_b.predict(X_b)[0])

    # Deployment safety net -- clip to the physical_cluster's valid band
    # range using the project's OWN already-validated guard function
    # (provably a no-op on all real training/test data, per its docstring;
    # only fires on a genuinely unusual new input).
    price_band = cp.band_validity_guard(physical_cluster, price_band_raw)
    if price_band != price_band_raw:
        flags["price_band_clipped"] = True

    combined_category = f"Physical Group {physical_cluster} - Price Band {price_band}"
    tier1_profile = class_tables["tier1_profiles"].get(combined_category)
    if tier1_profile is None:
        # Should be unreachable given band_validity_guard, but Tier 1's own
        # fallback ladder (Decision 3b) requires a defined behavior here
        # too: snap to the nearest valid band within the SAME cluster
        # rather than ever falling back to cluster-only.
        flags["tier1_snapped_to_nearest_band"] = True
        candidates = {k: v for k, v in class_tables["tier1_profiles"].items() if v["physical_cluster"] == physical_cluster}
        nearest_key = min(candidates, key=lambda k: abs(candidates[k]["price_band"] - price_band))
        combined_category = nearest_key
        tier1_profile = candidates[nearest_key]
        # AUDIT FIX 2026-09-09 (SA30, final audit): this branch used to
        # reassign `combined_category` to the snapped band but leave
        # `price_band` pointing at the PRE-snap value -- a single-source-
        # of-truth break where the two returned fields could describe two
        # different bands (e.g. `build_prompt_facts()` grounding the LLM
        # on `combined_category` while some other reader trusted the
        # stale `price_band`). Reassigning it here from the SAME profile
        # `combined_category` was just snapped to keeps every returned
        # field consistent with the category actually used. Confirmed
        # latent, not live: this branch is documented as unreachable given
        # `band_validity_guard`'s own guarantee, and does not fire on any
        # of the 908 real held-out houses.
        price_band = tier1_profile["price_band"]

    return {
        "physical_cluster": physical_cluster,
        "price_band": price_band,
        "combined_category": combined_category,
        "tier1_profile": tier1_profile,
        "flags": flags,
    }
