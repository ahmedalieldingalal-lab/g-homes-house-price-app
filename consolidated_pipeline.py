#!/usr/bin/env python3
"""
Consolidated House Price Pipeline -- Steps 1, 2 and 3 of the project, ONE file
================================================================================

WHAT THIS FILE IS, IN PLAIN ENGLISH
------------------------------------
This is the "clean base" for the project. Before this file existed, the
work lived in three separate scripts that were built one after another as
new questions came up:

  1. full_project_pipeline.py           -- cleaning, price prediction,
                                            business insights, and a big
                                            "try everything" clustering grid
                                            (36 combinations) kept as an
                                            evidence trail.
  2. best_clustering_pipeline.py        -- Model A: the ONE clustering
                                            recipe out of those 36 that
                                            actually won, grouping houses by
                                            physical profile (size, age,
                                            renovation status).
  3. nested_price_segmentation_pipeline.py -- Model B: inside each of Model
                                            A's 4 physical groups, a second,
                                            separate clustering step that
                                            sorts houses into price bands.

Those three files shared a lot of logic (the same cleaning rules, the same
engineered features), just copy-pasted three times. That's a real risk for
a graduation project someone else has to review: three copies means three
places a fix could be applied to only one of them by mistake. This file
merges them into ONE script with each shared step written ONCE, so:

  - Cleaning and feature engineering happen in exactly one place and are
    reused by every downstream step (price prediction, the exploration
    grid, Model A, and Model B).
  - Every bug fix that was found and fixed during the project (see
    PROJECT_TODO.md for the list) is already folded in here -- this file
    is not "fixed later", it already IS the fixed version.
  - Nothing about the actual math changed in this merge. Every function
    below is the same validated code that was already tested and reported
    on, just organized in one place instead of three. Re-running this file
    is expected to reproduce every previously-reported number exactly.

WHAT THIS FILE COVERS (PDF pipeline steps)
--------------------------------------------
  Step 1: Data Cleaning & Understanding      -> clean_data()
  Step 2: Exploratory Analysis & Business
          Insights                            -> business_insights(), plots
  Step 3: Unsupervised Learning (Clustering),
          with meaningful cluster categories  -> Model A + Model B, named
                                                 clusters, business-meaning
                                                 write-ups

Step 4 onward (classification, deployment, etc.) are NOT in this file --
those come next, built ON TOP of this clean base.

Usage:
    python consolidated_pipeline.py --input data.csv --output-dir consolidated_output

    # Faster run while testing (smaller cluster search grid for Part B):
    python consolidated_pipeline.py --input data.csv --quick

Run `python consolidated_pipeline.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
# BUG FIX 2026-09-03 (ground-up audit P2 item 27): a blanket ignore() also
# swallows sklearn's InconsistentVersionWarning -- the one warning
# requirements.txt's own scikit-learn==1.8.0 pin explicitly exists to
# protect against missing. Registered AFTER the blanket ignore so it takes
# precedence: filterwarnings() prepends to the filter list, so the
# most-recently-added filter is checked first.
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("always", category=InconsistentVersionWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

logger = logging.getLogger("consolidated_pipeline")


# =============================================================================
# PART 0 -- shared setup (CLI, logging, loading)
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data.csv", help="Path to the raw house-sale CSV.")
    p.add_argument("--output-dir", default="consolidated_output", help="Folder for all reports/plots/CSVs.")
    p.add_argument("--test-size", type=float, default=0.2, help="Regression test split fraction (Part A).")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--n-clusters", type=int, default=4,
        help="Number of KMeans clusters for Model A / physical clustering (default 4, confirmed best during testing).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smaller cluster-search grid (k up to 6 instead of 10, fewer DBSCAN candidates) "
        "for the Part B exploration grid, for a faster run while testing. Numbers may shift "
        "slightly from the full grid; Model A / Model B are unaffected by --quick.",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_data(input_path: str) -> pd.DataFrame:
    """Question this answers: none directly -- just getting the raw file in."""
    csv_path = Path(input_path)
    if not csv_path.exists():
        logger.error("Input file not found: %s", csv_path)
        sys.exit(1)
    logger.info("Loaded %s", csv_path)
    return pd.read_csv(csv_path)


# =============================================================================
# PART 1 -- Cleaning (Step 1). Shared by EVERYTHING downstream: price
# prediction, the clustering exploration grid, Model A, and Model B all call
# this ONE function instead of each having their own copy.
# =============================================================================
def clean_data(df: pd.DataFrame, cap_outliers: bool) -> tuple[pd.DataFrame, dict]:
    """Fix types, fill missing values, drop duplicates/invalid rows, and
    (optionally) cap extreme price/sqft_living/sqft_lot values using the
    1.5*IQR rule.

    Question this answers: "how many records are outliers?" and "is capping
    the outliers going to affect the regression / clustering models?" --
    this function is what actually performs the capping being discussed, and
    reports exactly how many rows it touched.

    **Audit finding R13 (documented, not changed): this capping is a
    PRE-SPLIT, whole-dataset statistic, and `price` -- the regression
    target -- is one of the capped columns.** That is intentional and
    correct for what this function is actually used for in this project:
    Part A/B's EDA and the flat baseline-vs-enhanced regression comparison,
    both of which are run on the full dataset before any train/test split
    exists, purely to characterize and compare capped vs. uncapped
    distributions -- there is no split to leak across at that stage. It is
    NOT safe to call this function on data that has already been split, or
    to reuse its capping for a model that will be evaluated on a held-out
    test set: doing so would let test-set values influence the train-set
    capping bounds (and, for the target, cap the very thing being
    predicted). For that use case -- Step 5's train/test-based models --
    use `fit_feature_caps()` instead, which computes bounds from TRAIN only
    and raises if asked to cap anything in `FORBIDDEN_LEAKAGE_COLUMNS`
    (including `price`), and apply those bounds to train and test
    separately.
    """
    df = df.copy()
    stats = {"rows_in": len(df)}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # BUG FIX (found in code review): an unparseable date becomes NaT, which
    # is neither a numeric nor an object/str dtype, so it was silently
    # skipped by the fillna loop below -- unlike every other column. A NaT
    # here would propagate into sale_year -> house_age -> renovation-recency
    # features as NaN, and could even crash model training downstream. Not
    # triggered by the current data.csv (no bad dates in it), but treated
    # the same way as other invalid rows: dropped and reported, not guessed.
    n_bad_dates = int(df["date"].isna().sum())
    if n_bad_dates:
        df = df[df["date"].notna()].copy()
    stats["unparseable_dates_dropped"] = n_bad_dates

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            mode_val = df[c].mode()
            df[c] = df[c].fillna(mode_val[0] if not mode_val.empty else "Unknown")

    before = len(df)
    df = df.drop_duplicates()
    stats["duplicates_dropped"] = before - len(df)

    before = len(df)
    df = df[df["price"] > 0]
    df = df[(df["bedrooms"] > 0) & (df["bathrooms"] > 0)]
    stats["invalid_rows_dropped"] = before - len(df)

    def cap_iqr(series: pd.Series, k: float = 1.5) -> tuple[pd.Series, int]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        n_touched = int(((series < lower) | (series > upper)).sum())
        return series.clip(lower=lower, upper=upper), n_touched

    stats["outliers_capped"] = {}
    stats["capping_applied"] = cap_outliers
    for col in ["price", "sqft_living", "sqft_lot"]:
        capped_series, n_touched = cap_iqr(df[col])
        stats["outliers_capped"][col] = n_touched  # always report, even if not applied
        if cap_outliers:
            # BUG FIX 2026-08-27 (audit finding H1): sqft_living has two
            # COMPONENTS -- sqft_above + sqft_basement -- that must keep summing
            # to it. Clipping sqft_living alone while leaving the components
            # untouched made the ratio features in add_new_features() physically
            # impossible: 60 rows ended up with above_ratio > 1 (max 2.1707,
            # i.e. a house whose above-ground area was 217% of its whole living
            # area). Verified the uncapped frame is correctly bounded at exactly
            # 1.0, so this was purely an artifact of clipping the total but not
            # the parts. Fixed by shrinking both components by the SAME factor
            # the total was shrunk by, which preserves
            # sqft_above + sqft_basement == sqft_living and keeps every derived
            # ratio in its valid range.
            if col == "sqft_living":
                scale = (capped_series / df[col]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
                df["sqft_above"] = df["sqft_above"] * scale
                df["sqft_basement"] = df["sqft_basement"] * scale
            df[col] = capped_series
        # if not capping, we still log1p-transform skew later at the modeling
        # step, per the "just log-transform them" instruction from testing.

    stats["rows_out"] = len(df)
    logger.info(
        "Cleaned: %d -> %d rows (dropped %d duplicates, %d invalid; outlier rows found: price=%d, "
        "sqft_living=%d, sqft_lot=%d; capping %s)",
        stats["rows_in"], stats["rows_out"], stats["duplicates_dropped"], stats["invalid_rows_dropped"],
        stats["outliers_capped"]["price"], stats["outliers_capped"]["sqft_living"], stats["outliers_capped"]["sqft_lot"],
        "APPLIED" if cap_outliers else "SKIPPED (log1p only, used later)",
    )
    return df, stats


# =============================================================================
# PART 2 -- Feature engineering (Step 1/2). Also shared by everything
# downstream. best_clustering_pipeline.py's old `engineer_features` computed
# a subset of these SAME columns with byte-identical formulas -- confirmed
# during the consolidation review -- so Model A and Model B now simply reuse
# this one engineering path instead of a second, separate copy.
# =============================================================================
def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's original engineered features.
    Question this answers: baseline of "any recommended extra feature engineering?"
    """
    df = df.copy()
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["house_age"] = (df["sale_year"] - df["yr_built"]).clip(lower=0)
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    df["zipcode"] = df["statezip"].astype(str).str.extract(r"(\d{5})")[0]
    # BUG FIX 2026-09-03 (dual-audit fix pass, edge case flagged in the ground-up
    # audit's "Critical Missing Parts & Edge Cases" table): the fixed bin list
    # [0, 10, 25, 50, 100, max_age+1] is non-monotonic -- and pd.cut raises
    # `ValueError: bins must increase monotonically` -- whenever max_age <= 100,
    # since the last edge (max_age+1) then falls at or below an earlier fixed
    # edge (100). Unreachable on the current data (max house_age is 114) but
    # was NOT in PROJECT_TODO.md's dormant-edge-case list, unlike its two
    # siblings, and a newer-construction dataset (or this same dataset
    # filtered to newer homes) would crash cleaning with an opaque pandas
    # error instead of a useful message. Only include a fixed interior edge if
    # it is actually below the observed max, so the bins are always strictly
    # increasing; the top bin's label always describes "at or above the
    # highest fixed edge actually used" rather than assuming 100y+ is always
    # reachable.
    max_age = df["house_age"].max()
    fixed_edges = [10, 25, 50, 100]
    fixed_labels = ["0-10y", "11-25y", "26-50y", "51-100y"]
    kept = [(e, lbl) for e, lbl in zip(fixed_edges, fixed_labels) if e < max_age + 1]
    age_bins = [0] + [e for e, _ in kept] + [max_age + 1]
    if kept and kept[-1][0] == 100:
        # Normal case (today's data): all 4 fixed edges fit below max_age+1,
        # so the top bin is genuinely "100y+" same as before this fix.
        age_labels = [lbl for _, lbl in kept] + ["100y+"]
    else:
        # max_age <= 100 (or <= any earlier fixed edge): the top bin's true
        # upper bound is max_age itself, not a clean round number -- label it
        # accordingly rather than falsely claiming "100y+".
        last_edge = kept[-1][0] if kept else 0
        age_labels = [lbl for _, lbl in kept] + [f"{last_edge}y+"]
    df["age_bucket"] = pd.cut(df["house_age"], bins=age_bins, labels=age_labels, include_lowest=True)
    return df


NEW_NUMERIC_FEATURES = [
    "years_since_renovation", "renovation_recency_ratio", "has_basement", "basement_ratio",
    "above_ratio", "lot_utilization", "bath_bed_ratio", "avg_room_size", "premium_outlook_score",
    "condition_x_age", "sale_month_sin", "sale_month_cos", "city_sale_volume", "zip_sale_volume",
    "log_sqft_lot",
]


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """The new engineered features recommended during this project, PLUS a
    data-quality fix that was discovered along the way.

    Question this answers: "any recommended extra feature engineering?"

    THE FIX: 193 rows (4.2% of the data) have yr_renovated < yr_built, which
    is physically impossible (a renovation can't happen before the house was
    built -- almost certainly a data-entry error, e.g. a swapped digit).
    Left unfixed, these rows produce extreme, wrong values for
    years_since_renovation / renovation_recency_ratio (ratios as high as 91),
    which was later found to badly distort RobustScaler-based clustering.
    Here, those 193 rows are treated as "not renovated" instead of trusting
    the bad renovation year.
    """
    df = df.copy()

    df["years_since_renovation"] = np.where(
        df["was_renovated"] == 1, df["sale_year"] - df["yr_renovated"], np.nan
    )
    df["years_since_renovation"] = df["years_since_renovation"].clip(lower=0)
    df["renovation_recency_ratio"] = np.where(
        (df["was_renovated"] == 1) & (df["house_age"] > 0),
        df["years_since_renovation"] / df["house_age"],
        np.nan,
    )
    df["years_since_renovation"] = df["years_since_renovation"].fillna(-1)
    df["renovation_recency_ratio"] = df["renovation_recency_ratio"].fillna(-1)

    # --- data-quality fix: impossible renovation dates ---
    bad_renovation = (df["was_renovated"] == 1) & (df["yr_renovated"] < df["yr_built"])
    n_bad = int(bad_renovation.sum())
    if n_bad:
        df.loc[bad_renovation, "was_renovated"] = 0
        df.loc[bad_renovation, "years_since_renovation"] = -1
        df.loc[bad_renovation, "renovation_recency_ratio"] = -1
        logger.info(
            "Data-quality fix: %d rows (%.1f%%) had yr_renovated < yr_built (impossible) -- "
            "treated as not-renovated instead of using the bad renovation year.",
            n_bad, 100 * n_bad / len(df),
        )

    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"]
    df["above_ratio"] = df["sqft_above"] / df["sqft_living"]
    df["lot_utilization"] = df["sqft_living"] / df["sqft_lot"]
    df["bath_bed_ratio"] = df["bathrooms"] / df["bedrooms"]
    df["avg_room_size"] = df["sqft_living"] / df["total_rooms"]
    df["premium_outlook_score"] = np.maximum(df["view"], df["waterfront"] * 3)
    df["condition_x_age"] = df["condition"] * df["house_age"]
    df["sale_month_sin"] = np.sin(2 * np.pi * df["sale_month"].fillna(0) / 12)
    df["sale_month_cos"] = np.cos(2 * np.pi * df["sale_month"].fillna(0) / 12)
    df["city_sale_volume"] = df.groupby("city")["price"].transform("count")
    df["zip_sale_volume"] = df.groupby("zipcode")["price"].transform("count")
    df["log_sqft_lot"] = np.log1p(df["sqft_lot"])
    return df


def kfold_target_encode(train_df, test_df, group_col, target_col, n_splits=5, random_state=42, smoothing=10.0):
    """Mean-encode a category (e.g. city) using only OTHER folds of the
    training data, so the model never sees its own target leak into the
    encoding. Rare categories are pulled toward the overall average.

    Question this answers: this is the safe way to turn "city" / "zipcode"
    into a number without the target-leakage problem raised about
    price_per_sqft -- same leakage risk, different feature, handled properly.
    """
    global_mean = train_df[target_col].mean()
    train_encoded = np.full(len(train_df), global_mean, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_reset = train_df.reset_index(drop=True)
    for fit_idx, hold_idx in kf.split(train_reset):
        fit_part = train_reset.iloc[fit_idx]
        stats = fit_part.groupby(group_col)[target_col].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        mapped = train_reset.iloc[hold_idx][group_col].map(smoothed).fillna(global_mean)
        train_encoded[hold_idx] = mapped.values
    full_stats = train_df.groupby(group_col)[target_col].agg(["mean", "count"])
    full_smoothed = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
    test_encoded = test_df[group_col].map(full_smoothed).fillna(global_mean).values
    return train_encoded, test_encoded


def leak_safe_count_encode(train_df, test_df, group_col):
    """Leak-safe version of a group-size feature (e.g. "how many sales does
    this city/zipcode have"), same spirit as kfold_target_encode above but
    for a raw count instead of a target mean.

    Bug fix 2026-08-27 (pre-Step-5 audit, user-confirmed): city_sale_volume
    and zip_sale_volume were originally computed once, in add_new_features(),
    with a single groupby(...).transform("count") over the WHOLE dataset --
    before any train/test split existed. That meant a training row's feature
    value was quietly influenced by how many TEST-set houses happened to
    share its city/zipcode -- information about the test set that a model
    should never get to see during training. Verified real: 99.6% of
    training rows' counts change if test rows are excluded from the count.

    This function computes the count from TRAIN rows only, and maps that
    same train-derived count onto the test rows -- mirroring exactly how
    kfold_target_encode above already keeps the test set untouched.

    Note: unlike kfold_target_encode, this deliberately does NOT k-fold
    cross-fit the training portion. Cross-fitting exists there to stop a
    training row's own PRICE (its target) from leaking into its own encoded
    feature. A raw count of "how many houses are in this city" is not
    derived from any row's target value, so a training row seeing its own
    city's train-only count is not target leakage -- it's the same kind of
    information the "city" categorical feature itself already carries.

    Scope note: this fix is applied here, in the supervised price-modeling
    step, and will be reused the same way for the Step 5 regressor. It is
    deliberately NOT applied to Model A's clustering (city_sale_volume is
    also one of Model A's 21 clustering features) -- clustering has no
    train/test split to leak across, so "leak-safe" isn't a meaningful
    concept there. This was checked empirically, not assumed: reclustering
    with a leak-safe version of city_sale_volume produced an Adjusted Rand
    Index of 1.0 against the current production physical_cluster labels --
    identical cluster membership and sizes -- so leaving Model A's own
    feature untouched costs nothing and avoids re-running/re-verifying
    Classifier A and B for zero benefit.
    """
    train_counts = train_df.groupby(group_col).size()
    train_encoded = train_df[group_col].map(train_counts).values.astype(float)
    # BUG FIX 2026-08-27 (audit finding M3): the unseen-category fallback used
    # to be train_counts.mean(), which is directionally BACKWARDS. A category
    # that never appears in the training data is, by definition, among the
    # rarest we could encounter -- but the mean group size on this data is 86.6
    # (median 27.5), so the two test-set cities absent from train were being
    # told they were high-volume markets. Using the smallest count actually
    # observed in training keeps the value inside the range the model was
    # trained on while carrying the correct "this is rare" signal.
    test_encoded = test_df[group_col].map(train_counts).fillna(train_counts.min()).values.astype(float)
    return train_encoded, test_encoded


# =============================================================================
# LEAK-SAFE TOOLKIT FOR STEP 5 (added 2026-08-27 after the pre-regression audit)
# -----------------------------------------------------------------------------
# The three helpers below exist so the regression step cannot repeat mistakes
# the audit found upstream. Each one replaces a specific defect:
#   * fit_feature_caps / apply_feature_caps  -> replaces the global, pre-split,
#     target-rewriting capping in clean_data() (audit finding C1, secondary).
#   * group_train_test_split                 -> replaces a plain random split,
#     which put repeat sales of the SAME address on both sides (finding H3).
#   * assert_no_leakage                      -> replaces the tautological
#     `assert not (set(CONSTANT) & set(CONSTANT))` guard in both classifiers,
#     which compared two module-level literals and could never fire (finding
#     in audit section 5).
# =============================================================================

# One shared definition, so Step 5 and both classifiers can stop each keeping
# their own copy. Every column here is derived from `price` in some way.
FORBIDDEN_LEAKAGE_COLUMNS = frozenset({
    "price", "price_per_sqft", "city_price_index", "zip_price_index",
    "is_outlier_price", "is_outlier", "price_band", "combined_category",
    "friendly_category",
})

# Columns that are NOT target-derived, but ARE pre-split whole-dataset
# statistics (computed over all 4,549 rows before any train/test split
# exists). Reading these straight from enriched_dataset.csv into a
# split-based (supervised) model leaks test-set information into train, even
# though the columns don't reveal price directly (audit finding R4).
# Legitimate for clustering, which has no split concept -- see
# `assert_no_leakage`'s `presplit_columns_ok` parameter.
MUST_RECOMPUTE_IN_SPLIT_COLUMNS = frozenset({
    "city_sale_volume", "zip_sale_volume",
    "is_outlier_sqft_living", "is_outlier_sqft_lot",
})


def fit_feature_caps(train_df: pd.DataFrame, cols, k: float = 1.5) -> dict:
    """Learn 1.5*IQR clip bounds from the TRAINING ROWS ONLY.

    Why this exists (audit finding C1, secondary defect): clean_data()'s
    cap_iqr computes its bounds over the whole dataset before any split, which
    means test-set rows help decide the clip fence -- and because `price` is one
    of the capped columns, it also rewrites test-set TARGET values. That is a
    textbook pre-split leak.

    Pair with apply_feature_caps(). Deliberately takes a `cols` list so the
    caller must name what it caps: the target must NEVER be passed in here.

    Enforced, not just documented (audit finding R12, fixed 2026-08-28): the
    docstring above used to be the only thing stopping a caller from passing
    the target -- `fit_feature_caps(train, ["price"])` returned clip bounds
    without complaint. It now raises instead.
    """
    forbidden = FORBIDDEN_LEAKAGE_COLUMNS.intersection(cols)
    if forbidden:
        raise ValueError(
            f"fit_feature_caps: {sorted(forbidden)} must never be capped here -- "
            f"capping the target (or another forbidden column) with bounds fit on "
            f"train is still capping it, which rewrites values downstream code may "
            f"treat as ground truth. Cap engineered predictors only."
        )
    bounds = {}
    for col in cols:
        q1, q3 = train_df[col].quantile(0.25), train_df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - k * iqr, q3 + k * iqr)
    return bounds


def apply_feature_caps(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Apply bounds learned by fit_feature_caps() to any frame (train or test).

    Note the asymmetry that makes this leak-safe: the bounds come from train
    only, but they are applied to both sides -- exactly how a fitted scaler
    behaves. Returns a copy; never mutates the caller's frame.
    """
    out = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def group_train_test_split(df: pd.DataFrame, group_col=("street", "statezip"),
                           test_size: float = 0.2, random_state: int = 42):
    """Train/test split that keeps every sale of the SAME property on one side.

    Why this exists (audit finding H3): 72 street addresses appear more than
    once in this dataset, covering 147 rows (3.2%). Under a plain random split,
    sale #1 of a house lands in train and sale #2 in test, so the model is
    scored on a property whose price it has effectively already been shown.

    `group_col` defaults to a COMPOUND key (street, statezip) rather than
    `street` alone (audit finding R16, fixed 2026-08-28): exactly one street
    name spans two different cities in this dataset ('Burke-Gilman Trail' --
    a $463,000 Lake Forest Park property and a $1,675,000 Seattle property),
    so grouping on `street` alone over-groups two genuinely different
    properties into one. Pass a single column name (e.g. `"street"`) to
    restore the old single-key behavior if ever needed.

    Returns (train_df, test_df) with the original index preserved, so callers
    can cross-reference rows against other splits the way
    e2e_validate_unified.py does.
    """
    if isinstance(group_col, str):
        groups = df[group_col]
    else:
        groups = df[list(group_col)].astype(str).agg("||".join, axis=1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


def assert_no_leakage(X: pd.DataFrame, context: str = "", allowed_derivations: dict | None = None,
                       source_frame: pd.DataFrame | None = None, corr_threshold: float = 0.999,
                       presplit_columns_ok: bool = False) -> None:
    """Runtime guard: fail loudly if a price-derived column reached the model.

    Why this exists (audit section 5): both classifiers used to guard themselves
    with `assert not (set(BASE_FEATURES) & FORBIDDEN_LEAKAGE_COLUMNS)`, where
    BOTH operands are module-level constants -- a static tautology that can
    never fire at runtime. This checks the real thing: the columns actually
    handed to .fit().

    THREE INDEPENDENT CHECKS. No hatch can excuse check 1; check 2 requires an
    explicit opt-in; check 3 requires real data to prove the exception is safe.

    1. NAME CHECK (always enforced, unconditionally -- audit finding R2, fixed
       2026-08-28). If any column in X is spelled exactly like a forbidden
       column, this raises. `allowed_derivations` CANNOT suppress this. The
       previous version could be defeated with:
           assert_no_leakage(X[["bedrooms", "price_per_sqft"]],
                             allowed_derivations={"price_per_sqft": "..."})  # used to PASS
       A raw forbidden column reaching .fit() is never legitimate, whatever
       reason is offered -- if it's genuinely safe, give it a different name
       (that's what check 3 is for).

    2. PRE-SPLIT WHOLE-DATASET STATISTIC CHECK (audit finding R4). Columns in
       MUST_RECOMPUTE_IN_SPLIT_COLUMNS (city_sale_volume, zip_sale_volume,
       is_outlier_sqft_living, is_outlier_sqft_lot) are computed over ALL
       rows before any split exists -- not target-derived, but still a leak
       into a split-based model. Raises unless `presplit_columns_ok=True`
       (legitimate for clustering, which has no split concept).

    3. VALUE CHECK -- the actual escape hatch, and the only one with teeth.
       Some features ARE built from a forbidden column but are safe because
       they are cross-fitted within the training folds -- Classifier B's
       `location_price_per_sqft` (derived from `price_per_sqft`) is the
       standing example. Naming it in `allowed_derivations` used to be pure
       documentation with no enforcement (a column that merely has a
       different NAME already passes check 1 for free). Pass `source_frame`
       (the pre-feature-selection frame that still has the forbidden columns
       in it) and this now verifies the claim: if the named column correlates
       >= `corr_threshold` with any forbidden column present in
       `source_frame`, it raises anyway -- it looks like a leak wearing a
       different name, not a genuinely safe derivation.
           assert_no_leakage(X_train, "Step 5 regressor", source_frame=df_train,
                             allowed_derivations={"location_price_per_sqft":
                                 "k-fold cross-fitted within train; see build_location_feature_kfold"})
    """
    if not hasattr(X, "columns"):
        raise TypeError(
            f"assert_no_leakage expects a DataFrame (needs .columns) so it can name the "
            f"offending column; got {type(X).__name__}. Wrap it before checking."
        )

    hits = sorted(set(X.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if hits:
        raise AssertionError(
            f"Leakage guard tripped{f' in {context}' if context else ''}: "
            f"price-derived column(s) {hits} are in the assembled feature matrix. "
            f"A raw forbidden column can never be excused via allowed_derivations -- "
            f"it must be dropped, or recomputed under a DIFFERENT name using a leak-safe "
            f"pattern (see check 3 in this function's docstring)."
        )

    if not presplit_columns_ok:
        presplit_hits = sorted(set(X.columns) & MUST_RECOMPUTE_IN_SPLIT_COLUMNS)
        if presplit_hits:
            raise AssertionError(
                f"Leakage guard tripped{f' in {context}' if context else ''}: "
                f"column(s) {presplit_hits} are pre-split, whole-dataset statistics. Used "
                f"as-is in a split-based model they leak test-set information into train. "
                f"Recompute them inside your own split (e.g. leak_safe_count_encode(), "
                f"fit-on-train-only), or pass presplit_columns_ok=True with a written "
                f"justification if this call genuinely has no split (e.g. clustering)."
            )

    allowed = allowed_derivations or {}
    for col, why in allowed.items():
        if col not in X.columns:
            continue
        if source_frame is None:
            logger.info(
                "Leakage guard: '%s' allowed by explicit exception -- %s "
                "(no source_frame given, so this claim was NOT value-checked -- pass "
                "source_frame to actually verify it)", col, why,
            )

    # BUG FIX 2026-09-03 (dual-audit fix pass, finding D3 / B11): check 3
    # used to iterate ONLY `allowed.items()` -- i.e. only columns the CALLER
    # already named as a documented exception. A column that is an exact
    # (or near-exact) copy of a forbidden column under an UNDECLARED name
    # was checked by nothing: check 1 is name-only, check 2 covers a fixed
    # 4-name list, and check 3 never saw it because it was never in
    # `allowed`. Both independent Sep-3 audits proved this live: an
    # undeclared column built as a verbatim copy of a forbidden column
    # passed all three checks with source_frame supplied. Fixed by scanning
    # EVERY column of X (not just the declared ones) whenever source_frame
    # is given -- a documented column that clears the correlation threshold
    # logs the same "allowed by explicit exception" message as before; an
    # UNDECLARED column that clears it raises with a message calling out
    # that it was never declared, since that is the realistic failure mode
    # (an engineer adding a leaky feature is exactly the person who won't
    # think to declare it). Verified before and after by tests/…:
    # test_renamed_high_correlation_leak covers the declared case;
    # test_undeclared_renamed_leak (added this pass) covers the new one.
    if source_frame is not None:
        # Same positional-alignment reasoning as before this fix (see the
        # comment this block replaces, preserved in spirit): this checker
        # runs right after a feature matrix is assembled FROM source_frame
        # (same rows, same order), so X's index is a fresh 0..n-1 range that
        # almost never overlaps source_frame's original index -- comparison
        # must be positional, not an index-label join.
        if len(source_frame) != len(X):
            raise ValueError(
                f"assert_no_leakage: source_frame has {len(source_frame)} rows but X has "
                f"{len(X)} rows -- cannot reliably value-check without a positional row "
                f"correspondence. Pass a source_frame with the same rows in the same order "
                f"X was assembled from."
            )
        forbidden_present = [c for c in sorted(FORBIDDEN_LEAKAGE_COLUMNS) if c in source_frame.columns]
        for col in X.columns:
            for fcol in forbidden_present:
                try:
                    derived_vals = X[col].astype(float).to_numpy()
                    forbidden_vals = source_frame[fcol].astype(float).to_numpy()
                except (TypeError, ValueError):
                    continue
                if len(derived_vals) < 2:
                    continue
                corr = np.corrcoef(derived_vals, forbidden_vals)[0, 1]
                if corr == corr and abs(corr) >= corr_threshold:  # corr==corr excludes NaN
                    # A high correlation ALWAYS means the claim of safety is
                    # false, whether or not the column was declared -- being
                    # declared in allowed_derivations is a claim to verify,
                    # not immunity. The only difference is the message: a
                    # declared column that still correlates means the
                    # engineer's documented justification was wrong; an
                    # undeclared one means nobody even tried to justify it.
                    if col in allowed:
                        raise AssertionError(
                            f"Leakage guard tripped{f' in {context}' if context else ''}: "
                            f"'{col}' was declared in allowed_derivations as {allowed[col]!r} "
                            f"but correlates {corr:.6f} with forbidden column '{fcol}' "
                            f"(compared positionally against source_frame) -- the declared "
                            f"justification does not hold up. This looks like the forbidden "
                            f"column renamed, not a genuinely safe (e.g. cross-fitted) "
                            f"derivation. Either the derivation isn't actually leak-safe, or "
                            f"the justification needs correcting."
                        )
                    raise AssertionError(
                        f"Leakage guard tripped{f' in {context}' if context else ''}: "
                        f"'{col}' correlates {corr:.6f} with forbidden column '{fcol}' "
                        f"(compared positionally against source_frame) and was NOT declared "
                        f"in allowed_derivations -- it looks like an undeclared leak wearing a "
                        f"different name, not a genuinely safe derivation. If this is actually "
                        f"safe (e.g. cross-fitted within train), name it explicitly in "
                        f"allowed_derivations with a justification."
                    )
        for col, why in allowed.items():
            if col not in X.columns:
                continue
            n_checked = len(forbidden_present)
            logger.info(
                "Leakage guard: '%s' allowed by explicit exception -- %s (value-checked "
                "positionally against %d forbidden column(s) in source_frame, none correlated "
                ">= %.3f)", col, why, n_checked, corr_threshold,
            )


# =============================================================================
# PART A -- Price prediction: baseline vs enhanced (Step 1/2)
# =============================================================================
PROPERTY_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "house_age", "was_renovated", "total_rooms",
]


def _one_hot_align(train_df, test_df, col, prefix):
    """Audit finding R21, actually fixed here 2026-08-28 (previously only
    documented via a comment in the two notebooks' own copies of this
    helper, never fixed in this, the canonical file): reindexing
    `test_dummies` onto `train_dummies`' columns makes a test-set category
    never seen in TRAIN (e.g. a city absent from train after the split)
    collapse to all-zero dummies -- IDENTICAL to how `drop_first=True`
    already encodes the dropped REFERENCE category. Without more
    information the model could not tell "confidently the reference
    category" apart from "a category it has never seen at all". Adds an
    explicit `{prefix}_unseen` indicator column so the two cases are
    distinguishable (0 for every train row, and for any test row whose
    category was present in train; 1 only for a genuinely novel test-set
    category). Low blast radius by design: only used by
    `run_price_experiments`'s BASELINE (one-hot) price/price_per_sqft
    models, not by the enhanced models, Model A/B clustering, or either
    classifier.
    """
    train_dummies = pd.get_dummies(train_df[col], prefix=prefix, drop_first=True).reset_index(drop=True)
    test_dummies = pd.get_dummies(test_df[col], prefix=prefix, drop_first=True)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0).reset_index(drop=True)

    known_categories = set(train_df[col].unique())
    train_dummies[f"{prefix}_unseen"] = 0
    test_dummies[f"{prefix}_unseen"] = (~test_df[col].reset_index(drop=True).isin(known_categories)).astype(int)

    return train_dummies, test_dummies


def train_rf(X_train, y_train, X_test, y_test, random_state=42, n_estimators=300):
    """Random Forest -- deliberately chosen because it needs NO feature
    scaling (it splits on raw feature thresholds, one feature at a time), so
    the scaler debate that mattered so much for clustering doesn't apply
    here. Question this answers: "what type of scaler is used?" for regression.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    r2_test = model.score(X_test, y_test)
    rmse_test = float(np.sqrt(((pred_test - y_test) ** 2).mean()))
    return model, {
        "train_r2": round(model.score(X_train, y_train), 4),
        "test_r2": round(r2_test, 4),
        "test_rmse": round(rmse_test, 2),
        "test_min_pred": round(float(pred_test.min()), 2),
        "test_max_pred": round(float(pred_test.max()), 2),
    }


def get_feature_importance(model: RandomForestRegressor, feature_names) -> list[dict]:
    """Ranks features by how much the enhanced price model actually relies
    on them (scikit-learn's built-in importance: roughly, how much each
    feature reduces prediction error across all the trees).

    Question this answers: the project brief's Step 1 asks for
    "distributions, correlations, AND feature importance" -- this is the
    feature-importance piece, using the real trained model rather than a
    raw correlation number (which can be misleading for a tree model that
    captures non-linear effects and feature interactions).
    """
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in ranked]


def business_insights(df: pd.DataFrame, feature_importance: list[dict], min_sales: int = 15) -> dict:
    """Turns the raw analysis into the kind of summary a real estate
    company could actually act on.

    Question this answers: the project brief's Step 2 asks for "analyze
    price drivers" AND "provide recommendations... where to invest, which
    features matter most" -- this function builds exactly that, from the
    same numbers already computed elsewhere in this script (top cities by
    price efficiency, and the feature-importance ranking above), instead of
    leaving those findings scattered as separate charts with no narrative.
    """
    city_stats = (
        df.groupby("city")
        .agg(n_sales=("price", "size"), avg_price=("price", "mean"), avg_price_per_sqft=("price_per_sqft", "mean"))
        .query("n_sales >= @min_sales")
        .sort_values("avg_price_per_sqft", ascending=False)
    )
    top_cities = city_stats.head(5).round(1).reset_index().to_dict("records")
    bottom_cities = city_stats.tail(5).round(1).reset_index().to_dict("records")
    top_features = [f["feature"] for f in feature_importance[:5]]

    # Describe what KIND of signal dominates, generated from the actual ranked
    # list rather than a fixed assumption -- so this sentence can't drift out
    # of sync with whatever the model actually learned on a given run.
    #
    # BUG FIX (found in code review): this used to check "is location present
    # in the top 5" / "is size present" / "is renovation present" and always
    # append them in that fixed location-then-size-then-renovation order if
    # present -- regardless of which one actually ranked highest. Fixed to
    # walk top_features in their real rank order instead.
    location_flags = {"city_target_enc", "zip_target_enc", "city_sale_volume", "zip_sale_volume", "city_price_index", "zip_price_index"}
    size_flags = {"sqft_living", "sqft_above", "sqft_basement", "sqft_lot", "log_sqft_lot", "total_rooms", "avg_room_size"}
    renovation_flags = {"was_renovated", "years_since_renovation", "renovation_recency_ratio", "condition", "condition_x_age"}

    def _category(feat: str) -> str | None:
        if feat in location_flags:
            return "location (city/zip)"
        if feat in size_flags:
            return "property size"
        if feat in renovation_flags:
            return "condition/renovation status"
        return None

    drivers = []
    for feat in top_features:  # already in descending-importance order
        cat = _category(feat)
        if cat and cat not in drivers:
            drivers.append(cat)
    driver_summary = ", then ".join(drivers) if drivers else "a mix of property-specific features"

    # BUG FIX (found in code review): top_cities/bottom_cities would be empty
    # if zero cities meet min_sales, and indexing top_cities[0]/bottom_cities[-1]
    # would crash with an IndexError before any report got written. Not
    # triggered by the current data.csv (30 cities qualify), but guarded now
    # so a smaller/different dataset degrades gracefully instead of crashing.
    if not top_cities or not bottom_cities:
        # BUG FIX 2026-09-03 (dual-audit fix pass, finding D17): this branch
        # is only reachable when `city_stats` (cities meeting the `min_sales`
        # per-city threshold) is EMPTY -- i.e. when NO city qualifies at all,
        # not merely "fewer than min_sales cities qualify" (any nonzero
        # count > 0 still qualifies and reaches the `else` branch below).
        # `min_sales` (default 15) is the minimum SALES a city needs, not a
        # count of cities -- the old wording conflated the two, misstating
        # its own trigger condition. Corrected to describe what actually
        # fired: zero cities cleared the per-city sales floor.
        recommendation = (
            f"No city has at least {min_sales} recorded sales, "
            f"so no city-level investment recommendation is made this run. "
            f"The price model's top 5 predictors, in order, are "
            f"{', '.join(top_features)} -- ranked broadly by {driver_summary}."
        )
    else:
        recommendation = (
            f"Based on {len(city_stats)} cities with at least {min_sales} recorded sales, "
            f"{top_cities[0]['city']} has the highest average price per square foot "
            f"(${top_cities[0]['avg_price_per_sqft']:.0f}/sqft), making it the strongest "
            f"candidate for value-per-square-foot investment among cities with reliable sales "
            f"volume. The price model's top 5 predictors, in order, are "
            f"{', '.join(top_features)} -- ranked broadly by {driver_summary}. "
            f"Conversely, {bottom_cities[-1]['city']} has the lowest price efficiency "
            f"among cities with reliable volume, and may represent either an undervalued "
            f"opportunity or a genuinely lower-demand market -- worth a closer, city-specific "
            f"look rather than a blanket recommendation either way."
        )
    return {"top_cities_by_price_per_sqft": top_cities, "bottom_cities_by_price_per_sqft": bottom_cities,
            "top_price_driving_features": top_features, "recommendation": recommendation}


def run_price_experiments(df: pd.DataFrame, test_size: float, random_state: int) -> dict:
    """Baseline (notebook features) vs Enhanced (+ new engineered features,
    with leak-safe target encoding for city/zipcode, AND leak-safe sale-volume
    counts for city/zipcode -- see leak_safe_count_encode(), fixed 2026-08-27)
    for two targets: price, and price_per_sqft.

    IMPORTANT -- Question this answers: "is it more effective to add
    price_per_sqft as a new feature for modeling the data?"
    Answer, enforced by this code: NO. price_per_sqft is only ever used as
    the TARGET in its own separate model below -- it is never included as an
    INPUT feature in PROPERTY_FEATURES or NEW_NUMERIC_FEATURES when
    predicting `price`. Using it as an input to predict price would be
    target leakage: price_per_sqft is literally price divided by
    sqft_living, so it would let the model "cheat" by almost reversing the
    formula instead of learning real patterns.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    results = {}

    logger.info("Training models for target = 'price'")
    city_tr, city_te = _one_hot_align(train_df, test_df, "city", "city")
    Xb_train = pd.concat([train_df[PROPERTY_FEATURES].reset_index(drop=True), city_tr], axis=1)
    Xb_test = pd.concat([test_df[PROPERTY_FEATURES].reset_index(drop=True), city_te], axis=1)
    y_train, y_test = train_df["price"], test_df["price"]
    _, baseline_price = train_rf(Xb_train, y_train, Xb_test, y_test, random_state)

    city_enc_train, city_enc_test = kfold_target_encode(train_df, test_df, "city", "price", random_state=random_state)
    zip_enc_train, zip_enc_test = kfold_target_encode(train_df, test_df, "zipcode", "price", random_state=random_state)
    Xe_train = train_df[PROPERTY_FEATURES + NEW_NUMERIC_FEATURES].reset_index(drop=True).copy()
    Xe_train["city_target_enc"] = city_enc_train
    Xe_train["zip_target_enc"] = zip_enc_train
    Xe_test = test_df[PROPERTY_FEATURES + NEW_NUMERIC_FEATURES].reset_index(drop=True).copy()
    Xe_test["city_target_enc"] = city_enc_test
    Xe_test["zip_target_enc"] = zip_enc_test
    # bug fix 2026-08-27: overwrite the naive full-dataset city/zip sale-volume
    # counts (computed pre-split in add_new_features) with leak-safe,
    # train-only counts -- see leak_safe_count_encode() docstring above.
    city_vol_train, city_vol_test = leak_safe_count_encode(train_df, test_df, "city")
    zip_vol_train, zip_vol_test = leak_safe_count_encode(train_df, test_df, "zipcode")
    Xe_train["city_sale_volume"] = city_vol_train
    Xe_train["zip_sale_volume"] = zip_vol_train
    Xe_test["city_sale_volume"] = city_vol_test
    Xe_test["zip_sale_volume"] = zip_vol_test
    enhanced_price_model, enhanced_price = train_rf(Xe_train, y_train, Xe_test, y_test, random_state)
    results["price"] = {"baseline": baseline_price, "enhanced": enhanced_price}
    results["_feature_importance"] = get_feature_importance(enhanced_price_model, Xe_train.columns)

    logger.info("Training models for target = 'price_per_sqft'")
    zipc_tr, zipc_te = _one_hot_align(train_df, test_df, "zipcode", "zip")
    Xb_train_s = pd.concat([train_df[PROPERTY_FEATURES].reset_index(drop=True), city_tr, zipc_tr], axis=1)
    Xb_test_s = pd.concat([test_df[PROPERTY_FEATURES].reset_index(drop=True), city_te, zipc_te], axis=1)
    y_train_s, y_test_s = train_df["price_per_sqft"], test_df["price_per_sqft"]
    _, baseline_sqft = train_rf(Xb_train_s, y_train_s, Xb_test_s, y_test_s, random_state)

    city_enc_train_s, city_enc_test_s = kfold_target_encode(train_df, test_df, "city", "price_per_sqft", random_state=random_state)
    zip_enc_train_s, zip_enc_test_s = kfold_target_encode(train_df, test_df, "zipcode", "price_per_sqft", random_state=random_state)
    Xe_train_s = train_df[PROPERTY_FEATURES + NEW_NUMERIC_FEATURES].reset_index(drop=True).copy()
    Xe_train_s["city_target_enc"] = city_enc_train_s
    Xe_train_s["zip_target_enc"] = zip_enc_train_s
    Xe_test_s = test_df[PROPERTY_FEATURES + NEW_NUMERIC_FEATURES].reset_index(drop=True).copy()
    Xe_test_s["city_target_enc"] = city_enc_test_s
    Xe_test_s["zip_target_enc"] = zip_enc_test_s
    # bug fix 2026-08-27: same leak-safe sale-volume overwrite as the price
    # model above -- see leak_safe_count_encode() docstring.
    city_vol_train_s, city_vol_test_s = leak_safe_count_encode(train_df, test_df, "city")
    zip_vol_train_s, zip_vol_test_s = leak_safe_count_encode(train_df, test_df, "zipcode")
    Xe_train_s["city_sale_volume"] = city_vol_train_s
    Xe_train_s["zip_sale_volume"] = zip_vol_train_s
    Xe_test_s["city_sale_volume"] = city_vol_test_s
    Xe_test_s["zip_sale_volume"] = zip_vol_test_s
    _, enhanced_sqft = train_rf(Xe_train_s, y_train_s, Xe_test_s, y_test_s, random_state)
    results["price_per_sqft"] = {"baseline": baseline_sqft, "enhanced": enhanced_sqft}

    return results


# =============================================================================
# PART B -- Clustering exploration grid (Step 3 evidence trail: every
# combination tried, kept so the instructor can see the reasoning, not just
# the final winner).
# =============================================================================
ORIGINAL_CLUSTER_FEATURES = [
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "house_age", "was_renovated",
]
ENGINEERED_CLUSTER_FEATURES = ORIGINAL_CLUSTER_FEATURES[1:] + [
    "total_rooms", "has_basement", "basement_ratio", "lot_utilization", "bath_bed_ratio",
    "avg_room_size", "premium_outlook_score", "condition_x_age", "years_since_renovation",
    "renovation_recency_ratio", "city_sale_volume",
]
LOG_COLS_BY_FEATURESET = {
    "original": ["price", "sqft_living", "sqft_lot"],
    "engineered": ["price", "sqft_living", "sqft_lot", "city_sale_volume"],
}
MIN_CLUSTERS = 4

# Same minimum-cluster-size floor used and validated by Model B's recipes: reject any
# result whose smallest cluster is below half the size an even k-way split would give
# (floor never below MIN_CLUSTER_SIZE_ABS), so a tiny rare-feature-isolated cluster
# (e.g. the ~30 waterfront homes) can't quietly win on silhouette alone.
#
# Fixed 2026-08-28 (audit finding M9): the floor used to be a flat 10% of ALL rows,
# regardless of k. For this dataset (~4,549 rows) that floor is ~455 -- which is also
# roughly the AVERAGE cluster size at k=10 (4,549/10=455). Passing the old floor at
# k=10 therefore required an almost perfectly even split, which essentially never
# happens with real data. That silently made k=6..10 unreachable even though the
# search loop claimed to try them (see run_clustering_grid's k_max). Scaling the floor
# to the EXPECTED size at each k (n_rows/k) instead of a flat share of all rows makes
# every k in range genuinely reachable, not just nominally in the loop.
MIN_CLUSTER_SIZE_ABS = 15
MIN_CLUSTER_SIZE_FRACTION_OF_EXPECTED = 0.5


def _passes_size_floor(labels: np.ndarray, n_rows: int, k: int) -> bool:
    sizes = pd.Series(labels)[pd.Series(labels) != -1].value_counts()
    if sizes.empty:
        return False
    expected_cluster_size = n_rows / max(k, 1)
    floor = max(MIN_CLUSTER_SIZE_ABS, MIN_CLUSTER_SIZE_FRACTION_OF_EXPECTED * expected_cluster_size)
    return bool(sizes.min() >= floor)


def suggest_eps_candidates(X, k, percentiles=(50, 65, 75, 85, 92, 97)):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, -1])
    return sorted({round(float(np.percentile(kth, p)), 4) for p in percentiles})


def eval_kmeans(X, k_max, random_state):
    best = None
    n_rows = len(X)
    for k in range(MIN_CLUSTERS, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        if not _passes_size_floor(labels, n_rows, k):
            continue  # rejects e.g. a tiny all-waterfront cluster -- see note above
        sil = silhouette_score(X, labels)
        if best is None or sil > best["silhouette"]:
            best = {
                "k": k, "silhouette": round(sil, 4),
                "davies_bouldin": round(davies_bouldin_score(X, labels), 4),
                "calinski_harabasz": round(calinski_harabasz_score(X, labels), 1),
                "labels": labels,
            }
    return best


def eval_agglomerative(X, k_max):
    best = None
    n_rows = len(X)
    for k in range(MIN_CLUSTERS, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if not _passes_size_floor(labels, n_rows, k):
            continue
        sil = silhouette_score(X, labels)
        if best is None or sil > best["silhouette"]:
            best = {
                "k": k, "silhouette": round(sil, 4),
                "davies_bouldin": round(davies_bouldin_score(X, labels), 4),
                "calinski_harabasz": round(calinski_harabasz_score(X, labels), 1),
                "labels": labels,
            }
    return best


def eval_dbscan(X, min_samples_list):
    best = None
    n_rows = len(X)
    for min_samples in min_samples_list:
        for eps in suggest_eps_candidates(X, min_samples):
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            mask = labels != -1
            n_clusters = len(set(labels[mask]))
            noise = 1 - mask.mean()
            if n_clusters < MIN_CLUSTERS or mask.sum() < 2 or noise > 0.5:
                continue
            # R18 fix (2026-08-28): the size-floor's EXPECTED cluster size is
            # n_rows/k -- for DBSCAN that denominator must be the number of
            # rows actually assigned to a cluster (mask.sum()), not the full
            # n_rows including noise points that were dropped as -1. Passing
            # the unfiltered n_rows made the floor up to ~2x stricter for
            # DBSCAN than for KMeans/Agglomerative at the same k (noise
            # inflates the numerator's implied "expected size" without ever
            # being in a cluster to meet it). Immaterial on the current
            # winner (568.6 vs. the corrected 562.4 for 4 clusters, 1.1%
            # noise), but a real inconsistency in the check itself.
            if not _passes_size_floor(labels, int(mask.sum()), n_clusters):
                continue  # individual cluster sizes still exclude noise (-1) inside _passes_size_floor itself
            sil = silhouette_score(X[mask], labels[mask])
            if best is None or sil > best["silhouette"]:
                best = {
                    "eps": eps, "min_samples": min_samples, "silhouette": round(sil, 4),
                    "davies_bouldin": round(davies_bouldin_score(X[mask], labels[mask]), 4),
                    "calinski_harabasz": round(calinski_harabasz_score(X[mask], labels[mask]), 1),
                    "n_clusters": n_clusters, "noise_pct": round(noise * 100, 1),
                    "labels": labels,
                }
    return best


def log1p_selected_columns(X: pd.DataFrame, feature_label: str, features: list[str]) -> pd.DataFrame:
    """Log1p's the skewed columns for this feature set (LOG_COLS_BY_FEATURESET),
    exactly as run_clustering_grid does before scaling/PCA. Factored out
    (audit finding M4) so every downstream re-projection of a grid config's
    clusters applies the identical transform the clustering algorithm
    actually saw, instead of silently skipping it."""
    X = X.copy()
    for c in LOG_COLS_BY_FEATURESET[feature_label]:
        if c in features:
            X[c] = np.log1p(X[c])
    return X


def project_grid_config(df: pd.DataFrame, index, features: list[str], feature_label: str,
                         scaler_name: str, n_components: int, random_state: int = 0) -> np.ndarray:
    """log1p -> scale -> PCA, in exactly the sequence run_clustering_grid
    uses to fit/score a config. Shared by run_clustering_grid itself and by
    every plot that re-projects a grid config's clusters into 2D/3D
    (03_winner_clusters_pca.png, 08_grid_winner_clusters_3d.png).

    Audit finding M4: those two plots used to re-scale the winner's raw
    feature columns with the winner's scaler and go straight to PCA,
    omitting the np.log1p step run_clustering_grid actually applied to the
    skewed columns before scoring every config -- so the plots showed
    geometry the clustering algorithm never saw. Routing both the grid
    search and the plots through this one function makes that impossible
    to re-introduce by accident.
    """
    X_raw = log1p_selected_columns(df.loc[index, features], feature_label, features)
    Xs = ALL_SCALER_CLASSES[scaler_name]().fit_transform(X_raw)
    return PCA(n_components=n_components, random_state=random_state).fit_transform(Xs)


def run_clustering_grid(df_capped: pd.DataFrame, df_uncapped: pd.DataFrame, k_max: int,
                         dbscan_min_samples: list[int], random_state: int) -> list[dict]:
    """Runs every combination we tested by hand during this project:
    2 capping choices x 3 scalers x 2 feature sets x 3 algorithms = 36 runs.

    Question this answers, all at once:
      - "is capping the outliers ... or same as the original dataset?"
      - "is there any possibility that changing the scaler would enhance results?"
      - "did you work on the engineered features or the originals?"
      - "why is Agglomerative not better than KMeans?" (see davies_bouldin /
        cluster balance in the report -- a config whose smallest cluster
        falls below the size floor (scales with k -- see
        MIN_CLUSTER_SIZE_ABS / MIN_CLUSTER_SIZE_FRACTION_OF_EXPECTED, fixed
        2026-08-28, audit finding M9) is rejected outright by
        _passes_size_floor(), not just flagged after the fact, so a tiny
        rare-feature-isolated cluster -- e.g. the ~30 waterfront homes --
        can't quietly win on silhouette alone)
    """
    rows = []
    datasets = {"capped": df_capped, "uncapped (log1p only)": df_uncapped}
    scalers = {"StandardScaler": StandardScaler, "RobustScaler": RobustScaler, "MinMaxScaler": MinMaxScaler}
    feature_sets = {"original": ORIGINAL_CLUSTER_FEATURES, "engineered": ENGINEERED_CLUSTER_FEATURES}

    for capping_label, df in datasets.items():
        for feature_label, features in feature_sets.items():
            X_index = df[features].dropna().index
            X_raw = log1p_selected_columns(df.loc[X_index, features], feature_label, features)

            for scaler_name, scaler_cls in scalers.items():
                Xs = scaler_cls().fit_transform(X_raw)
                pca_scan = PCA(n_components=len(features), random_state=random_state).fit(Xs)
                cum_var = np.cumsum(pca_scan.explained_variance_ratio_)
                n_comp = int(np.argmax(cum_var >= 0.80) + 1)
                Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xs)

                km = eval_kmeans(Xp, k_max, random_state)
                ag = eval_agglomerative(Xp, k_max)
                db = eval_dbscan(Xp, dbscan_min_samples)

                for algo, res in [("KMeans", km), ("Agglomerative(Ward)", ag), ("DBSCAN", db)]:
                    if res is None:
                        continue
                    rows.append({
                        "capping": capping_label, "features": feature_label, "scaler": scaler_name,
                        "algorithm": algo, "n_rows": len(X_raw), "pca_components": n_comp,
                        "pca_variance": round(float(cum_var[n_comp - 1]), 3),
                        "silhouette": res["silhouette"], "davies_bouldin": res["davies_bouldin"],
                        "calinski_harabasz": res["calinski_harabasz"],
                        "n_clusters": res.get("n_clusters", res.get("k")),
                        # Audit finding M1: DBSCAN's silhouette is computed on
                        # non-noise points only, so it isn't a like-for-like
                        # comparison against KMeans/Agglomerative (which score
                        # every row) unless the reader can also see how many
                        # rows got dropped as noise. eval_dbscan already
                        # computes this; it just never made it into the row
                        # dict before. None for KMeans/Agglomerative, which
                        # have no noise concept.
                        "noise_pct": res.get("noise_pct"),
                        # noise (-1) excluded here so the count of sizes always
                        # matches n_clusters for every algorithm.
                        "cluster_sizes": sorted(pd.Series(res["labels"])[pd.Series(res["labels"]) != -1].value_counts().tolist(), reverse=True),
                        "_X_index": X_raw.index, "_labels": res["labels"], "_df": df,
                        # Justification-chart data only (added 2026-08-24, not used for
                        # winner selection) -- the full cumulative-variance curve, so the
                        # "why this many PCA components" decision can be plotted, not just
                        # asserted. Excluded from grid_public (underscore prefix), so it
                        # doesn't change the JSON report at all.
                        "_cum_var_curve": cum_var,
                    })
                logger.info("  grid: %s / %s / %s done", capping_label, feature_label, scaler_name)
    return rows


def price_variance_between_clusters(prices: np.ndarray, labels: np.ndarray) -> dict:
    """The ONE shared "does cluster membership track real price" calculation.

    Audit finding L1: this used to exist as 3 separately-written, byte-for-
    byte-equivalent copies (here, in price_variance_check for Model A, and
    in price_variance for Model B's per-group report) -- verified numerically
    identical (all three gave 1.59% on the same input) but only this one
    masked DBSCAN's noise label (-1) before computing sums of squares. The
    other two never needed to (their labels only ever come from KMeans /
    Agglomerative, which never emit -1), but three copies of the same math
    is a maintenance risk regardless -- a fix to one silently doesn't reach
    the others. Collapsed into this single function; price_variance_explained,
    price_variance_check, and price_variance below are now thin wrappers
    around it (audit finding L2: the dead `within_ss` accumulator from the
    original copy here is gone too -- it was computed and never used).
    """
    labels = np.asarray(labels)
    mask = labels != -1  # exclude DBSCAN noise, if any (no-op for KMeans/Agglomerative labels)
    price = np.asarray(prices)[mask]
    labels = labels[mask]
    overall_mean = price.mean()
    total_ss = float(((price - overall_mean) ** 2).sum())
    between_ss = 0.0
    per_cluster = {}
    for lbl in sorted(set(labels)):
        cluster_prices = price[labels == lbl]
        cmean = cluster_prices.mean()
        between_ss += len(cluster_prices) * (cmean - overall_mean) ** 2
        per_cluster[int(lbl)] = {
            "n": int(len(cluster_prices)), "mean_price": round(float(cmean), 2),
            "min_price": round(float(cluster_prices.min()), 2), "max_price": round(float(cluster_prices.max()), 2),
        }
    pct_between = round(100 * between_ss / total_ss, 2) if total_ss else 0.0
    return {
        "pct_price_variance_between_clusters": pct_between,
        "pct_price_variance_within_clusters": round(100 - pct_between, 2),
        "per_cluster": per_cluster,
    }


def price_variance_explained(df: pd.DataFrame, index, labels: np.ndarray) -> dict:
    """For the winning clustering config, checks how much of TRUE house price
    varies BETWEEN clusters vs WITHIN them.

    Question this answers: "if we used this clustering model as a
    preprocessing step for a regression model, would capping / this
    configuration affect price prediction?" -- a cluster ID is only a useful
    regression feature if clusters actually separate by price. This measures
    exactly that, in the units that matter (dollars), not in the abstract
    PCA space the clustering algorithm sees. Thin wrapper around
    price_variance_between_clusters (audit L1) -- kept as its own function
    because callers pass a (df, index) pair rather than a raw price array.
    """
    return price_variance_between_clusters(df.loc[index, "price"].values, labels)


# =============================================================================
# PART C -- Model A: physical-profile clustering (Step 3): uncapped/log1p +
# engineered features + MinMaxScaler + KMeans(k=4).
#
# Audit finding M1 -- corrected 2026-08-28: this used to be described as
# "the winning recipe out of the Part B grid", which is not accurate. The
# Part B grid's own top-silhouette row is DBSCAN (0.6137), not KMeans
# (0.6044) -- see run_clustering_grid / the "Winner of the flat exploration
# grid" section of full_project_report.md, which already reports DBSCAN
# honestly. Model A DELIBERATELY uses KMeans here instead of the grid's
# silhouette winner, for two concrete reasons:
#   1. DBSCAN's silhouette is computed on non-noise points only (it drops
#      ~1.1% of rows as noise on this config, more on the capped one) --
#      it is scoring itself on an easier, self-selected subset, so its
#      raw silhouette is not directly comparable to KMeans/Agglomerative's,
#      which score every row.
#   2. Model A needs every house assigned to exactly one physical group,
#      because Model B (below) partitions ALL rows within each Model-A
#      group -- a "noise" bucket with no group at all would leave those
#      houses with no price-band recipe to apply. Full coverage is a hard
#      requirement here, not just a preference.
# KMeans and Agglomerative(Ward) tie exactly on this config (0.6044,
# identical cluster sizes) because Ward linkage on this data converges to
# the same partition; KMeans is used because it is the one Model B's
# GROUP_RECIPES were originally tuned against.
# =============================================================================
CLUSTER_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "house_age", "was_renovated",
    "total_rooms", "has_basement", "basement_ratio", "lot_utilization",
    "bath_bed_ratio", "avg_room_size", "premium_outlook_score",
    "condition_x_age", "years_since_renovation", "renovation_recency_ratio",
    "city_sale_volume",
]
LOG_FEATURES = ["sqft_living", "sqft_lot", "city_sale_volume"]
PCA_VARIANCE_THRESHOLD = 0.80
N_CLUSTERS = 4  # confirmed best k for this configuration during testing


def prepare_matrix(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    X_raw = df[CLUSTER_FEATURES].copy().dropna()
    for c in LOG_FEATURES:
        X_raw[c] = np.log1p(X_raw[c])
    logger.info("Model A - Log-transformed skewed columns: %s", LOG_FEATURES)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    logger.info("Model A - Scaled all %d features to a 0-1 range with MinMaxScaler.", len(CLUSTER_FEATURES))

    pca_scan = PCA(n_components=len(CLUSTER_FEATURES), random_state=random_state).fit(X_scaled)
    cum_var = np.cumsum(pca_scan.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= PCA_VARIANCE_THRESHOLD) + 1)
    pca_final = PCA(n_components=n_comp, random_state=random_state)
    X_reduced = pca_final.fit_transform(X_scaled)
    pca_info = {
        "n_components": n_comp, "variance_covered": round(float(cum_var[n_comp - 1]), 4),
        # Justification-chart data only (added 2026-08-24): the full curve so the "80%
        # rule" choice of n_comp can be plotted, not just stated as a number.
        "cum_var_curve": cum_var.tolist(),
    }
    logger.info(
        "Model A - PCA: %d features -> %d components covering %.1f%% variance (>=80%% rule).",
        len(CLUSTER_FEATURES), n_comp, pca_info["variance_covered"] * 100,
    )
    return df.loc[X_raw.index], X_reduced, pca_info


def fit_and_evaluate(X: np.ndarray, n_clusters: int, random_state: int) -> dict:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    metrics = {
        "silhouette": round(float(silhouette_score(X, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
    }
    logger.info("Model A - Fit KMeans (k=%d): silhouette=%.4f, DBI=%.4f", n_clusters, metrics["silhouette"], metrics["davies_bouldin"])
    return {"model": model, "labels": labels, "metrics": metrics}


def price_variance_check(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """The honesty check: does cluster membership actually track real price?
    Thin wrapper around price_variance_between_clusters (audit L1)."""
    return price_variance_between_clusters(df["price"].values, labels)


def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profile = df[CLUSTER_FEATURES].copy()
    profile["cluster"] = labels
    return profile.groupby("cluster").agg(["mean", "count"])


# --------------------------------------------------------------------------- #
# Cluster naming -- gives each cluster a short, human-readable name instead
# of just a number. This is auto-generated and rule-based, not a separate
# model -- treat the wording as a first draft, worth a human read before it
# goes into a presentation. Reused by Model B for its physical-group names
# too, so the wording is consistent everywhere it appears.
#
# Audit finding M8, fixed 2026-08-28: the size/age words used to be picked
# by comparing each cluster's MEAN sqft/age against the WHOLE DATASET's
# quartiles. Means regress toward the middle relative to the population
# they're drawn from, so on the real data 3 of Model A's 4 clusters landed
# in the two middle quartile bins and got called "Spacious", while
# "Compact" and "Large" were reachable only in theory. Fixed by ranking the
# clusters' MEDIANS against EACH OTHER instead of against the dataset as a
# whole -- with N clusters and 4 size words, the N clusters are divided into
# 4 evenly-sized rank buckets, so the full word range is used whenever there
# are at least 4 clusters (verified: on this project's 4 physical clusters,
# the fix produces one cluster per word -- Compact/Mid-size/Spacious/Large
# and New/Established/Older/Vintage each used exactly once, see
# RUNNING_LOG.md for the before/after).
# --------------------------------------------------------------------------- #
def compute_naming_reference(df: pd.DataFrame) -> dict:
    return {
        "premium_mean": df["premium_outlook_score"].mean(),
        "renov_rate": df["was_renovated"].mean(),
    }


SIZE_WORDS = ["Compact", "Mid-size", "Spacious", "Large"]
AGE_WORDS = ["New", "Established", "Older", "Vintage"]


def _rank_bucket_words(values_by_cluster: dict[int, float], words: list[str]) -> dict[int, str]:
    """Ranks each cluster's value against the OTHER clusters' values
    (ascending) and assigns `words` by evenly dividing the ranked clusters
    across the word buckets -- e.g. with 4 clusters and 4 words, the
    lowest-ranked cluster gets words[0], the highest gets words[-1], one
    cluster per word. This replaces comparing a cluster's mean against the
    whole dataset's quartiles (audit finding M8 -- see the block comment
    above)."""
    ordered = sorted(values_by_cluster, key=lambda cid: values_by_cluster[cid])
    n = len(ordered)
    return {cid: words[min(int(rank * len(words) / n), len(words) - 1)] for rank, cid in enumerate(ordered)}


def _renovation_finish_name(renov_rate: float, basement_rate: float) -> str | None:
    """Names a cluster from its renovation/basement-finish signature -- the
    ACTUAL naming scheme every shipped artifact and document uses today
    (audit finding X-1, fixed [2026-09-18]). This was a hand rename applied
    directly to the output artifacts (saved_models/cluster_profile_stats.json,
    consolidated_output's friendly_category column) that never had a
    generating code path -- describe_physical_cluster() still built the OLD
    M8 Compact/Mid-size/.../New/Established/... name below, so re-running
    this pipeline would have silently reverted every shipped name. Verified
    against enriched_dataset.csv: `was_renovated` and `has_basement` are
    each cleanly binary (0.0 or 1.0) per cluster, and the four combinations
    reproduce the four shipped names exactly:
      renov=0, basement=0 -> "Untouched Classics"            (cluster 0)
      renov=1, basement=1 -> "Refreshed & Finished Below"    (cluster 1)
      renov=0, basement=1 -> "Untouched, Finished Below"     (cluster 2)
      renov=1, basement=0 -> "Refreshed Classics"            (cluster 3)
    Returns None (caller falls back to the old M8 scheme) if a cluster's
    rates aren't cleanly binary -- e.g. a re-cluster with different data or
    a different --n-clusters value -- so a mismatch fails loud/falls back
    instead of silently mislabeling."""
    if not (renov_rate <= 0.05 or renov_rate >= 0.95):
        return None
    if not (basement_rate <= 0.05 or basement_rate >= 0.95):
        return None
    renovated = renov_rate >= 0.95
    has_basement = basement_rate >= 0.95
    base = "Refreshed" if renovated else "Untouched"
    if not has_basement:
        return f"{base} Classics"
    return f"{base} & Finished Below" if renovated else f"{base}, Finished Below"


def describe_physical_cluster(cluster_df: pd.DataFrame, ref: dict, size_word: str, age_word: str) -> dict:
    """Builds the name/description text for ONE cluster. The name prefers
    the renovation/basement-finish scheme (_renovation_finish_name, audit
    finding X-1) that matches every shipped artifact and document; it falls
    back to the OLD M8 rank-based size_word/age_word scheme (Compact/
    Mid-size/... , New/Established/...) only when a cluster's renovation/
    basement split isn't cleanly binary, so a differently-shaped re-cluster
    still gets a name instead of erroring, while today's normal run
    reproduces the current canonical names exactly."""
    mean_sqft = cluster_df["sqft_living"].mean()
    mean_age = cluster_df["house_age"].mean()
    mean_premium = cluster_df["premium_outlook_score"].mean()
    renov_rate = cluster_df["was_renovated"].mean()
    basement_rate = cluster_df["has_basement"].mean() if "has_basement" in cluster_df.columns else 0.0

    tags = []
    # Threshold retuned 2026-08-27 (user confirmed): the original 2x/0.3
    # threshold worked out to 0.471, which no cluster's real premium mean
    # ever crosses (clusters are 0.10/0.45/0.41/0.12 on the real data) --
    # so this tag never fired despite clusters 1 and 2 genuinely showing
    # 3-4x more premium/waterfront concentration than 0 and 3. Same
    # relative-threshold pattern as the "Frequently Renovated" tag below
    # (multiplier x dataset mean, with a floor), just recalibrated so it
    # can actually detect the real spread instead of sitting above it.
    if mean_premium > max(ref["premium_mean"] * 1.6, 0.25):
        tags.append("Premium/View")
    if renov_rate > max(ref["renov_rate"] * 1.5, 0.15):
        tags.append("Frequently Renovated")

    name = _renovation_finish_name(renov_rate, basement_rate)
    if name is None:
        # Fallback path -- only reachable for a cluster shape that doesn't
        # match today's clean renovation/basement split.
        name = f"{size_word}, {age_word} Homes"
        if tags:
            name += " (" + ", ".join(tags) + ")"

    description = f"Average {mean_sqft:.0f} sqft living area, average house age {mean_age:.0f} years"
    if renov_rate > 0.05:
        description += f", {renov_rate * 100:.0f}% renovated"
    if mean_premium > 0.05:
        description += f", premium outlook score {mean_premium:.2f} (0-4 scale)"
    description += "."

    return {"name": name, "description": description}


def compute_physical_cluster_names(df: pd.DataFrame, labels: np.ndarray) -> dict[int, dict]:
    """Names EVERY physical cluster at once -- the normal entry point for
    cluster naming (audit finding M8). Batched (rather than one cluster at
    a time) because the naming rule is now rank-based across clusters: each
    cluster's size/age word depends on where its median sqft/house_age
    ranks among the OTHER clusters, which requires seeing all of them
    together. Used identically by Model A (naming its own 4 clusters) and
    Model B (re-describing those same physical groups), so both reports
    always agree on a cluster's name."""
    labels = np.asarray(labels)
    cluster_ids = sorted(set(int(l) for l in labels))
    median_sqft = {cid: float(df.loc[labels == cid, "sqft_living"].median()) for cid in cluster_ids}
    median_age = {cid: float(df.loc[labels == cid, "house_age"].median()) for cid in cluster_ids}
    size_words = _rank_bucket_words(median_sqft, SIZE_WORDS)
    age_words = _rank_bucket_words(median_age, AGE_WORDS)

    ref = compute_naming_reference(df)
    return {
        cid: describe_physical_cluster(df.loc[labels == cid], ref, size_words[cid], age_words[cid])
        for cid in cluster_ids
    }


# =============================================================================
# PART D -- Model B: nested price segmentation (Step 3, second half).
# Inside each of Model A's 4 physical groups, a SEPARATE clustering step
# sorts houses into price bands, using a recipe tuned for that group
# specifically. See NESTED_PRICE_EXPLAINED.md for the comparison that
# produced GROUP_RECIPES below.
# =============================================================================
CURATED_FEATURES_TEMPLATE = [
    "price_per_sqft", "{index_col}", "condition", "premium_outlook_score",
    "renovation_recency_ratio", "sqft_living",
]
LOG_COLS = ["price_per_sqft", "sqft_living"]

GROUP_RECIPES = {
    0: {"index": "zip", "scaler": "RobustScaler", "algorithm": "Agglomerative",
        "linkage": "complete", "metric": "cosine", "k": 6},
    1: {"index": "zip", "scaler": "RobustScaler", "algorithm": "Agglomerative",
        "linkage": "complete", "metric": "manhattan", "k": 2},
    2: {"index": "city", "scaler": "StandardScaler", "algorithm": "KMeans",
        "linkage": None, "metric": "euclidean", "k": 5},
    3: {"index": "zip", "scaler": "StandardScaler", "algorithm": "Agglomerative",
        "linkage": "complete", "metric": "cosine", "k": 5},
}
SCALER_CLASSES = {"StandardScaler": StandardScaler, "RobustScaler": RobustScaler}
# Used for re-projecting the FLAT GRID winner's own scaler (which can be any
# of the 3, unlike Model B's per-cluster recipes above, which never pick
# MinMaxScaler) -- bug fix 2026-08-27, see PROJECT_TODO.md: two plotting
# functions used to hardcode MinMaxScaler() here regardless of which scaler
# actually won the grid search, which happened to be harmless only because
# MinMaxScaler is what wins on the current data.
ALL_SCALER_CLASSES = {"StandardScaler": StandardScaler, "RobustScaler": RobustScaler, "MinMaxScaler": MinMaxScaler}

# Ordered, low -> high price-tier vocabulary. Every group's bands map onto
# one of these lists, chosen by how many bands that group has, so "Budget"
# and "Premium" always mean the lowest/highest band whatever the band count.
#
# BUG FIX (found in code review): an earlier version picked names by
# rounding evenly-spaced positions on the 6-name list (np.linspace + round).
# That's mathematically fine for 2, 3, 4, and 6 bands, but for exactly 5
# bands, Python's round-half-to-even tie-breaking at position 2.5 silently
# skipped "Upper-Mid" every time -- confirmed live in an earlier
# enriched_dataset.csv for Physical Groups 2 and 3 (both 5 bands), where a
# band that should read "Upper-Mid" was instead labeled "High-End". Replaced
# with an explicit, hand-picked name list per band count -- no rounding, no
# ambiguity, and a human (not a formula) decided which name to drop when 5
# bands can't fit all 6 names.
TIER_NAMES = ["Budget", "Value", "Mid-Tier", "Upper-Mid", "High-End", "Premium"]
TIER_NAMES_BY_COUNT = {
    1: ["Budget"],
    2: ["Budget", "Premium"],
    3: ["Budget", "Mid-Tier", "Premium"],
    4: ["Budget", "Mid-Tier", "Upper-Mid", "Premium"],
    5: ["Budget", "Value", "Mid-Tier", "Upper-Mid", "Premium"],
    6: TIER_NAMES,
}


def describe_price_tier(band_rank: int, n_bands: int) -> str:
    """band_rank is 0-indexed, 0 = cheapest band in that physical group.
    Maps it onto a curated name list chosen for that exact band count, so
    every band count (1-6) gets a clean, unambiguous, human-reviewed set of
    names instead of a rounding formula that can skip a name unpredictably.

    Also guards against an unassigned band (price_band == -1, meaning a row
    never got clustered into a price band at all) -- an earlier version let
    band_rank=-1 silently wrap around via Python's negative indexing and
    return "Premium", the single most misleading label possible for a
    missing value. Now it returns "Unclassified" instead."""
    if band_rank < 0 or n_bands < 1 or band_rank >= n_bands:
        return "Unclassified"
    if n_bands in TIER_NAMES_BY_COUNT:
        names = TIER_NAMES_BY_COUNT[n_bands]
    else:
        # Fallback for a band count outside our curated 1-6 range (not
        # currently possible with this project's GROUP_RECIPES, but kept
        # so the function degrades gracefully instead of crashing).
        positions = np.linspace(0, len(TIER_NAMES) - 1, n_bands)
        names = [TIER_NAMES[int(round(p))] for p in positions]
    return names[band_rank]


def add_price_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds price_per_sqft, plus two location-value signals:
      - city_price_index: this city's typical price_per_sqft vs. the
        dataset-wide typical value.
      - zip_price_index: same idea, but at the more precise zipcode level,
        smoothed toward the city figure when a zipcode has few sales (so a
        zipcode with 2 sales doesn't produce a wild, noisy index).

    Audit finding L7, fixed 2026-08-28: this used to group the zip-level
    index by `statezip` (e.g. "WA 98103"), while zip_sale_volume elsewhere
    (add_new_features) groups by the extracted 5-digit `zipcode` column
    instead. Verified `statezip` <-> `zipcode` is a 1:1 mapping on this
    dataset (every statezip maps to exactly one zipcode and vice versa), so
    this was two spellings of the same key, not a different partition -- the
    fix changes nothing numerically, it just uses the one column name
    everything else in this file already uses for "zipcode"."""
    df = df.copy()
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    overall_median = df["price_per_sqft"].median()

    city_median = df.groupby("city")["price_per_sqft"].transform("median")
    df["city_price_index"] = city_median / overall_median

    zip_median = df.groupby("zipcode")["price_per_sqft"].transform("median")
    zip_count = df.groupby("zipcode")["price_per_sqft"].transform("count")
    smoothing = 5
    zip_smoothed = (zip_median * zip_count + city_median * smoothing) / (zip_count + smoothing)
    df["zip_price_index"] = zip_smoothed / overall_median
    return df


def add_outlier_flags(df: pd.DataFrame, cols: tuple[str, ...] = ("price", "sqft_living", "sqft_lot"), k: float = 1.5) -> pd.DataFrame:
    """Same 1.5*IQR rule used for capping in Part A, applied here as a FLAG
    column instead of a clip, so the enriched CSV keeps true, uncapped
    values throughout."""
    df = df.copy()
    any_flag = pd.Series(False, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        flag = (df[col] < lower) | (df[col] > upper)
        df[f"is_outlier_{col}"] = flag.astype(int)
        any_flag = any_flag | flag
    df["is_outlier"] = any_flag.astype(int)
    return df


def fit_price_subclusters(sub: pd.DataFrame, recipe: dict, random_state: int,
                           feature_template: list[str] | None = None, log_cols: list[str] | None = None) -> dict:
    """feature_template/log_cols default to the module-level
    CURATED_FEATURES_TEMPLATE/LOG_COLS (i.e. calling this with no override
    reproduces the exact validated Model B behavior, byte-for-byte). The
    override params exist so diagnostic_price_per_sqft_circularity_check
    (audit finding M2) can re-fit the SAME GROUP_RECIPES on a different
    feature set as a control, without touching GROUP_RECIPES or the
    module-level constants themselves."""
    feature_template = CURATED_FEATURES_TEMPLATE if feature_template is None else feature_template
    log_cols = LOG_COLS if log_cols is None else log_cols
    index_col = "zip_price_index" if recipe["index"] == "zip" else "city_price_index"
    features = [f.format(index_col=index_col) for f in feature_template]

    X_raw = sub[features].copy().dropna()
    idx = X_raw.index
    for c in log_cols:
        if c in features:
            X_raw[c] = np.log1p(X_raw[c])

    scaler = SCALER_CLASSES[recipe["scaler"]]()
    Xs = scaler.fit_transform(X_raw)
    pca_scan = PCA(n_components=len(features), random_state=random_state).fit(Xs)
    cum_var = np.cumsum(pca_scan.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= 0.80) + 1)
    Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xs)

    k = recipe["k"]
    if recipe["algorithm"] == "KMeans":
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(Xp)
        metric_for_sil = "euclidean"
    else:
        labels = AgglomerativeClustering(n_clusters=k, linkage=recipe["linkage"], metric=recipe["metric"]).fit_predict(Xp)
        metric_for_sil = recipe["metric"]

    sil = silhouette_score(Xp, labels, metric=metric_for_sil)
    dbi = davies_bouldin_score(Xp, labels)

    # Rename price-band labels 0..k-1 so band 0 = cheapest, ascending by mean price
    prices = sub.loc[idx, "price"].values
    order = sorted(range(k), key=lambda lbl: prices[labels == lbl].mean())
    remap = {old: new for new, old in enumerate(order)}
    ordered_labels = np.array([remap[l] for l in labels])

    return {"index": idx, "labels": ordered_labels, "features": features,
            "n_components": n_comp, "silhouette": round(float(sil), 4), "davies_bouldin": round(float(dbi), 4),
            # Justification-chart data only (added 2026-08-24): lets Model B's
            # per-group PCA component choice be plotted too, same as Model A's.
            "cum_var_curve": cum_var.tolist()}


def price_variance(prices: np.ndarray, labels: np.ndarray) -> float:
    """Model B's per-group price-variance-explained number. Thin wrapper
    around price_variance_between_clusters (audit L1) -- returns just the
    between-clusters percentage, which is all run_nested_model needs."""
    return price_variance_between_clusters(prices, labels)["pct_price_variance_between_clusters"]


def run_nested_model(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, dict]:
    """Runs Model B (price-band clustering) separately inside each of Model
    A's physical groups, using the group's hand-tuned recipe from
    GROUP_RECIPES.

    SAFETY CHECK (added after code review, 2026-08-24): GROUP_RECIPES is
    hardcoded per physical-cluster ID (0-3), each with its own tuned
    scaler/algorithm/k, found by testing many combinations for THAT
    specific group. This only makes sense if Model A actually produced
    exactly those 4 groups. If Model A were ever run with a different
    number of clusters (e.g. --n-clusters 5), a recipe would be missing
    for the new group and this used to fail with a raw, confusing
    KeyError deep inside the loop below. Now it's checked up front with a
    clear explanation instead.

    This does NOT fully solve a related, subtler risk: even with exactly 4
    groups, GROUP_RECIPES trusts that KMeans cluster ID "0" always means
    the same real-world group (e.g. "Untouched Classics") on
    every run. That holds today because of the fixed random seed, but
    isn't independently verified anywhere. As a lightweight, visible
    safeguard (not a hard check), each group's actual stats are logged
    right next to the recipe about to be applied to it, so a human
    reviewing the run log has something concrete to sanity-check against
    -- a real mismatch would show up as "recipe tuned for a small,
    renovated group" being applied to a group whose logged stats say
    large and never-renovated, for example.
    """
    actual_groups = set(int(c) for c in df["physical_cluster"].unique())
    expected_groups = set(GROUP_RECIPES.keys())
    if actual_groups != expected_groups:
        raise ValueError(
            f"Model B has a hand-tuned recipe for physical groups {sorted(expected_groups)}, "
            f"but Model A actually produced groups {sorted(actual_groups)}. Model B's recipes "
            f"are NOT a generic 'any k' setting -- each one was found by testing combinations "
            f"specifically for that group. If you intentionally want a different number of "
            f"physical groups, Model B's recipes need to be re-tuned for the new groups first "
            f"(see the Part B exploration grid for how the original recipes were found)."
        )

    df = df.copy()
    df["price_band"] = -1
    group_reports = {}

    # Rule-based, auto-generated names -- not a separate ML model. Named all
    # at once, ranked against each other (audit finding M8 -- see
    # compute_physical_cluster_names), so a word like "Spacious" means the
    # same relative thing whichever group it's attached to, and so Model A
    # and Model B always agree on a cluster's name (both call this same
    # function on the same physical_cluster column).
    physical_names: dict[int, dict] = compute_physical_cluster_names(df, df["physical_cluster"].values)

    for cluster_id in sorted(int(c) for c in df["physical_cluster"].unique()):
        sub = df[df["physical_cluster"] == cluster_id]
        recipe = GROUP_RECIPES[cluster_id]

        # Sanity-check log line (see docstring above): shows this group's
        # own stats right next to the recipe it's about to receive, so a
        # human can visually confirm they still look like a sensible match
        # (e.g. group 1's recipe was tuned for a small, heavily-renovated
        # group -- its logged stats below should still look that way).
        logger.info(
            "Group %d sanity check before applying its recipe: n=%d, mean_sqft=%.0f, "
            "mean_age=%.0f, renovated=%.0f%% -- about to apply recipe: %s/%s, k=%d",
            cluster_id, len(sub), sub["sqft_living"].mean(), sub["house_age"].mean(),
            100 * sub["was_renovated"].mean(), recipe["scaler"], recipe["algorithm"], recipe["k"],
        )

        fit = fit_price_subclusters(sub, recipe, random_state)
        df.loc[fit["index"], "price_band"] = fit["labels"]

        prices = sub.loc[fit["index"], "price"].values
        pv = price_variance(prices, fit["labels"])

        physical_desc = physical_names[cluster_id]

        group_reports[cluster_id] = {
            "n": len(sub), "recipe": recipe, "n_bands": recipe["k"],
            "silhouette": fit["silhouette"], "davies_bouldin": fit["davies_bouldin"],
            "price_variance_explained": pv,
            "physical_group_name": physical_desc["name"],
            "physical_group_description": physical_desc["description"],
        }
        logger.info(
            "Group %d (n=%d): %s + %s -> %d price bands, silhouette=%.3f, price variance explained=%.1f%% | name: %s",
            cluster_id, len(sub), recipe["scaler"], recipe["algorithm"], recipe["k"], fit["silhouette"], pv,
            physical_desc["name"],
        )

    df["combined_category"] = "Physical Group " + df["physical_cluster"].astype(str) + " - Price Band " + df["price_band"].astype(str)

    # Friendly, human-readable version of the same label, e.g.
    # "Untouched Classics -- Mid-Tier". Price-tier word depends on
    # this row's band position WITHIN its own physical group's band count,
    # via the shared, ordered TIER_NAMES scale (see describe_price_tier).
    def _friendly(row: pd.Series) -> str:
        cid = int(row["physical_cluster"])
        band = int(row["price_band"])
        n_bands = GROUP_RECIPES[cid]["k"]
        tier = describe_price_tier(band, n_bands)
        # bug fix 2026-08-27: "Mid-Tier" already contains the word "Tier",
        # so the naive f"{tier} Tier" produced "...-- Mid-Tier Tier" for
        # ~19% of the dataset (3 of 18 categories). Every other TIER_NAMES
        # entry (Budget/Value/Upper-Mid/High-End/Premium) still gets the
        # " Tier" suffix as originally intended.
        suffix = tier if tier == "Mid-Tier" else f"{tier} Tier"
        return f"{physical_names[cid]['name']} -- {suffix}"

    df["friendly_category"] = df.apply(_friendly, axis=1)

    total_n = sum(g["n"] for g in group_reports.values())
    weighted_pv = sum(g["n"] * g["price_variance_explained"] for g in group_reports.values()) / total_n
    group_reports["_overall"] = {"weighted_price_variance_explained": round(weighted_pv, 2), "total_categories": sum(g["n_bands"] for g in group_reports.values())}
    group_reports["_physical_group_names"] = {cid: physical_names[cid] for cid in physical_names}

    return df, group_reports


def diagnostic_price_per_sqft_circularity_check(df_b: pd.DataFrame, random_state: int) -> dict:
    """Control for audit finding M2: Model B's headline weighted
    price-variance-explained number is partly circular, because its
    clustering feature set (CURATED_FEATURES_TEMPLATE) includes
    price_per_sqft and sqft_living, and price ~= price_per_sqft *
    sqft_living -- so part of what the metric measures is the clusters
    separating on a near-restatement of the target, not an independent
    signal.

    This re-fits the SAME GROUP_RECIPES (untouched -- passed straight
    through, not modified) on the SAME physical groups, with price_per_sqft
    removed from the clustering feature set via fit_price_subclusters'
    override params, and recomputes the same weighted price-variance-
    explained metric run_nested_model reports. Purely diagnostic, like
    diagnostic_scaler_comparison / diagnostic_group_recipe_comparison above
    -- not called by run_nested_model or main()'s primary path, and changes
    no validated Model B number. Its result is the evidence cited in the
    M2 caveat in write_model_b_report.
    """
    control_features = [f for f in CURATED_FEATURES_TEMPLATE if f != "price_per_sqft"]
    control_log_cols = [c for c in LOG_COLS if c != "price_per_sqft"]
    per_group = {}
    for cid, recipe in GROUP_RECIPES.items():
        sub = df_b[df_b["physical_cluster"] == cid]
        fit = fit_price_subclusters(sub, recipe, random_state, feature_template=control_features, log_cols=control_log_cols)
        prices = sub.loc[fit["index"], "price"].values
        pv = price_variance(prices, fit["labels"])
        per_group[cid] = {"n": len(sub), "price_variance_explained": pv}
    total_n = sum(g["n"] for g in per_group.values())
    weighted_pv = sum(g["n"] * g["price_variance_explained"] for g in per_group.values()) / total_n
    return {"weighted_price_variance_explained_excluding_price_per_sqft": round(weighted_pv, 2), "per_group": per_group}


def band_validity_guard(physical_cluster: int, predicted_band: int) -> int:
    """Deployment safety net (audit finding L8): clips a predicted
    price_band to the nearest band that is actually VALID for its
    predicted physical_cluster (band 0..k-1, where k = GROUP_RECIPES[
    physical_cluster]["k"] -- Model B's per-group band count, e.g. group 1
    only has bands 0-1, group 0 has bands 0-5). Nothing in Classifier B's
    training or feature set enforces this today; a mispredicted, out-of-
    range (cluster, band) pair would otherwise reach a downstream
    regressor or app UI as something structurally impossible.

    Provably free on this project's real data: hard-enforcing this exact
    constraint was tested against Classifier B's held-out predictions
    (classifier_b_unified_output/classifier_b_final_report.md, section 5c)
    and changed ZERO predictions on 3639 training rows and 910 held-out
    test rows -- because physical_cluster is already one of Classifier B's
    own input features, so it has nothing left to gain by being told a
    rule it already learned. This only starts doing anything on a future,
    unusual input the classifier gets structurally wrong, which is exactly
    the case it exists to catch.

    Previously this logic existed ONLY inside build_notebooks_classifier_b.py's
    App-notebook cell generator, and only as a WARNING label (it flagged an
    invalid prediction but never corrected it), so it was never a reusable
    artifact a deployed app could import and call, and never actually
    prevented an invalid (cluster, band) pair from reaching downstream code.
    This is the first place it exists as an importable function; it changes
    no prediction that was already valid.
    """
    n_bands = GROUP_RECIPES.get(int(physical_cluster), {}).get("k")
    if n_bands is None:
        return int(predicted_band)  # unknown group -- nothing to clip against, leave as-is
    return int(min(max(int(predicted_band), 0), n_bands - 1))


# =============================================================================
# Plots -- one plots/ folder for the whole run. Filenames are already
# distinct across Part A/B/Model A/Model B, so nothing collides.
# =============================================================================
def make_price_and_grid_plots(df: pd.DataFrame, price_results: dict, grid_rows: list[dict], winner: dict, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -- Distributions (Step 1: "explore distributions, correlations, and
    # feature importance") --
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    sns.histplot(df["price"], bins=50, kde=True, ax=axes[0, 0]); axes[0, 0].set_title("Distribution of Price")
    sns.histplot(df["sqft_living"], bins=50, kde=True, ax=axes[0, 1]); axes[0, 1].set_title("Distribution of Living Area (sqft)")
    sns.histplot(df["bedrooms"], bins=range(0, 10), ax=axes[1, 0]); axes[1, 0].set_title("Distribution of Bedrooms")
    sns.histplot(df["house_age"], bins=40, ax=axes[1, 1]); axes[1, 1].set_title("Distribution of House Age (years)")
    plt.tight_layout()
    fig.savefig(plots_dir / "00_distributions.png", dpi=120)
    plt.close(fig)

    # -- Correlation heatmap --
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap - Numeric Features (incl. engineered)")
    plt.tight_layout()
    fig.savefig(plots_dir / "00_correlation_heatmap.png", dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    labels = ["price\n(baseline)", "price\n(enhanced)", "price_per_sqft\n(baseline)", "price_per_sqft\n(enhanced)"]
    values = [
        price_results["price"]["baseline"]["test_r2"], price_results["price"]["enhanced"]["test_r2"],
        price_results["price_per_sqft"]["baseline"]["test_r2"], price_results["price_per_sqft"]["enhanced"]["test_r2"],
    ]
    bars = plt.bar(labels, values, color=["#95E1D3", "#4ECDC4", "#95E1D3", "#4ECDC4"], edgecolor="black")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    plt.ylabel("Test R²"); plt.title("Price Prediction: Baseline vs. Enhanced Features")
    plt.tight_layout(); fig.savefig(plots_dir / "01_price_baseline_vs_enhanced.png", dpi=120); plt.close(fig)

    grid_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in grid_rows])
    grid_df["config"] = grid_df["capping"] + " | " + grid_df["features"] + " | " + grid_df["scaler"]
    fig = plt.figure(figsize=(11, 8))
    pivot = grid_df.pivot_table(index="config", columns="algorithm", values="silhouette")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako", cbar_kws={"label": "Silhouette"})
    plt.title("Clustering Exploration Grid: Silhouette by Configuration")
    plt.tight_layout(); fig.savefig(plots_dir / "02_clustering_grid_heatmap.png", dpi=120); plt.close(fig)

    # Audit finding M4: this used to scale df_winner's raw feature columns
    # and go straight to PCA, silently skipping the np.log1p step
    # run_clustering_grid actually applied to the skewed columns before
    # scoring this config -- so the plot showed geometry the algorithm
    # never clustered on. Now routed through the same helper the grid
    # search itself uses (project_grid_config), so the two can't drift
    # apart again.
    X_idx, labels_arr, df_winner = winner["_X_index"], np.array(winner["_labels"]), winner["_df"]
    winner_features = ENGINEERED_CLUSTER_FEATURES if winner["features"] == "engineered" else ORIGINAL_CLUSTER_FEATURES
    pca2 = project_grid_config(df_winner, X_idx, winner_features, winner["features"], winner["scaler"], n_components=2)
    fig = plt.figure(figsize=(8, 7))
    palette = sns.color_palette("husl", len(set(labels_arr)) + 1)
    for lbl in sorted(set(labels_arr)):
        mask = labels_arr == lbl
        color = "#BBBBBB" if lbl == -1 else palette[lbl % len(palette)]
        plt.scatter(pca2[mask, 0], pca2[mask, 1], s=14, alpha=0.7, color=color,
                    label="noise" if lbl == -1 else f"cluster {lbl}")
    plt.title(f"Winning config clusters (2D projection): {winner['algorithm']}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); fig.savefig(plots_dir / "03_winner_clusters_pca.png", dpi=120); plt.close(fig)

    logger.info("Saved Part A/B charts to %s", plots_dir)


def make_model_a_plots(X: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pca2 = PCA(n_components=2, random_state=0).fit_transform(X)
    fig = plt.figure(figsize=(8, 7))
    palette = sns.color_palette("husl", len(set(labels)))
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        plt.scatter(pca2[mask, 0], pca2[mask, 1], s=14, alpha=0.7, color=palette[lbl], label=f"cluster {lbl}")
    plt.title("Model A: KMeans physical clusters, projected to 2D")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "clusters_pca.png", dpi=120)
    plt.close(fig)
    logger.info("Saved Model A cluster plot to %s", plots_dir)


def make_model_b_plots(df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # BUG FIX (found in code review, 2026-08-24): this used to hardcode
    # `plt.subplots(1, 4, ...)`, assuming exactly 4 physical groups. Now
    # sized from however many physical groups actually exist, so it can't
    # silently break (an IndexError past the 4th axis) if that ever
    # changes -- consistent with the same GROUP_RECIPES safety check in
    # run_nested_model().
    n_groups = df["physical_cluster"].nunique()
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, cid in enumerate(sorted(df["physical_cluster"].unique())):
        sub = df[df["physical_cluster"] == cid]
        order = sorted(sub["price_band"].unique())
        sns.boxplot(data=sub, x="price_band", y="price", order=order, ax=axes[i])
        axes[i].set_title(f"Physical Group {cid}")
        axes[i].set_xlabel("Price band (0 = cheapest)")
        if i == 0:
            axes[i].set_ylabel("Price ($)")
        else:
            axes[i].set_ylabel("")
    plt.suptitle("Price bands within each physical group")
    plt.tight_layout()
    fig.savefig(plots_dir / "price_bands_by_group.png", dpi=120)
    plt.close(fig)
    logger.info("Saved Model B plot to %s", plots_dir)


# =============================================================================
# Reports
# =============================================================================
def _oxford_join(items) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(str(i) for i in items[:-1]) + f", and {items[-1]}"


def write_price_and_grid_report(cleaning_capped: dict, cleaning_uncapped: dict, price_results: dict,
                                 grid_rows: list[dict], winner: dict, price_variance_result: dict, insights: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in grid_rows]
    # Audit finding L3: this used to .pop() straight out of the CALLER's
    # price_results dict, so a second call to this function (or anything
    # else touching the same dict afterward) raised KeyError. Copy first --
    # the pop then only mutates this function's own local copy.
    price_results = dict(price_results)
    feature_importance = price_results.pop("_feature_importance")

    report = {
        "cleaning_capped": cleaning_capped,
        "cleaning_uncapped": cleaning_uncapped,
        "price_prediction": price_results,
        "feature_importance": feature_importance,
        "business_insights": insights,
        "clustering_grid": grid_public,
        "clustering_winner": {k: v for k, v in winner.items() if not k.startswith("_")},
        "clustering_winner_price_variance_check": price_variance_result,
    }
    (output_dir / "full_project_report.json").write_text(json.dumps(report, indent=2, default=str))

    grid_df = pd.DataFrame(grid_public).sort_values("silhouette", ascending=False)
    top10 = grid_df.head(10)[["capping", "features", "scaler", "algorithm", "n_clusters", "silhouette", "davies_bouldin", "noise_pct"]].copy()
    # Audit finding M1: make the noise column readable -- KMeans/Agglomerative
    # rows have no noise concept (None), DBSCAN rows get a "-- N.N% noise" tag
    # so the silhouette column next to it can't be misread as directly
    # comparable to the full-coverage algorithms.
    top10["noise_pct"] = top10["noise_pct"].apply(lambda v: f"{v}% noise" if pd.notna(v) else "-")
    importance_df = pd.DataFrame(feature_importance).head(10)

    # Audit finding L6: this paragraph used to be a fixed string asserting
    # "only MinMaxScaler + engineered survived" -- true on the data this was
    # written against, but hardcoded, so a future grid re-run with different
    # results would silently go stale (the same class of bug as the
    # already-fixed M5/M6 "12 candidates" issue). Derived from grid_df here
    # instead, so it can't drift out of sync with the table right below it.
    attempted_scalers = sorted(ALL_SCALER_CLASSES.keys())
    attempted_feature_sets = ["original", "engineered"]
    present_scalers = sorted(grid_df["scaler"].unique())
    present_feature_sets = sorted(grid_df["features"].unique())
    missing_scalers = [s for s in attempted_scalers if s not in present_scalers]
    missing_feature_sets = [f for f in attempted_feature_sets if f not in present_feature_sets]
    cappings_present = sorted(grid_df["capping"].unique())
    surviving_combos = sorted(set(zip(grid_df["scaler"], grid_df["features"])))
    capping_txt = _oxford_join(cappings_present)
    survivor_txt = _oxford_join([f"{s} + the {f} feature set" for s, f in surviving_combos])

    degenerate_bits = []
    if missing_scalers:
        degenerate_bits.append(f"every {_oxford_join(missing_scalers)} config")
    if missing_feature_sets:
        tag = "/".join(missing_feature_sets)
        degenerate_bits.append(
            f"every '{tag}' (non-engineered) feature-set config" if "original" in missing_feature_sets
            else f"every {tag} feature-set config"
        )
    if degenerate_bits:
        grid_survivor_narrative = (
            f"In this run, {_oxford_join(degenerate_bits)} failed this check at every cluster count "
            f"tried -- not just scored lower, genuinely rejected as degenerate. Only {survivor_txt} "
            f"produced valid, non-degenerate clusters at all, for {capping_txt} data, which is a "
            f"stronger point in favor of "
            f"{'that combination' if len(surviving_combos) == 1 else 'these combinations'} "
            f"than the silhouette score alone would suggest."
        )
    else:
        grid_survivor_narrative = (
            f"In this run, every scaler/feature-set combination attempted produced at least one "
            f"valid, non-degenerate result for {capping_txt} data ({survivor_txt})."
        )

    md = [
        "# Consolidated Pipeline - Part A/B Run Report",
        "",
        "## Part A: Price Prediction",
        "",
        f"Rows after cleaning (capped): {cleaning_capped['rows_out']} "
        f"(outlier rows found in price/sqft_living/sqft_lot: "
        f"{cleaning_capped['outliers_capped']['price']}/{cleaning_capped['outliers_capped']['sqft_living']}/{cleaning_capped['outliers_capped']['sqft_lot']})",
        "",
        "| Target | Model | Test R² | Test RMSE |",
        "|---|---|---|---|",
        f"| price | baseline | {price_results['price']['baseline']['test_r2']} | {price_results['price']['baseline']['test_rmse']} |",
        f"| price | enhanced | {price_results['price']['enhanced']['test_r2']} | {price_results['price']['enhanced']['test_rmse']} |",
        f"| price_per_sqft | baseline | {price_results['price_per_sqft']['baseline']['test_r2']} | {price_results['price_per_sqft']['baseline']['test_rmse']} |",
        f"| price_per_sqft | enhanced | {price_results['price_per_sqft']['enhanced']['test_r2']} | {price_results['price_per_sqft']['enhanced']['test_rmse']} |",
        "",
        "See the price / sqft / bedrooms / house age distributions and the numeric feature "
        "correlations (including the engineered ones) below, and the baseline-vs-enhanced model "
        "comparison that follows (audit finding R17: these plots are generated by this same run "
        "but were previously only named in prose, never actually embedded).",
        "",
        "![](plots/00_distributions.png)",
        "",
        "![](plots/00_correlation_heatmap.png)",
        "",
        "![](plots/01_price_baseline_vs_enhanced.png)",
        "",
        "### Feature importance (top 10, from the enhanced price model)",
        "",
        "What the model actually relies on to predict price -- distributions and correlations only "
        "show pairwise relationships; this shows what the trained model itself leans on, including "
        "non-linear effects and interactions between features.",
        "",
        importance_df.to_markdown(index=False),
        "",
        "![](plots/06_feature_importance.png)",
        "",
        "## Business Insights & Recommendations",
        "",
        f"Top 5 cities by average price per sqft (minimum 15 sales, for reliability; computed on "
        f"TRUE, uncapped prices -- audit finding L5, fixed 2026-08-28: this table used to run on "
        f"the IQR-capped frame, while the CSV, Model A, and Model B all use uncapped prices):",
        "",
        pd.DataFrame(insights["top_cities_by_price_per_sqft"])[["city", "n_sales", "avg_price_per_sqft"]].to_markdown(index=False),
        "",
        f"Bottom 5 cities by average price per sqft (minimum 15 sales, uncapped):",
        "",
        pd.DataFrame(insights["bottom_cities_by_price_per_sqft"])[["city", "n_sales", "avg_price_per_sqft"]].to_markdown(index=False),
        "",
        "**Recommendation:**",
        "",
        insights["recommendation"],
        "",
        f"## Part B: Clustering exploration grid (top {len(top10)} of {len(grid_df)} valid "
        f"configurations, by silhouette)",
        "",
        f"36 combinations were attempted (2 capping choices x 2 feature sets x 3 scalers x 3 "
        f"algorithms). Only {len(grid_df)} produced a non-degenerate result -- every combination "
        f"is rejected outright if its smallest cluster falls below "
        f"max({MIN_CLUSTER_SIZE_ABS}, half the EXPECTED cluster size at that k, i.e. "
        f"{MIN_CLUSTER_SIZE_FRACTION_OF_EXPECTED} x rows/k), which catches the classic trap of "
        f"one rare feature (like the ~30 waterfront homes) isolating its own tiny cluster and "
        f"faking a good silhouette score (floor scales with k as of 2026-08-28, audit finding "
        f"M9 -- a flat 10%-of-all-rows floor used to make most k values unreachable). "
        f"{grid_survivor_narrative}",
        "",
        top10.to_markdown(index=False),
        "",
        f"(`noise_pct` is DBSCAN-only: the % of rows it dropped as unclustered noise rather than "
        f"assigning to a cluster. Its silhouette is computed only on the rows it kept, so it is "
        f"NOT a like-for-like comparison against KMeans/Agglomerative, which score every row --"
        f" audit finding M1.)",
        "",
        "![](plots/02_clustering_grid_heatmap.png)",
        "",
        "![](plots/09_scaler_comparison.png)",
        "",
        "(Why MinMaxScaler won: best raw KMeans silhouette per configuration -- the size floor is "
        "not applied in this particular chart, so a config can still show a decent bar here and "
        "still be rejected elsewhere in this report for isolating a tiny, degenerate cluster.)",
        "",
        "![](plots/10_capped_vs_uncapped.png)",
        "",
        "(Why clustering uses uncapped/log1p-only values while the price model above uses capped "
        "values -- the actual distributions, side by side.)",
        "",
        f"## Winner of the flat exploration grid: {winner['capping']} | {winner['features']} features | {winner['scaler']} | {winner['algorithm']}",
        f"- silhouette = {winner['silhouette']}, Davies-Bouldin = {winner['davies_bouldin']}, "
        f"clusters = {winner['n_clusters']}, cluster sizes = {winner['cluster_sizes']}"
        + (f", noise = {winner['noise_pct']}% of rows dropped" if winner.get("noise_pct") is not None else ""),
        "",
        "![](plots/03_winner_clusters_pca.png)",
        "",
        "![](plots/04_pca_variance_grid_winner.png)",
        "",
        "![](plots/08_grid_winner_clusters_3d.png)",
        "",
        "### Does this winning clustering actually separate homes by PRICE?",
        f"- Price variance explained BETWEEN clusters: {price_variance_result['pct_price_variance_between_clusters']}%",
        f"- Price variance remaining WITHIN clusters: {price_variance_result['pct_price_variance_within_clusters']}%",
        "",
        "(A low 'between clusters' number means: strong silhouette score, but the clusters are NOT "
        "meaningfully organizing homes by price.) This flat, single-pass clustering result is kept "
        "here as the evidence trail for WHY a different approach was adopted for the actual Step 3 "
        "deliverable: see the Model A (physical-profile clustering) and Model B (price segmentation "
        "nested inside each physical group, ~25% price variance explained) reports below.",
        "",
        "**Note (audit finding M1):** this grid's own top-silhouette row is not necessarily what "
        "Model A (below) actually clusters with. Model A always uses KMeans + MinMaxScaler + "
        "engineered features + uncapped/log1p, deliberately, regardless of which algorithm wins "
        "this exploration grid on a given run -- because Model A needs full coverage (every house "
        "in exactly one physical group, so Model B has a group to apply its price-band recipe to), "
        "and because a noise-dropping algorithm's silhouette here isn't scored like-for-like "
        "against one that covers every row. See `best_clustering_report.md` for the full "
        "explanation.",
    ]
    (output_dir / "full_project_report.md").write_text("\n".join(md))
    logger.info("Wrote full_project_report.json / .md to %s", output_dir)


def write_model_a_report(metrics: dict, pca_info: dict, price_check: dict, profile: pd.DataFrame, n_rows: int,
                          cluster_names: dict, output_dir: Path, grid_winner: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_rows": n_rows, "features": CLUSTER_FEATURES, "log_transformed": LOG_FEATURES,
        "pca": pca_info, "kmeans_metrics": metrics, "price_variance_check": price_check,
        "cluster_names": cluster_names,
    }
    (output_dir / "best_clustering_report.json").write_text(json.dumps(report, indent=2, default=str))

    md = [
        "# Model A: Physical Clustering - Run Report",
        "",
        "KMeans + MinMaxScaler + engineered features + uncapped (log1p only)",
        "",
        f"- Rows clustered: {n_rows}",
        f"- PCA: {pca_info['n_components']} components, {pca_info['variance_covered']*100:.1f}% variance covered",
        f"- Silhouette: {metrics['silhouette']}  |  Davies-Bouldin: {metrics['davies_bouldin']}  |  Calinski-Harabasz: {metrics['calinski_harabasz']}",
        "",
    ]
    # Audit finding M1: this cluster IS built with KMeans, but that is NOT
    # the Part B exploration grid's own top-silhouette pick -- say so
    # explicitly, and say why, instead of letting the reader assume KMeans
    # won the grid outright. grid_winner is optional so this function still
    # works if ever called without the grid context available.
    if grid_winner is not None and grid_winner.get("algorithm") != "KMeans":
        noise_txt = (
            f", after dropping {grid_winner['noise_pct']}% of rows as unclustered noise"
            if grid_winner.get("noise_pct") is not None else ""
        )
        md += [
            f"**Why KMeans, not the exploration grid's own silhouette winner?** The Part B "
            f"exploration grid's top-silhouette configuration is actually "
            f"**{grid_winner['algorithm']}** (silhouette={grid_winner['silhouette']}{noise_txt}), "
            f"not KMeans (silhouette={metrics['silhouette']}) -- see `full_project_report.md`. "
            f"Model A deliberately uses KMeans here anyway, for two reasons: (1) DBSCAN's "
            f"silhouette is computed only on the rows it keeps, not the ones it drops as noise, "
            f"so it is not a like-for-like comparison against KMeans/Agglomerative, which score "
            f"every row; (2) Model A needs full coverage -- every house must land in exactly one "
            f"physical group, because Model B (below) applies a price-band recipe to every row "
            f"within every group, and a noise bucket would leave some houses with no group at all.",
            "",
        ]
    md += [
        "## Cluster categories",
        "",
        "Names are auto-generated by ranking each cluster's median size/age against the OTHER "
        "clusters (not against the whole dataset's quartiles -- fixed 2026-08-28, audit finding "
        "M8: comparing per-cluster MEANS to dataset-wide quartiles regressed toward the middle, "
        "so 'Compact' and 'Large' were almost never reachable) -- a first draft worth a human "
        "read before using in a presentation.",
        "",
        "| Cluster | Name | Description |",
        "|---|---|---|",
    ]
    for lbl, info in cluster_names.items():
        md.append(f"| {lbl} | **{info['name']}** | {info['description']} |")
    md += [
        "",
        "![](plots/clusters_pca.png)",
        "",
        "![](plots/05_pca_variance_model_a.png)",
        "",
        "![](plots/07_model_a_clusters_3d.png)",
        "",
        "## Cluster sizes and price range",
        "",
        "| Cluster | Size | Mean price | Min price | Max price |",
        "|---|---|---|---|---|",
    ]
    for lbl, stats in price_check["per_cluster"].items():
        md.append(f"| {lbl} | {stats['n']} | ${stats['mean_price']:,.0f} | ${stats['min_price']:,.0f} | ${stats['max_price']:,.0f} |")
    md += [
        "",
        f"## Honesty check: does this clustering separate homes by price?",
        f"- Price variance explained BETWEEN clusters: {price_check['pct_price_variance_between_clusters']}%",
        f"- Price variance left WITHIN clusters: {price_check['pct_price_variance_within_clusters']}%",
        "",
        "**In plain English: no, not really.** Every cluster spans a very wide price range (see table "
        "above). This clustering is good at grouping houses by physical/renovation profile, but a "
        "cluster label here would add very little signal if you plugged it into a price-prediction model. "
        "That's exactly WHY Model B (below) exists -- to find the price structure hiding inside each "
        "physical group.",
        "",
        "## Cluster profile (average feature values, original units)",
        "",
        "```",
        profile.to_string(),
        "```",
    ]
    (output_dir / "best_clustering_report.md").write_text("\n".join(md))
    logger.info("Wrote best_clustering_report.json / .md to %s", output_dir)


def write_model_b_report(group_reports: dict, df: pd.DataFrame, output_dir: Path,
                          price_variance_control: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = dict(group_reports)
    if price_variance_control is not None:
        report_json["_price_per_sqft_circularity_control"] = price_variance_control
    (output_dir / "nested_report.json").write_text(json.dumps(report_json, indent=2, default=str))

    md = ["# Model B: Nested Price Segmentation - Run Report", "",
          "Model A (physical clustering) groups all homes into 4 physical types. "
          "Model B then sorts homes into price bands SEPARATELY inside each physical "
          "group, using a recipe tuned for that group specifically.", "",
          f"**Overall (weighted) price variance explained: {group_reports['_overall']['weighted_price_variance_explained']}%** "
          f"across {group_reports['_overall']['total_categories']} total price bands.", ""]
    # Audit finding M2: this headline number is partly circular -- the
    # clustering feature set includes price_per_sqft and sqft_living, and
    # price ~= price_per_sqft * sqft_living, so part of what it measures is
    # the clusters separating on a near-restatement of the target. Caveat
    # plus a control number (price_per_sqft removed from the feature set,
    # same GROUP_RECIPES, same physical groups -- see
    # diagnostic_price_per_sqft_circularity_check) so the reader can see how
    # much of the headline number survives without it.
    if price_variance_control is not None:
        control_pct = price_variance_control["weighted_price_variance_explained_excluding_price_per_sqft"]
        headline_pct = group_reports["_overall"]["weighted_price_variance_explained"]
        md += [
            f"**Caveat (audit finding M2, fixed 2026-08-28):** this metric is partly circular. "
            f"`price_per_sqft` (and `sqft_living`) are both in Model B's clustering feature set, "
            f"and `price ~= price_per_sqft x sqft_living` -- so part of what {headline_pct}% "
            f"measures is the clusters separating on a near-restatement of the price target, not "
            f"purely an independent signal. **Control: re-fitting the same GROUP_RECIPES on the "
            f"same physical groups with `price_per_sqft` removed from the clustering feature set "
            f"gives {control_pct}%** (vs. {headline_pct}% with it included) -- most of the headline "
            f"number survives without the circular feature, but not all of it. See "
            f"`diagnostic_price_per_sqft_circularity_check` in `consolidated_pipeline.py`.",
            "",
        ]
    md += [
        "![](plots/price_bands_by_group.png)",
        "",
        "![](plots/11_price_variance_breakdown.png)",
        "",
        "![](plots/12_model_b_recipe_comparison.png)",
        "",
        "(The recipe-comparison chart above shows, per physical group, the chosen recipe against "
        "genuinely close alternatives that still pass the minimum-cluster-size floor, and against "
        "alternatives that only score higher by isolating a tiny, degenerate handful of houses into "
        "their own band -- see `diagnostic_group_recipe_comparison`.)",
        "",
        "![](plots/13_pca_variance_model_b.png)",
        "",
    ]
    md += ["## Cluster categories", "",
          "| Physical group | Name | Description |", "|---|---|---|"]
    excluded_keys = {"_overall", "_physical_group_names"}
    for cid in sorted(k for k in group_reports if k not in excluded_keys):
        g = group_reports[cid]
        md.append(f"| {cid} | {g['physical_group_name']} | {g['physical_group_description']} |")
    md.append("")
    # Audit finding M8 fallout: the worked example used to hardcode a specific
    # cluster's name ("Spacious, Established Homes"), which went stale the
    # moment the naming rule changed which cluster gets which name. Built
    # from the actual first group's real name instead, so it can't drift out
    # of sync with the table right above it (same class of fix as L6).
    example_cid = sorted(k for k in group_reports if k not in excluded_keys)[0]
    example_name = group_reports[example_cid]["physical_group_name"]
    md.append(f"Each physical group's price bands are further named using a shared, "
               f"ordered price-tier scale (Budget -> Value -> Mid-Tier -> Upper-Mid -> "
               f"High-End -> Premium), so e.g. band 0 of a 2-band group is \"Budget\" and "
               f"band 1 is \"Premium\", while a 6-band group uses the full scale. The "
               f"combined friendly name looks like \"{example_name} -- "
               f"Budget Tier\" (or, for the one tier whose own name already contains "
               f"the word \"Tier\", just \"...-- Mid-Tier\", not \"Mid-Tier Tier\"). "
               f"See the `friendly_category` column in `enriched_dataset.csv`.")
    md.append("")
    md.append("## Per physical group")
    md.append("")
    for cid in sorted(k for k in group_reports if k not in excluded_keys):
        g = group_reports[cid]
        r = g["recipe"]
        md.append(
            f"### Physical Group {cid}: {g['physical_group_name']} (n={g['n']})\n"
            f"- Recipe: {r['scaler']} + {r['algorithm']}"
            + (f" ({r['linkage']} linkage, {r['metric']} distance)" if r['algorithm'] == "Agglomerative" else "")
            + f", {r['index']}-level price index\n"
            f"- {g['n_bands']} price bands, silhouette={g['silhouette']}, Davies-Bouldin={g['davies_bouldin']}\n"
            f"- Price variance explained within this group: {g['price_variance_explained']}%\n"
        )
        sub = df[df["physical_cluster"] == cid]
        band_stats = sub.groupby("price_band")["price"].agg(["size", "mean", "min", "max"]).round(0)
        band_stats.index = [f"{b} ({describe_price_tier(int(b), g['n_bands'])})" for b in band_stats.index]
        band_stats.index.name = "price_band (tier)"
        md.append("```\n" + band_stats.to_string() + "\n```\n")

    (output_dir / "nested_report.md").write_text("\n".join(md))
    logger.info("Wrote nested_report.json / .md to %s", output_dir)


# =============================================================================
# PART E -- Justification plots (added 2026-08-24, per the user's standing
# rule that every selection/decision should be backed by a picture, not just
# a number or a sentence). Everything below is READ-ONLY on top of results
# Parts A-D already computed and validated -- nothing here changes any
# clustering winner, regression result, or exported CSV/report number.
#
# Two of these functions (diagnostic_scaler_comparison and
# diagnostic_group_recipe_comparison) intentionally explore combinations
# OUTSIDE the validated winning configuration, purely to make the "why we
# didn't pick X instead" case visually. They are clearly named
# "diagnostic_*", are never used to choose anything, and don't touch
# GROUP_RECIPES, the grid winner, or any other already-decided result.
# =============================================================================
def plot_pca_variance_curve(cum_var: list, chosen_n: int, threshold: float, title: str, save_path: Path) -> None:
    """Justifies the '80% variance rule' PCA component count used in the
    grid, Model A, and Model B, by actually plotting the cumulative
    explained-variance curve instead of just stating the chosen number."""
    cum_var = np.asarray(cum_var)
    fig = plt.figure(figsize=(7, 4.5))
    xs = np.arange(1, len(cum_var) + 1)
    plt.plot(xs, cum_var, marker="o", markersize=4, color="#4ECDC4")
    plt.axhline(threshold, color="#888888", linestyle="--", linewidth=1, label=f"{threshold*100:.0f}% threshold")
    plt.axvline(chosen_n, color="#E67E22", linestyle=":", linewidth=1.5,
                label=f"chosen: {chosen_n} components ({cum_var[chosen_n-1]*100:.1f}%)")
    plt.scatter([chosen_n], [cum_var[chosen_n - 1]], color="#E67E22", zorder=5, s=50)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative variance explained")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pca_variance_curves_grid(curves: list[tuple], threshold: float, suptitle: str, save_path: Path) -> None:
    """Multi-panel version of plot_pca_variance_curve, one panel per item
    -- used for Model B, where each physical group has its own PCA fit."""
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, cum_var, chosen_n) in zip(axes, curves):
        cum_var = np.asarray(cum_var)
        xs = np.arange(1, len(cum_var) + 1)
        ax.plot(xs, cum_var, marker="o", markersize=4, color="#4ECDC4")
        ax.axhline(threshold, color="#888888", linestyle="--", linewidth=1)
        ax.axvline(chosen_n, color="#E67E22", linestyle=":", linewidth=1.5)
        ax.scatter([chosen_n], [cum_var[chosen_n - 1]], color="#E67E22", zorder=5, s=50)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("PCA components")
    axes[0].set_ylabel("Cumulative variance explained")
    plt.suptitle(suptitle)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(feature_importance: list[dict], save_path: Path, top_n: int = 10) -> None:
    """Visual version of the price-driver ranking already in the report's
    markdown table -- Step 2 asks to 'analyze price drivers', and a bar
    chart makes 'what matters most' immediately obvious."""
    top = feature_importance[:top_n][::-1]  # reversed so #1 ends up on top of a horizontal bar chart
    labels = [f["feature"] for f in top]
    values = [f["importance"] for f in top]
    fig = plt.figure(figsize=(8, 5.5))
    bars = plt.barh(labels, values, color="#4ECDC4", edgecolor="black")
    for bar, v in zip(bars, values):
        plt.text(v + 0.002, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    plt.xlabel("Feature importance (enhanced price model)")
    plt.title(f"Top {top_n} price-driving features")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_clusters_3d(X3: np.ndarray, labels: np.ndarray, title: str, save_path: Path) -> None:
    """3D version of the existing 2D PCA cluster scatter plots -- shows
    cluster separation more richly than 2D alone, useful for a
    presentation. X3 must already have >=3 columns (first 3 are used)."""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    palette = sns.color_palette("husl", len(set(labels)) + 1)
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        color = "#BBBBBB" if lbl == -1 else palette[lbl % len(palette)]
        ax.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2], s=14, alpha=0.7, color=color,
                   label="noise" if lbl == -1 else f"cluster {lbl}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def diagnostic_best_kmeans_ignoring_floor(X: np.ndarray, k_max: int, random_state: int) -> dict:
    """DIAGNOSTIC ONLY -- not used to pick anything. Same PCA/KMeans setup
    as the real grid search, but does NOT apply the minimum-cluster-size
    floor. Used only to show, visually, that a low-scoring scaler often did
    fine on raw silhouette and got rejected specifically for producing
    tiny, isolated clusters -- not because the algorithm failed outright."""
    best = None
    n_rows = len(X)
    for k in range(MIN_CLUSTERS, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        sil = silhouette_score(X, labels)
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "silhouette": round(float(sil), 4), "passes_floor": _passes_size_floor(labels, n_rows, k)}
    return best


def diagnostic_scaler_comparison(df_capped: pd.DataFrame, df_uncapped: pd.DataFrame, random_state: int, k_max: int = 8) -> pd.DataFrame:
    """DIAGNOSTIC ONLY (see Part E docstring) -- justifies 'why MinMaxScaler
    won' by showing, for every capping x feature-set x scaler combination,
    the best raw KMeans silhouette WITHOUT the minimum-cluster-size floor,
    colored by whether that best result would have passed the floor. This
    never overrides or recomputes the real grid winner in run_clustering_grid."""
    rows = []
    datasets = {"capped": df_capped, "uncapped (log1p only)": df_uncapped}
    scalers = {"StandardScaler": StandardScaler, "RobustScaler": RobustScaler, "MinMaxScaler": MinMaxScaler}
    feature_sets = {"original": ORIGINAL_CLUSTER_FEATURES, "engineered": ENGINEERED_CLUSTER_FEATURES}
    for capping_label, df in datasets.items():
        for feature_label, features in feature_sets.items():
            X_raw = df[features].copy().dropna()
            log_cols = [c for c in LOG_COLS_BY_FEATURESET[feature_label] if c in features]
            for c in log_cols:
                X_raw[c] = np.log1p(X_raw[c])
            for scaler_name, scaler_cls in scalers.items():
                Xs = scaler_cls().fit_transform(X_raw)
                pca_scan = PCA(n_components=len(features), random_state=random_state).fit(Xs)
                cum_var = np.cumsum(pca_scan.explained_variance_ratio_)
                n_comp = int(np.argmax(cum_var >= 0.80) + 1)
                Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xs)
                best = diagnostic_best_kmeans_ignoring_floor(Xp, k_max, random_state)
                rows.append({"capping": capping_label, "features": feature_label, "scaler": scaler_name,
                             "best_silhouette": best["silhouette"], "passes_size_floor": best["passes_floor"]})
    return pd.DataFrame(rows)


def plot_scaler_comparison(diag_df: pd.DataFrame, save_path: Path) -> None:
    fig = plt.figure(figsize=(11, 6.5))
    diag_df = diag_df.copy()
    diag_df["config"] = diag_df["capping"] + " | " + diag_df["features"] + " | " + diag_df["scaler"]
    diag_df = diag_df.sort_values(["scaler", "capping", "features"])
    colors = diag_df["passes_size_floor"].map({True: "#4ECDC4", False: "#E67E22"})
    bars = plt.barh(diag_df["config"], diag_df["best_silhouette"], color=colors, edgecolor="black")
    plt.xlabel("Best raw KMeans silhouette (size floor NOT applied)")
    plt.title(
        "Why MinMaxScaler won: raw silhouette by config\n"
        "teal = would pass the min-cluster-size floor  |  orange = would be rejected as degenerate",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_capped_vs_uncapped(df_capped: pd.DataFrame, df_uncapped: pd.DataFrame, save_path: Path) -> None:
    """Justifies the decision to cap outliers for price prediction but NOT
    for clustering, by actually showing the two distributions side by side
    instead of just describing the difference in text."""
    cols = ["price", "sqft_living", "sqft_lot"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(cols):
        combined = pd.concat([
            pd.DataFrame({col: df_capped[col], "version": "capped (used for price model)"}),
            pd.DataFrame({col: df_uncapped[col], "version": "uncapped, log1p only (used for clustering)"}),
        ])
        sns.boxplot(data=combined, x="version", y=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=20)
        axes[i].set_xlabel("")
    plt.suptitle("Why we don't cap outliers for clustering: capped vs. uncapped distributions")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_price_variance_breakdown(grid_pv: dict, model_a_pv: dict, model_b_weighted_pct: float, save_path: Path) -> None:
    """Visual version of the clustering 'honesty check' -- how much of
    real price variance is explained BETWEEN clusters vs left WITHIN them,
    for the flat grid winner, Model A, and Model B. Numbers already exist
    in the reports as text; this makes the comparison land at a glance."""
    labels = ["Flat grid winner\n(evidence trail)", "Model A\n(physical groups)", "Model B\n(nested price bands)"]
    between = [grid_pv["pct_price_variance_between_clusters"], model_a_pv["pct_price_variance_between_clusters"], model_b_weighted_pct]
    within = [100 - b for b in between]
    fig = plt.figure(figsize=(8, 5.5))
    x = np.arange(len(labels))
    plt.bar(x, between, label="Between clusters (explained)", color="#4ECDC4", edgecolor="black")
    plt.bar(x, within, bottom=between, label="Within clusters (unexplained)", color="#E0E0E0", edgecolor="black")
    for i, b in enumerate(between):
        plt.text(i, b + 1, f"{b:.1f}%", ha="center", fontweight="bold", fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel("% of real price variance")
    plt.title("Does each clustering actually separate homes by price?")
    plt.legend(loc="upper right", fontsize=8)
    plt.ylim(0, 108)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _alt_recipes_for(recipe: dict) -> list[dict]:
    """DIAGNOSTIC ONLY -- builds 3 contrast candidates for the per-group
    recipe comparison chart: swap the scaler, swap the algorithm, and swap
    the zip/city location signal. These mirror the exact dimensions
    NESTED_PRICE_EXPLAINED.md already describes having been tested."""
    alts = []
    other_scaler = "RobustScaler" if recipe["scaler"] == "StandardScaler" else "StandardScaler"
    alts.append({**recipe, "scaler": other_scaler})
    if recipe["algorithm"] == "KMeans":
        alts.append({**recipe, "algorithm": "Agglomerative", "linkage": "complete", "metric": "cosine"})
    else:
        alts.append({**recipe, "algorithm": "KMeans", "linkage": None, "metric": "euclidean"})
    other_index = "city" if recipe["index"] == "zip" else "zip"
    alts.append({**recipe, "index": other_index})
    return alts


def _recipe_short_label(recipe: dict) -> str:
    algo = recipe["algorithm"] if recipe["algorithm"] == "KMeans" else f"Agglom.({recipe['metric']})"
    return f"{recipe['scaler']}\n{algo}\n{recipe['index']}-level"


def diagnostic_group_recipe_comparison(df_b: pd.DataFrame, random_state: int) -> dict:
    """DIAGNOSTIC ONLY (see Part E docstring) -- for each physical group,
    re-fits the chosen recipe PLUS 3 contrast alternatives (same k, holding
    everything else about the search space fixed) and records each one's
    real price-variance-explained, so the 'why this recipe for this group'
    claim in NESTED_PRICE_EXPLAINED.md can be shown, not just told. Does
    NOT change GROUP_RECIPES or any already-validated Model B number."""
    results = {}
    for cluster_id, recipe in GROUP_RECIPES.items():
        sub = df_b[df_b["physical_cluster"] == cluster_id]
        candidates = [("chosen", recipe)] + [(f"alt {i+1}", alt) for i, alt in enumerate(_alt_recipes_for(recipe))]
        rows = []
        for tag, cand in candidates:
            fit = fit_price_subclusters(sub, cand, random_state)
            prices = sub.loc[fit["index"], "price"].values
            pv = price_variance(prices, fit["labels"])
            # Same minimum-cluster-size floor used everywhere else in this project (see
            # MIN_CLUSTER_SIZE_ABS/MIN_CLUSTER_SIZE_FRACTION_OF_EXPECTED) -- applied here too
            # so an alternative that only "wins" on price-variance by isolating a tiny handful
            # of houses into their own band doesn't look like a legitimately better recipe.
            passes_floor = _passes_size_floor(fit["labels"], len(sub), cand["k"])
            rows.append({"tag": tag, "recipe": cand, "label": _recipe_short_label(cand),
                         "price_variance_explained": pv, "silhouette": fit["silhouette"],
                         "cum_var_curve": fit["cum_var_curve"], "n_components": fit["n_components"],
                         "passes_size_floor": passes_floor})
        results[cluster_id] = rows
    return results


def _recipe_bar_color(r: dict) -> str:
    if r["tag"] == "chosen":
        return "#4ECDC4"
    # Same size-floor convention as the scaler-comparison chart: an alternative
    # that "wins" on price variance only by isolating a tiny cluster is flagged
    # orange, same as a degenerate config would be flagged in the flat grid.
    return "#BBBBBB" if r.get("passes_size_floor", True) else "#E67E22"


def plot_group_recipe_comparison(results: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (cluster_id, rows) in zip(axes, sorted(results.items())):
        colors = [_recipe_bar_color(r) for r in rows]
        ax.bar([r["label"] for r in rows], [r["price_variance_explained"] for r in rows], color=colors, edgecolor="black")
        ax.set_title(f"Physical Group {cluster_id}")
        ax.tick_params(axis="x", labelsize=7)
        if cluster_id == sorted(results.keys())[0]:
            ax.set_ylabel("Price variance explained (%)")
    plt.suptitle(
        "Why this recipe for each group? (same k held fixed per group)\n"
        "teal = chosen | gray = a genuinely close/valid alternative | orange = only 'wins' by isolating a tiny, degenerate cluster",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =============================================================================
# main -- runs Steps 1, 2 and 3 end to end, in the order they naturally
# depend on each other:
#   raw data -> clean (capped, for price regression) -> clean (uncapped, for
#   every clustering step) -> Part A (price prediction + business insights)
#   -> Part B (exploration grid, evidence trail) -> Model A (physical
#   clusters, the actual winner) -> Model B (price bands nested inside each
#   physical group) -> enriched_dataset.csv (the final labeled dataset).
# =============================================================================
def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # BUG FIX (found in code review, 2026-08-24): --n-clusters is a real,
    # documented CLI flag on Model A, but Model B's GROUP_RECIPES has 4
    # hand-tuned recipes (one per physical group) that only make sense for
    # exactly 4 groups. Failing fast HERE, before any of the expensive
    # cleaning/feature-engineering/grid-search work runs, is friendlier
    # than letting Model B crash with a KeyError after everything else
    # already ran. (run_nested_model() has the same check as a second line
    # of defense, in case Model A/B are ever called directly instead of
    # through main().)
    if args.n_clusters != N_CLUSTERS:
        logger.error(
            "--n-clusters was set to %d, but Model B's recipes (GROUP_RECIPES) were only "
            "tuned for %d physical groups. Changing --n-clusters would silently break Model "
            "B. If you want to explore a different number of physical groups, use the Part B "
            "exploration grid instead (it already searches k=%d..10), and Model B's recipes "
            "would need to be re-tuned for the new groups before this pipeline could run "
            "end-to-end with a different --n-clusters.",
            args.n_clusters, N_CLUSTERS, MIN_CLUSTERS,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    k_max = 6 if args.quick else 10
    dbscan_min_samples = [10, 15] if args.quick else [5, 10, 15]

    logger.info("=== STEP 1/2: Load, clean, engineer features (shared by everything below) ===")
    raw = load_data(args.input)

    df_capped, cleaning_capped = clean_data(raw, cap_outliers=True)
    df_capped = add_new_features(add_base_features(df_capped))

    df_uncapped, cleaning_uncapped = clean_data(raw, cap_outliers=False)
    df_uncapped = add_new_features(add_base_features(df_uncapped))

    logger.info("=== PART A: Price Prediction ===")
    price_results = run_price_experiments(df_capped, args.test_size, args.random_state)
    logger.info("price: baseline R2=%.3f -> enhanced R2=%.3f",
                price_results["price"]["baseline"]["test_r2"], price_results["price"]["enhanced"]["test_r2"])

    logger.info("=== PART B: Clustering Exploration Grid (evidence trail) ===")
    grid_rows = run_clustering_grid(df_capped, df_uncapped, k_max, dbscan_min_samples, args.random_state)
    winner = max(grid_rows, key=lambda r: r["silhouette"])
    logger.info("Clustering winner: %s | %s | %s | %s -> silhouette=%.4f",
                winner["capping"], winner["features"], winner["scaler"], winner["algorithm"], winner["silhouette"])

    grid_price_variance = price_variance_explained(winner["_df"], winner["_X_index"], np.array(winner["_labels"]))
    logger.info("Winner price-variance check: %.1f%% between clusters, %.1f%% within",
                grid_price_variance["pct_price_variance_between_clusters"], grid_price_variance["pct_price_variance_within_clusters"])

    if not args.no_plots:
        make_price_and_grid_plots(df_capped, price_results, grid_rows, winner, output_dir)

    feature_importance_list = price_results["_feature_importance"]  # captured before write_price_and_grid_report pops it below
    # Audit finding L5, fixed 2026-08-28: this used to run on df_capped
    # (IQR-capped prices), while the CSV, Model A, and Model B all use
    # uncapped prices -- so the "top cities by $/sqft" table was computed on
    # artificially clipped prices. business_insights only reads
    # price/price_per_sqft/city columns (all present on df_uncapped too;
    # feature_importance is passed in, not recomputed), so this doesn't
    # require retraining anything.
    insights = business_insights(df_uncapped, price_results["_feature_importance"])
    write_price_and_grid_report(cleaning_capped, cleaning_uncapped, price_results, grid_rows, winner, grid_price_variance, insights, output_dir)

    logger.info("=== STEP 3a: Model A -- physical-profile clustering ===")
    df_a, X_a, pca_info_a = prepare_matrix(df_uncapped, args.random_state)
    result_a = fit_and_evaluate(X_a, args.n_clusters, args.random_state)
    price_check_a = price_variance_check(df_a, result_a["labels"])
    profile_a = cluster_profile(df_a, result_a["labels"])

    # Audit finding M8: names are now assigned all-at-once, ranked against
    # each other, via compute_physical_cluster_names -- see its docstring.
    cluster_names_a = compute_physical_cluster_names(df_a, result_a["labels"])
    for lbl, info in cluster_names_a.items():
        logger.info("Model A cluster %d name: %s", lbl, info["name"])

    if not args.no_plots:
        make_model_a_plots(X_a, result_a["labels"], output_dir)

    # Audit finding M1: winner is passed through so the report can state
    # honestly whether KMeans matches the grid's own top-silhouette pick,
    # and explain the deliberate choice when it doesn't.
    write_model_a_report(result_a["metrics"], pca_info_a, price_check_a, profile_a, len(df_a), cluster_names_a,
                          output_dir, grid_winner=winner)
    logger.info("Model A done: silhouette=%.4f | price variance explained by cluster=%.1f%%",
                result_a["metrics"]["silhouette"], price_check_a["pct_price_variance_between_clusters"])

    logger.info("=== STEP 3b: Model B -- nested price segmentation ===")
    df_b = df_a.copy()
    df_b["physical_cluster"] = result_a["labels"]
    logger.info("Model A groups feeding into Model B: sizes = %s", df_b["physical_cluster"].value_counts().sort_index().tolist())
    df_b = add_price_index_features(df_b)
    df_b, group_reports = run_nested_model(df_b, args.random_state)
    df_b = add_outlier_flags(df_b)

    if not args.no_plots:
        make_model_b_plots(df_b, output_dir)

    # Audit finding M2: control check for the circularity caveat (price_per_sqft
    # is in Model B's clustering feature set and price ~= price_per_sqft *
    # sqft_living). Cheap (4 small re-fits, same recipes) -- see the
    # function's own docstring.
    price_variance_control = diagnostic_price_per_sqft_circularity_check(df_b, args.random_state)
    logger.info("M2 control: weighted price variance explained excluding price_per_sqft = %.2f%% (vs %.2f%% with it)",
                price_variance_control["weighted_price_variance_explained_excluding_price_per_sqft"],
                group_reports["_overall"]["weighted_price_variance_explained"])
    write_model_b_report(group_reports, df_b, output_dir, price_variance_control=price_variance_control)

    # Slim file (kept for backward compatibility with earlier deliverables).
    df_b[["price", "physical_cluster", "price_band", "combined_category"]].to_csv(output_dir / "house_categories.csv", index=False)

    # Comprehensive export: every raw + engineered column, true (uncapped)
    # values throughout, plus an is_outlier flag instead of any clipping,
    # plus every segment label (physical cluster, price band, both category
    # strings) -- no model *predictions* are included, per the agreed scope.
    df_b.to_csv(output_dir / "enriched_dataset.csv", index=False)
    logger.info("Wrote enriched_dataset.csv (%d rows x %d cols) to %s", len(df_b), df_b.shape[1], output_dir)

    if not args.no_plots:
        logger.info("=== STEP 3c: Justification plots (per the standing rule: back every decision with a picture) ===")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # -- PCA variance curves: why this many components, in each of the 3 places we use PCA --
        plot_pca_variance_curve(winner["_cum_var_curve"], winner["pca_components"], 0.80,
                                 f"Flat grid winner PCA: {winner['scaler']} / {winner['features']} features",
                                 plots_dir / "04_pca_variance_grid_winner.png")
        plot_pca_variance_curve(pca_info_a["cum_var_curve"], pca_info_a["n_components"], PCA_VARIANCE_THRESHOLD,
                                 "Model A PCA: MinMaxScaler / engineered features",
                                 plots_dir / "05_pca_variance_model_a.png")

        # -- Feature importance bar chart (Step 2: "analyze price drivers") --
        plot_feature_importance(feature_importance_list, plots_dir / "06_feature_importance.png")

        # -- 3D cluster views (2D versions already exist; these add depth) --
        X_a_3d = PCA(n_components=3, random_state=0).fit_transform(X_a)
        plot_clusters_3d(X_a_3d, result_a["labels"], "Model A: physical clusters (3D PCA projection)",
                          plots_dir / "07_model_a_clusters_3d.png")
        # Audit finding M4: same fix as make_price_and_grid_plots' 2D
        # version above -- routed through project_grid_config so this plot
        # includes the np.log1p step run_clustering_grid actually applied
        # before scoring this config, instead of skipping it.
        X_idx, labels_arr, df_winner = winner["_X_index"], np.array(winner["_labels"]), winner["_df"]
        winner_features = ENGINEERED_CLUSTER_FEATURES if winner["features"] == "engineered" else ORIGINAL_CLUSTER_FEATURES
        winner_3d = project_grid_config(df_winner, X_idx, winner_features, winner["features"], winner["scaler"], n_components=3)
        plot_clusters_3d(winner_3d, labels_arr, f"Flat grid winner clusters (3D projection): {winner['algorithm']}",
                          plots_dir / "08_grid_winner_clusters_3d.png")

        # -- Diagnostic: why MinMaxScaler won (not just that it did) --
        scaler_diag = diagnostic_scaler_comparison(df_capped, df_uncapped, args.random_state)
        plot_scaler_comparison(scaler_diag, plots_dir / "09_scaler_comparison.png")

        # -- Capped vs. uncapped: why clustering uses uncapped data --
        plot_capped_vs_uncapped(df_capped, df_uncapped, plots_dir / "10_capped_vs_uncapped.png")

        # -- Price-variance honesty check, visualized across all 3 clustering results --
        plot_price_variance_breakdown(grid_price_variance, price_check_a,
                                       group_reports["_overall"]["weighted_price_variance_explained"],
                                       plots_dir / "11_price_variance_breakdown.png")

        # -- Diagnostic: why each physical group got the recipe it got --
        recipe_diag = diagnostic_group_recipe_comparison(df_b, args.random_state)
        plot_group_recipe_comparison(recipe_diag, plots_dir / "12_model_b_recipe_comparison.png")
        model_b_pca_curves = [
            (f"Group {cid}: {recipe_diag[cid][0]['label'].splitlines()[0]}",
             recipe_diag[cid][0]["cum_var_curve"], recipe_diag[cid][0]["n_components"])
            for cid in sorted(recipe_diag.keys())
        ]
        plot_pca_variance_curves_grid(model_b_pca_curves, 0.80, "Model B PCA: why this many components, per physical group",
                                       plots_dir / "13_pca_variance_model_b.png")

        logger.info("Saved 10 justification plots to %s", plots_dir)

    logger.info(
        "=== DONE. Price R2 %.3f -> %.3f | Grid winner silhouette=%.4f | Model A silhouette=%.4f "
        "(%.1f%% price variance) | Model B %.1f%% weighted price variance across %d categories. ===",
        price_results["price"]["baseline"]["test_r2"], price_results["price"]["enhanced"]["test_r2"],
        winner["silhouette"], result_a["metrics"]["silhouette"], price_check_a["pct_price_variance_between_clusters"],
        group_reports["_overall"]["weighted_price_variance_explained"], group_reports["_overall"]["total_categories"],
    )


if __name__ == "__main__":
    main()
