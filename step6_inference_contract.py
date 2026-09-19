"""Phase A, steps 2-3 (Step 6 -- LLM-based Model Interpretation, see
PROJECT_TODO.md "FINAL ACTION PLAN").

This is the single function a real app calls to turn ONE brand-new house
(raw fields only -- nothing engineered) into the exact 35-feature row the
shipped point-GBM regressor expects, using nothing but
`saved_models/inference_tables.json` (Phase A step 1) plus the same
per-row formulas `consolidated_pipeline.add_base_features`/
`add_new_features` use during training.

Step 2 -- temporal convention: ALL of training data shares one narrow
window (`sale_year` is 2014 for every row; `sale_month` only takes
{5, 6, 7}). A genuinely new house scored with today's real date would sit
~12 years outside anything the model has ever seen, so `sale_year`/
`sale_month` are PINNED to 2014/June by default here -- not read from the
system clock. (The two parameters exist so the parity test below can
override them to a row's own historical values and reproduce that row's
original prediction exactly -- that is a correctness check on the
formulas, not the deployment behavior.)

Step 3 -- age_bucket upper-edge guard: training's oldest house is exactly
114 years old (`yr_built` == 1900, the training data's actual minimum), so
the persisted bins top out at `[..., 100, 115]`. A house older than that
falls outside every bin under a naive `pd.cut`, and -- since
`drop_first=True` drops the "0-10y" reference category -- would silently
get ALL-ZERO age_bucket dummies, which is indistinguishable from "0-10y"
to the model. Fix: any house whose engineered `house_age` exceeds the
persisted `max_trained_house_age` is CLAMPED into the top bucket (which
already means "100y+", i.e. "older than the fixed edges" -- clamping is
the categorically correct assignment, not a hack) and a
`beyond_training_age_range` flag is raised for the confidence/caveat layer
built in a later Phase A step. This function also asserts, before
returning, that at least one age_bucket dummy column is set OR the row
legitimately belongs to the dropped reference category ("0-10y") --
i.e. it never silently ships a row that reached the "all zero dummies by
accident" failure mode this guard exists to prevent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# AUDIT FIX 2026-09-09 (SB4, final audit): anchored to this file's own
# location instead of the process cwd -- see step6_classify_house.py's
# identical fix for the full rationale.
REPO_ROOT = Path(__file__).resolve().parent
INFERENCE_TABLES_PATH = REPO_ROOT / "saved_models" / "inference_tables.json"

# AUDIT FIX 2026-09-09 (SA20, final audit): keyed by resolved path string
# instead of a single `dict | None` slot -- see step6_market_comparison.
# load_enriched_dataset's identical fix for the full rationale (a later
# call with a different `path` used to silently return the first path's
# cached tables instead of loading the new file).
_TABLES_CACHE: dict[str, dict] = {}


def load_inference_tables(path: Path = INFERENCE_TABLES_PATH) -> dict:
    key = str(Path(path).resolve())
    if key not in _TABLES_CACHE:
        with open(path) as f:
            _TABLES_CACHE[key] = json.load(f)
    return _TABLES_CACHE[key]


# 2026-09-18 ADDITION (PACKAGING_TODO item 7 -- the Tier B "outside training
# range" disclosure this project's own `step6_validation.py` docstring had
# promised but never actually built): `TIER_A_BOUNDS` there is deliberately
# far wider than any real house this model trained on -- e.g. bedrooms
# (1, 20) vs. training's actual (1, 9), sqft_lot (200, 5_000_000) vs.
# training's actual (638, 1_074_218). A submission inside Tier A's bounds
# but outside these actual training bounds still gets scored (that's
# intentional -- extrapolation isn't refused, just disclosed), but until
# now got zero caveat for it, unlike `yr_built`/`yr_renovated`, which
# already had this exact treatment (see `yr_built_after_training_window`/
# `yr_renovated_after_training_window` below and their matching reasons in
# step6_confidence.py).
#
# Values below are the TRAIN-SPLIT-ONLY min/max (leak-safe, same
# convention as `step6_confidence.TRAINING_PRICE_P1`/`P99` and
# `step6_validation.OPTIONAL_FIELD_DEFAULTS` -- never the full
# enriched_dataset.csv, which would leak the 908 held-out test rows into a
# threshold used to grade those same rows), computed once and hardcoded
# here as fixed, reviewable constants rather than read live, matching this
# project's established pattern for exactly this kind of number. Computed
# via:
#   df = pd.read_csv("consolidated_output/enriched_dataset.csv")
#   train_df, _ = consolidated_pipeline.group_train_test_split(df)
#   train_df[field].min(), train_df[field].max()
# `basement_ratio` is the one derived (non-raw) field included here --
# named explicitly in the PACKAGING_TODO item 7 write-up's own example
# ("a 90% basement ratio") as exactly the kind of relationship-level
# extrapolation Tier A's per-field bounds can't catch on their own.
# `waterfront`/`view`/`condition` are deliberately NOT included: their
# training min/max already equals Tier A's own bound exactly (full domain
# already observed), so a range-disclosure flag for them could never fire
# and would be dead code. Other derived ratios (`bath_bed_ratio`,
# `avg_room_size`, `lot_utilization`) are ALSO deliberately left out --
# scoped to `basement_ratio` only, per this item's own "avoid scope creep"
# framing in PACKAGING_TODO.md -- even though a house can extrapolate on
# one of those (e.g. 1 bed / 6.75 bath) while every raw field stays inside
# range. Adversarial review (2026-09-18, agent a3e6ab0ee4f548642) built
# several such houses and confirmed the CQR relative-width trigger already
# catches them (routes to "limited evidence" for a different, already-
# accurate reason) -- so the coverage gap here is in caveat WORDING
# specificity, not in whether the estimate gets flagged at all.
TRAINING_OBSERVED_RANGES: dict[str, tuple[float, float]] = {
    "bedrooms": (1.0, 9.0),
    "bathrooms": (0.75, 6.75),
    "floors": (1.0, 3.5),
    "sqft_living": (430.0, 10040.0),
    "sqft_lot": (638.0, 1074218.0),
    "sqft_above": (430.0, 7680.0),
    "sqft_basement": (0.0, 4820.0),
    "basement_ratio": (0.0, 0.5949820788530465),
}


def _extract_zipcode(statezip: str) -> str:
    import re
    m = re.search(r"(\d{5})", str(statezip))
    return m.group(1) if m else ""


def build_feature_row(raw_house: dict, tables: dict | None = None,
                       sale_year: int | None = None, sale_month: int | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """`raw_house` must contain: bedrooms, bathrooms, floors, waterfront,
    view, condition, sqft_living, sqft_lot, sqft_above, sqft_basement,
    yr_built, yr_renovated, city, statezip. (Field-level validity/Tier A
    rejection is a separate, already-designed layer that runs BEFORE this
    function -- this function assumes it already received a logically
    valid house and focuses purely on correct feature assembly.)

    Returns (X_row, flags, engineered): X_row is a single-row DataFrame
    with columns in `feature_order` (ready for `model.predict`); flags is
    a dict of booleans/strings describing anything noteworthy that
    happened during assembly (feeds the confidence/caveat layer, not the
    model); engineered is every raw + engineered scalar field computed
    along the way (house_age, total_rooms, basement_ratio, city,
    statezip, ...) -- `step6_classify_house.classify_house()` reuses this
    directly rather than recomputing the same feature-engineering pass a
    second time (standing rule 11).
    """
    tables = tables if tables is not None else load_inference_tables()
    conv = tables["inference_temporal_convention"]
    sale_year = conv["sale_year"] if sale_year is None else sale_year
    sale_month = conv["sale_month"] if sale_month is None else sale_month

    h = raw_house
    flags: dict = {}

    # --- add_base_features() equivalents, single row -----------------
    house_age = max(0, sale_year - int(h["yr_built"]))

    # 2026-09-17 DESIGN CHANGE: yr_built is now allowed past the pinned
    # sale_year (Tier A's bound moved from 2014 to a wide, real-world outer
    # limit -- see step6_validation.py). A house built after 2014 gets
    # house_age clamped to 0 by the max(0, ...) above -- priced as if it
    # were built in 2014, the newest the model has ever seen -- and this
    # flag tells the confidence layer to disclose that assumption by name,
    # rather than folding it into a generic "wide interval" caveat.
    if int(h["yr_built"]) > sale_year:
        flags["yr_built_after_training_window"] = True
    was_renovated = 1 if h["yr_renovated"] and h["yr_renovated"] > 0 else 0
    total_rooms = h["bedrooms"] + h["bathrooms"]
    zipcode = _extract_zipcode(h["statezip"])

    # --- add_new_features() equivalents, single row -------------------
    # Bad-renovation-date fix (193 training rows had yr_renovated <
    # yr_built -- physically impossible -- and were treated as
    # not-renovated instead of trusting the bad year; same rule here).
    #
    # 2026-09-18 DESIGN CHANGE (validation-logic audit): this block is now
    # UNREACHABLE from the deployed app -- `step6_validation.validate_tier_a()`
    # hard-rejects `yr_renovated < yr_built` before `build_feature_row()` is
    # ever called (this function's own docstring already says it "assumes
    # it already received a logically valid house"; this correction was the
    # one place that assumption didn't hold). Kept in place, not deleted,
    # as a defense-in-depth backstop for any caller that reaches this
    # function without going through Tier A first (e.g. a direct library
    # call, or a future entry point) -- same treatment this project already
    # gives other "documented as unreachable via the live app, kept as a
    # safety net" branches (see step6_classify_house.py's tier1-snap
    # fallback for the same pattern).
    if was_renovated and h["yr_renovated"] < h["yr_built"]:
        was_renovated = 0
        flags["bad_renovation_date_ignored"] = True

    if was_renovated:
        # 2026-09-17 DESIGN CHANGE: yr_renovated's Tier A bound moved from
        # 2014 to 2026 alongside yr_built (see step6_validation.py) so a
        # renovation on a post-2014-built house can be entered at all. A
        # renovation year past the pinned sale_year (2014) is clamped to
        # years_since_renovation=0 by the max(0, ...) below -- priced as if
        # renovated in 2014, the newest the model has ever seen -- and this
        # flag tells the confidence layer to disclose that assumption by
        # name, the same pattern as `yr_built_after_training_window` above.
        if int(h["yr_renovated"]) > sale_year:
            flags["yr_renovated_after_training_window"] = True
        years_since_renovation = max(0, sale_year - int(h["yr_renovated"]))
        renovation_recency_ratio = (years_since_renovation / house_age) if house_age > 0 else -1.0
    else:
        years_since_renovation = -1.0
        renovation_recency_ratio = -1.0

    has_basement = 1 if h["sqft_basement"] > 0 else 0
    basement_ratio = h["sqft_basement"] / h["sqft_living"] if h["sqft_living"] else 0.0

    # 2026-09-18 ADDITION (PACKAGING_TODO item 7): per-field "outside what
    # this model actually trained on" disclosure -- see
    # `TRAINING_OBSERVED_RANGES`'s definition above for the full rationale
    # and how these bounds were derived. Collects every out-of-range field
    # into ONE list (rather than one flag per field) so
    # `step6_confidence.compute_confidence()` can fold them into a single
    # combined caveat sentence, the same pattern already used for
    # `imputed_fields_in_top_k` there -- avoids caveat pile-up when several
    # fields are unusual on the same submission. `basement_ratio` is
    # checked here (not via the raw-field loop) since it isn't one of
    # `raw_house`'s own keys -- it's this function's own locally computed
    # value, checked immediately once available.
    beyond_range_fields = [
        field for field, (lo, hi) in TRAINING_OBSERVED_RANGES.items()
        if field != "basement_ratio" and h.get(field) is not None and not (lo <= h[field] <= hi)
    ]
    br_lo, br_hi = TRAINING_OBSERVED_RANGES["basement_ratio"]
    if not (br_lo <= basement_ratio <= br_hi):
        beyond_range_fields.append("basement_ratio")
    # 2026-09-18 FIX (adversarial subagent review, agent a3e6ab0ee4f548642):
    # `sqft_above` is mechanically capped at `sqft_living` (Tier A rejects
    # sqft_above > sqft_living outright), so whenever sqft_living itself is
    # already out of training range, sqft_above almost always is too --
    # confirmed on the real held-out set, this was an EXACT duplicate pair
    # in 4/4 live fires ("living area size and above-ground living area are
    # outside the range..." naming what reads as two separate problems but
    # is mechanically one). Suppress the redundant half rather than name
    # both -- same "don't tell the user about one fact twice" principle as
    # `group_top_features_by_label`'s display-name merge in
    # step6_llm_narrative.py, just for a different kind of duplication.
    if "sqft_living" in beyond_range_fields and "sqft_above" in beyond_range_fields:
        beyond_range_fields.remove("sqft_above")
    if beyond_range_fields:
        flags["beyond_training_range_fields"] = beyond_range_fields

    lot_utilization = h["sqft_living"] / h["sqft_lot"] if h["sqft_lot"] else 0.0
    bath_bed_ratio = h["bathrooms"] / h["bedrooms"] if h["bedrooms"] else 0.0
    avg_room_size = h["sqft_living"] / total_rooms if total_rooms else 0.0
    premium_outlook_score = max(h["view"], h["waterfront"] * 3)
    condition_x_age = h["condition"] * house_age
    sale_month_sin = math.sin(2 * math.pi * sale_month / 12)
    sale_month_cos = math.cos(2 * math.pi * sale_month / 12)
    log_sqft_lot = math.log1p(h["sqft_lot"])

    # --- age_bucket, with the upper-edge clamp guard -------------------
    bins = tables["age_bucket"]["bins"]
    labels = tables["age_bucket"]["labels"]
    max_trained_age = tables["age_bucket"]["max_trained_house_age"]
    if house_age > max_trained_age:
        age_bucket = labels[-1]  # top bucket already means "beyond the fixed edges"
        flags["beyond_training_age_range"] = True
    else:
        age_bucket = None
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if (lo <= house_age <= hi) if i == 0 else (lo < house_age <= hi):
                age_bucket = labels[i]
                break
        if age_bucket is None:  # defensive -- should be unreachable given the clamp above
            age_bucket = labels[-1]
            flags["age_bucket_fallback_used"] = True

    # --- location lookup, zip -> city -> overall -----------------------
    loc = tables["location_deployment_table"]
    if h["statezip"] in loc["zip_blended"]:
        location_price_per_sqft = loc["zip_blended"][h["statezip"]]
        flags["location_fallback"] = "zip"
    elif h["city"] in loc["city_median"]:
        location_price_per_sqft = loc["city_median"][h["city"]]
        flags["location_fallback"] = "city"
    else:
        location_price_per_sqft = loc["overall_median"]
        flags["location_fallback"] = "global"

    # --- train-only count encodings, with the same min-count fallback --
    city_count_map, city_fallback = tables["city_count_map"], tables["city_count_fallback"]
    zip_count_map, zip_fallback = tables["zip_count_map"], tables["zip_count_fallback"]
    city_sale_volume_safe = city_count_map.get(h["city"], city_fallback)
    if h["city"] not in city_count_map:
        flags["city_unseen_in_training"] = True
    zip_sale_volume_safe = zip_count_map.get(zipcode, zip_fallback)
    if zipcode not in zip_count_map:
        flags["zip_unseen_in_training"] = True

    row = {
        "city": h["city"], "statezip": h["statezip"], "zipcode": zipcode,
        "bedrooms": h["bedrooms"], "bathrooms": h["bathrooms"], "floors": h["floors"],
        "waterfront": h["waterfront"], "view": h["view"], "condition": h["condition"],
        "sqft_living": h["sqft_living"], "sqft_lot": h["sqft_lot"],
        "sqft_above": h["sqft_above"], "sqft_basement": h["sqft_basement"],
        "yr_built": h["yr_built"], "yr_renovated": h["yr_renovated"],
        "house_age": house_age, "was_renovated": was_renovated, "total_rooms": total_rooms,
        "years_since_renovation": years_since_renovation, "renovation_recency_ratio": renovation_recency_ratio,
        "has_basement": has_basement, "basement_ratio": basement_ratio, "lot_utilization": lot_utilization,
        "bath_bed_ratio": bath_bed_ratio, "avg_room_size": avg_room_size,
        "premium_outlook_score": premium_outlook_score, "condition_x_age": condition_x_age,
        "log_sqft_lot": log_sqft_lot,
        "sale_month": sale_month, "sale_month_sin": sale_month_sin, "sale_month_cos": sale_month_cos,
        "city_sale_volume_safe": city_sale_volume_safe, "zip_sale_volume_safe": zip_sale_volume_safe,
        "location_price_per_sqft": location_price_per_sqft,
    }
    for lbl in labels:
        col = f"age_bucket_{lbl}"
        if col in tables["feature_order"]:
            row[col] = 1 if age_bucket == lbl else 0

    X_row = pd.DataFrame([row]).reindex(columns=tables["feature_order"], fill_value=0)

    # Guard (step 3's explicit "assert no all-zero-dummy row" requirement):
    # a row is only allowed to have every age_bucket dummy at 0 if it
    # genuinely belongs to the dropped reference category ("0-10y" --
    # house_age <= 10). Anything else with all-zero dummies is exactly the
    # silent-misencoding failure mode this guard exists to catch.
    dummy_cols = [c for c in tables["feature_order"] if c.startswith("age_bucket_")]
    all_zero_dummies = all(X_row.iloc[0][c] == 0 for c in dummy_cols)
    if all_zero_dummies and house_age > 10:
        raise AssertionError(
            f"age_bucket produced all-zero dummies for house_age={house_age} "
            f"(age_bucket={age_bucket!r}), which is indistinguishable from the '0-10y' "
            f"reference category -- this should be unreachable after the clamp guard."
        )

    return X_row, flags, row
