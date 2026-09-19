"""Comprehensive dry-run QA pass: every input field's boundaries, every
required/optional field path, all cross-field checks, and a full
interpret_house() output-sanity sweep across a large, diverse real-house
sample -- run BEFORE the live-browser confirmation pass, per this
project's standing "dry run first, then confirm live" protocol.

This is a test SCRIPT (prints PASS/FAIL per check + a final summary), not
a pytest suite -- matches the existing test_yr_built.py/test_yr_renovated.py
style already in this directory.
"""
import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd

from step6_validation import (
    validate_tier_a, apply_optional_defaults, TIER_A_BOUNDS,
    OPTIONAL_FIELD_DEFAULTS, CRITICAL_FIELDS, MIN_SQFT_PER_BEDROOM, MIN_SQFT_PER_BATHROOM,
)
from step6_interpretation import interpret_house

FAILURES = []
CHECKS = 0


def check(name, condition, detail=""):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")
    else:
        print(f"OK    {name}")


BASE = {
    "bedrooms": 3, "bathrooms": 2.0, "floors": 1.0,
    "waterfront": 0, "view": 0, "condition": 3,
    "sqft_living": 2000, "sqft_lot": 7500,
    "sqft_above": None, "sqft_basement": 0,
    "yr_built": 2000, "yr_renovated": 0,
    "city": "Bellevue", "statezip": "WA 98004", "country": "USA",
}


# =====================================================================
# SECTION 1 -- every TIER_A_BOUNDS field, at and just past its bounds,
# holding every OTHER field at a safe mid-range value so only the field
# under test can trigger a rejection.
# =====================================================================
print("=" * 78)
print("SECTION 1: per-field Tier A boundary sweep")
print("=" * 78)

# Fields whose bound test needs companion fields nudged so no OTHER check
# (bedroom/bathroom ratio, basement/above relationship) fires incidentally.
SAFE_COMPANION = {
    "bedrooms": {"sqft_living": 4000},       # keep sqft/bedroom ratio safe at bedrooms=20
    "bathrooms": {"sqft_living": 4000},      # keep sqft/bathroom ratio safe at bathrooms=15
    "sqft_living": {"bedrooms": 1, "bathrooms": 0.5},  # avoid ratio checks firing at sqft_living=100
    # sqft_above/sqft_basement are RELATIONALLY bound to sqft_living (the
    # validation-logic audit's GAP1/GAP2 fixes), not independently boundable
    # -- their own per-field TIER_A_BOUNDS upper limit (20000/10000) can
    # only be reached at all if sqft_living is at least that large too.
    "sqft_above": {"sqft_living": 20000},
    "sqft_basement": {"sqft_living": 10001},  # +1: basement must be < living, never ==
}
# sqft_above's own per-field lower bound (0) is a DIFFERENT case: it is now
# provably unreachable as a passing value under ANY companion sqft_living,
# because sqft_living is always >= 100 (its own floor) and sqft_above=0
# always means zero above-ground area, which the GAP1-family relational
# check rejects regardless of sqft_living's value. This is a deliberate,
# understood consequence of that fix (documented in the report), not a
# test gap -- skip the generic lower-bound-passes assertion for this one
# field and assert the (correct) rejection instead.
FIELDS_WHERE_LOWER_BOUND_NO_LONGER_PASSES = {"sqft_above"}

for field, (lo, hi) in TIER_A_BOUNDS.items():
    extra = SAFE_COMPANION.get(field, {})
    step = 1 if isinstance(lo, int) else 0.5

    h_lo = dict(BASE, **extra, **{field: lo})
    reasons = validate_tier_a(h_lo)
    if field in FIELDS_WHERE_LOWER_BOUND_NO_LONGER_PASSES:
        check(f"{field} at LOWER bound {lo} -- correctly still rejected (relational check)",
              reasons != [], reasons)
    else:
        check(f"{field} at LOWER bound {lo} passes", reasons == [], reasons)

    h_hi = dict(BASE, **extra, **{field: hi})
    reasons = validate_tier_a(h_hi)
    check(f"{field} at UPPER bound {hi} passes", reasons == [], reasons)

    below = lo - step
    h_below = dict(BASE, **extra, **{field: below})
    reasons = validate_tier_a(h_below)
    check(f"{field} BELOW lower bound ({below}) rejects", reasons != [])

    above = hi + step
    h_above = dict(BASE, **extra, **{field: above})
    reasons = validate_tier_a(h_above)
    check(f"{field} ABOVE upper bound ({above}) rejects", reasons != [])


# =====================================================================
# SECTION 2 -- required (critical) field omission -> must reject
# =====================================================================
print()
print("=" * 78)
print("SECTION 2: required-field omission")
print("=" * 78)
for field in CRITICAL_FIELDS:
    h = dict(BASE)
    h[field] = None
    reasons = validate_tier_a(h)
    check(f"missing required field '{field}' rejects", reasons != [], reasons)
    if reasons:
        check(f"  rejection reason names '{field}'", field in reasons[0], reasons)

# every critical field present -> must pass (sanity baseline)
reasons = validate_tier_a(dict(BASE))
check("baseline house (all fields present, valid) passes", reasons == [], reasons)


# =====================================================================
# SECTION 3 -- optional field omission -> must pass Tier A, then get a
# training-derived default + disclosed as imputed by apply_optional_defaults
# =====================================================================
print()
print("=" * 78)
print("SECTION 3: optional-field omission -> default + disclosure")
print("=" * 78)
OPTIONAL_FIELDS = ["floors", "waterfront", "view", "condition", "sqft_lot", "yr_renovated"]
for field in OPTIONAL_FIELDS:
    h = dict(BASE)
    h[field] = None
    reasons = validate_tier_a(h)
    check(f"missing optional field '{field}' still passes Tier A", reasons == [], reasons)
    filled, imputed = apply_optional_defaults(h)
    check(f"  '{field}' imputed and disclosed", field in imputed, imputed)
    check(f"  '{field}' filled with training default {OPTIONAL_FIELD_DEFAULTS[field]}",
          filled[field] == OPTIONAL_FIELD_DEFAULTS[field], filled[field])

# sqft_above/sqft_basement special-case matrix (UI_AMENDMENTS entry 1 logic)
h = dict(BASE, sqft_above=None, sqft_basement=None, sqft_living=2000)
filled, imputed = apply_optional_defaults(h)
check("both above+basement missing -> both imputed", set(["sqft_above", "sqft_basement"]) <= set(imputed), imputed)
check("  derives basement=0 (no-basement default)", filled["sqft_basement"] == 0, filled)
check("  derives above=sqft_living", filled["sqft_above"] == 2000, filled)

h = dict(BASE, sqft_above=None, sqft_basement=500, sqft_living=2000)
filled, imputed = apply_optional_defaults(h)
check("only basement given -> above is EXACT ARITHMETIC, not flagged imputed",
      "sqft_above" not in imputed, imputed)
check("  above = living - basement exactly", filled["sqft_above"] == 1500, filled)

h = dict(BASE, sqft_above=1500, sqft_basement=None, sqft_living=2000)
filled, imputed = apply_optional_defaults(h)
check("only above given -> basement is EXACT ARITHMETIC, not flagged imputed",
      "sqft_basement" not in imputed, imputed)
check("  basement = living - above exactly", filled["sqft_basement"] == 500, filled)


# =====================================================================
# SECTION 4 -- cross-field relationship checks (consolidated from the
# validation-logic audit -- re-verified here as part of the full sweep)
# =====================================================================
print()
print("=" * 78)
print("SECTION 4: cross-field relationship checks")
print("=" * 78)
CROSS_FIELD_CASES = [
    ("bedroom ratio floor", {"sqft_living": 100, "bedrooms": 5}, True),
    ("bedroom ratio, just above floor", {"sqft_living": MIN_SQFT_PER_BEDROOM + 10, "bedrooms": 1}, False),
    ("bathroom ratio floor", {"sqft_living": 100, "bedrooms": 1, "bathrooms": 15.0}, True),
    # sqft_living=130, bathrooms=4.0 -> 32.5 sqft/bathroom, just above the 30
    # floor, while still respecting sqft_living's OWN absolute floor (>=100)
    # and the bedroom-ratio floor (130 >= 100*1) -- unlike the old
    # MIN_SQFT_PER_BATHROOM+10=40 value, which violated sqft_living's own
    # floor and was rejected for an unrelated reason (test-harness bug, not
    # an app bug).
    ("bathroom ratio, just above floor", {"sqft_living": 130, "bedrooms": 1, "bathrooms": 4.0}, False),
    ("basement == living (0 above)", {"sqft_basement": 2000, "sqft_above": None, "sqft_living": 2000}, True),
    ("basement = living - 1 (boundary ok)", {"sqft_basement": 1999, "sqft_above": None, "sqft_living": 2000}, False),
    ("above > living (negative basement)", {"sqft_above": 2500, "sqft_basement": None, "sqft_living": 2000}, True),
    ("above == living (basement=0 ok)", {"sqft_above": 2000, "sqft_basement": None, "sqft_living": 2000}, False),
    ("all 3 given, above=0", {"sqft_above": 0, "sqft_basement": 2000, "sqft_living": 2000}, True),
    ("all 3 given, sum mismatch", {"sqft_above": 100, "sqft_basement": 100, "sqft_living": 2000}, True),
    ("all 3 given, consistent", {"sqft_above": 1500, "sqft_basement": 500, "sqft_living": 2000}, False),
    ("yr_renovated < yr_built", {"yr_built": 1990, "yr_renovated": 1980}, True),
    ("yr_renovated == yr_built", {"yr_built": 1990, "yr_renovated": 1990}, False),
    ("yr_renovated > yr_built", {"yr_built": 1990, "yr_renovated": 2000}, False),
    ("not renovated (yr_renovated=0)", {"yr_built": 1990, "yr_renovated": 0}, False),
]
for name, overrides, expect_reject in CROSS_FIELD_CASES:
    h = dict(BASE, **overrides)
    reasons = validate_tier_a(h)
    check(f"cross-field: {name}", bool(reasons) == expect_reject, reasons)


# =====================================================================
# SECTION 5 -- full interpret_house() output-sanity sweep across a large,
# diverse REAL sample (all 4 clusters), checking every major output
# surface for internal consistency, not just "did it crash."
# =====================================================================
print()
print("=" * 78)
print("SECTION 5: full interpret_house() output sanity, real-data sample")
print("=" * 78)

df = pd.read_csv("consolidated_output/enriched_dataset.csv")
RAW_FIELDS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
              "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
              "city", "statezip", "country"]

class_tables = json.load(open("saved_models/classification_tables.json"))
RAW_KEY_LEAK_MARKERS = ["Physical Group", "Price Band", "combined_category"]
RAW_FEATURE_NAME_MARKERS = ["sqft_living", "sqft_lot", "condition_x_age", "premium_outlook_score",
                            "bath_bed_ratio", "log_sqft_lot", "avg_room_size", "lot_utilization"]

confidence_levels_seen = set()
narrative_sources_seen = set()
n_tested, n_exceptions, n_unexpected_reject = 0, 0, 0

rng = np.random.default_rng(42)
sample_idx = []
for cluster in [0, 1, 2, 3]:
    cluster_idx = df.index[df["physical_cluster"] == cluster].to_numpy()
    take = rng.choice(cluster_idx, size=min(40, len(cluster_idx)), replace=False)
    sample_idx.extend(take.tolist())

for idx in sample_idx:
    row = df.loc[idx]
    raw = {f: row[f] for f in RAW_FIELDS}
    n_tested += 1
    try:
        result = interpret_house(raw)
    except Exception as e:
        n_exceptions += 1
        print(f"  EXCEPTION on row {idx}: {type(e).__name__}: {e}")
        continue

    if not result.ok:
        is_known_bad_date = (row["yr_renovated"] > 0 and row["yr_renovated"] < row["yr_built"])
        if not is_known_bad_date:
            n_unexpected_reject += 1
            print(f"  UNEXPECTED REJECTION on row {idx}: {result.rejection_reasons}")
        continue

    confidence_levels_seen.add(result.confidence["level"])
    narrative_sources_seen.add(result.narrative_source)

    # --- price sanity ---
    check(f"row {idx}: predicted_price > 0", result.predicted_price > 0, result.predicted_price)
    check(f"row {idx}: cqr_lo <= predicted_price <= cqr_hi",
          result.cqr_lo <= result.predicted_price <= result.cqr_hi,
          (result.cqr_lo, result.predicted_price, result.cqr_hi))
    check(f"row {idx}: cqr_lo > 0", result.cqr_lo > 0, result.cqr_lo)

    # --- property type / category label consistency ---
    full_label = class_tables["tier1_profiles"][result.combined_category]["friendly_category"]
    check(f"row {idx}: property_type_label is a prefix of the full category label",
          full_label.startswith(result.property_type_label + " -- "),
          (result.property_type_label, full_label))

    # --- narrative: no raw internal keys / feature names leaked ---
    for marker in RAW_KEY_LEAK_MARKERS:
        check(f"row {idx}: narrative has no '{marker}' leak", marker not in result.narrative)
    for marker in RAW_FEATURE_NAME_MARKERS:
        check(f"row {idx}: narrative has no raw feature name '{marker}'", marker not in result.narrative)
    check(f"row {idx}: narrative non-empty", len(result.narrative.strip()) > 0)
    check(f"row {idx}: narrative mentions predicted price",
          f"{result.predicted_price:,.0f}" in result.narrative, result.predicted_price)

    # --- SHAP sanity ---
    check(f"row {idx}: shap has top_features", len(result.shap["top_features"]) > 0)

    # --- tier2 comps sanity ---
    check(f"row {idx}: tier2 has n_comps_found >= 0", result.tier2["n_comps_found"] >= 0)

    # --- confidence <-> flags consistency: any Tier B flag with a
    # confidence reason must actually appear in result.confidence["reasons"]
    # when confidence is not well-supported (spot-check the two known ones)
    if result.flags.get("yr_built_after_training_window"):
        check(f"row {idx}: yr_built flag has a matching confidence reason",
              any("built after 2014" in r for r in result.confidence["reasons"]),
              result.confidence["reasons"])
    if result.flags.get("yr_renovated_after_training_window"):
        check(f"row {idx}: yr_renovated flag has a matching confidence reason",
              any("renovation happened after 2014" in r for r in result.confidence["reasons"]),
              result.confidence["reasons"])

print()
print(f"Tested {n_tested} real houses across all 4 clusters.")
print(f"Exceptions: {n_exceptions}   Unexpected rejections: {n_unexpected_reject}")
print(f"Confidence levels observed: {sorted(confidence_levels_seen)}")
print(f"Narrative sources observed: {sorted(narrative_sources_seen)}")
check("all 3 confidence levels represented in sample",
      confidence_levels_seen == {"limited evidence", "typical", "well-supported"},
      confidence_levels_seen)
check("zero exceptions across the sample", n_exceptions == 0)
check("zero unexpected rejections across the sample", n_unexpected_reject == 0)


# =====================================================================
# SECTION 6 -- location fallback path (not-listed city / free-text)
# =====================================================================
print()
print("=" * 78)
print("SECTION 6: location fallback (unseen city/zip)")
print("=" * 78)
h = dict(BASE, city="Nowhereville", statezip="WA 00000")
try:
    result = interpret_house(h)
    check("unseen city/zip still produces a prediction", result.ok, result.rejection_reasons)
    if result.ok:
        check("unseen city/zip sets location_fallback flag",
              result.flags.get("location_fallback") == "global", result.flags)
        check("unseen city/zip sets city_unseen_in_training flag",
              result.flags.get("city_unseen_in_training") is True, result.flags)
except Exception as e:
    check("unseen city/zip does not crash", False, f"{type(e).__name__}: {e}")


# =====================================================================
# SUMMARY
# =====================================================================
print()
print("=" * 78)
print(f"TOTAL CHECKS: {CHECKS}   FAILURES: {len(FAILURES)}")
print("=" * 78)
if FAILURES:
    print("FAILURES:")
    for f in FAILURES:
        print(f"  - {f}")
else:
    print("ALL CHECKS PASSED")

# 2026-09-19 FIX (committee gap-review finding): this script printed a
# FAILURES summary but always exited 0 regardless -- a caller checking the
# exit code alone (e.g. `python3 test_full_app_audit.py && echo PASS`, or
# any future CI wiring) would see success even when checks failed. The
# printed summary was always accurate; only the process exit code was
# wrong. Output above is unchanged either way.
sys.exit(1 if FAILURES else 0)
