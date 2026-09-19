"""Dry-run tests for PACKAGING_TODO item 7 (the Tier B "outside training
range" disclosure) -- run BEFORE touching the live device copy, per the
project's standing "test the impact of the change before actually
changing it" instruction.

Covers:
  SECTION 1: exact boundary behavior per field (just inside train range ->
             no flag; just outside -> flag fires, confidence downgrades,
             message names the right field).
  SECTION 2: multiple fields out of range at once -> ONE combined sentence
             naming all of them, not a wall of separate reasons.
  SECTION 3: basement_ratio (the derived, non-raw field, and PACKAGING_TODO
             item 7's own "90% basement ratio" example).
  SECTION 4: a house entirely within every training range -> zero new
             flags, unaffected by this change (regression guard).
  SECTION 5: a full sweep of the real training set (train split, all 3,641
             rows the bounds were computed FROM) -- must show ZERO fires,
             since every bound is that split's own min/max inclusive. Any
             fire here would mean the bounds or the comparison are wrong.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd

from step6_inference_contract import TRAINING_OBSERVED_RANGES, build_feature_row
from step6_interpretation import interpret_house
from consolidated_pipeline import group_train_test_split

BASE_HOUSE = {
    "bedrooms": 3, "bathrooms": 2.0, "floors": 1.0,
    "waterfront": 0, "view": 0, "condition": 3,
    "sqft_living": 1800, "sqft_lot": 7500,
    "sqft_above": 1800, "sqft_basement": 0,
    "yr_built": 2000, "yr_renovated": 0,
    "city": "Bellevue", "statezip": "WA 98004", "country": "USA",
}

failures = []
checks = 0


def check(label, cond, detail=""):
    global checks
    checks += 1
    status = "OK  " if cond else "FAIL"
    print(f"{status}  {label}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(f"{label}  {detail}")


print("=" * 78)
print("SECTION 1: per-field boundary behavior")
print("=" * 78)

# Companion overrides so a field's OWN test value doesn't also trip an
# unrelated Tier A relational check (mirrors test_full_app_audit.py's
# SAFE_COMPANION pattern).
SAFE_COMPANION = {
    "bedrooms": {"sqft_living": 4000},
    "bathrooms": {"sqft_living": 4000},
    "sqft_living": {"bedrooms": 1, "bathrooms": 0.5},
    # sqft_living pinned at ITS OWN training-range upper bound (10040), not
    # beyond it -- large enough to keep Tier A's sqft_above<=sqft_living
    # relational check happy at every test value below, without itself
    # tripping TRAINING_OBSERVED_RANGES and triggering the sqft_above/
    # sqft_living de-dup (step6_inference_contract.py, 2026-09-18 FIX),
    # which would otherwise suppress the very flag this test is isolating.
    "sqft_above": {"sqft_living": 10040},
    "sqft_basement": {"sqft_living": 10001},
}

for field, (lo, hi) in TRAINING_OBSERVED_RANGES.items():
    if field == "basement_ratio":
        continue  # covered separately in SECTION 3
    companion = SAFE_COMPANION.get(field, {})

    # Just inside -> must NOT fire.
    h_in = dict(BASE_HOUSE, **companion)
    h_in[field] = lo
    _, flags_in, _ = build_feature_row(h_in)
    fired_in = field in flags_in.get("beyond_training_range_fields", [])
    check(f"{field} at train LOWER bound ({lo}) -> no flag", not fired_in)

    h_in_hi = dict(BASE_HOUSE, **companion)
    h_in_hi[field] = hi
    _, flags_in_hi, _ = build_feature_row(h_in_hi)
    fired_in_hi = field in flags_in_hi.get("beyond_training_range_fields", [])
    check(f"{field} at train UPPER bound ({hi}) -> no flag", not fired_in_hi)

    # Just below lower bound (if representable, i.e. lo > Tier A's own floor
    # for that field so this isn't ALSO a Tier A concern -- just checking
    # the raw flag function directly, not full validation, so Tier A
    # doesn't even run here).
    if lo > 0:
        h_below = dict(BASE_HOUSE, **companion)
        h_below[field] = lo - (0.01 if isinstance(lo, float) and lo != int(lo) else (0.5 if field == "bathrooms" else (0.1 if field == "floors" else 1)))
        _, flags_below, _ = build_feature_row(h_below)
        fired_below = field in flags_below.get("beyond_training_range_fields", [])
        check(f"{field} just BELOW train lower bound -> flag fires", fired_below,
              f"value={h_below[field]}")

    # Just above upper bound -> must fire.
    h_above = dict(BASE_HOUSE, **companion)
    h_above[field] = hi + (0.5 if field == "bathrooms" else (0.1 if field == "floors" else 1))
    _, flags_above, _ = build_feature_row(h_above)
    fired_above = field in flags_above.get("beyond_training_range_fields", [])
    check(f"{field} just ABOVE train upper bound -> flag fires", fired_above,
          f"value={h_above[field]}")

print()
print("=" * 78)
print("SECTION 2: multiple out-of-range fields combine into ONE sentence")
print("=" * 78)
h_multi = dict(BASE_HOUSE, bedrooms=15, sqft_lot=2_000_000, sqft_living=4000, sqft_above=4000, sqft_basement=0)
result = interpret_house(h_multi)
check("multi-field submission is accepted (Tier A still passes)", result.ok,
      str(getattr(result, "rejection_reasons", None)))
if result.ok:
    reasons_text = " ".join(result.confidence["reasons"])
    check("combined reason mentions 'bedrooms'", "bedroom" in reasons_text.lower())
    check("combined reason mentions 'lot'", "lot size" in reasons_text.lower() or "lot" in reasons_text.lower())
    # Exactly one reason for this whole family (not one per field).
    beyond_range_reasons = [r for r in result.confidence["reasons"] if "training data, so we're extrapolating" in r]
    check("exactly ONE combined beyond-range reason (not one per field)", len(beyond_range_reasons) == 1,
          str(result.confidence["reasons"]))
    check("confidence downgraded to 'limited evidence'", result.confidence["level"] == "limited evidence")
    print(f"    reason text: {beyond_range_reasons}")

print()
print("=" * 78)
print("SECTION 2b: sqft_above is suppressed as redundant when sqft_living")
print("also fires (adversarial subagent review finding, 2026-09-18 fix)")
print("=" * 78)
# sqft_living AND sqft_above both beyond training range, sqft_above == sqft_living
# (no basement) -- sqft_above must be suppressed as a duplicate of sqft_living.
h_dup = dict(BASE_HOUSE, sqft_living=15000, sqft_above=15000, sqft_basement=0)
_, flags_dup, _ = build_feature_row(h_dup)
fired_dup = flags_dup.get("beyond_training_range_fields", [])
check("sqft_living fires", "sqft_living" in fired_dup, str(fired_dup))
check("sqft_above is suppressed (redundant with sqft_living)", "sqft_above" not in fired_dup, str(fired_dup))

# sqft_above beyond range WITHOUT sqft_living also being beyond range (a big
# basement pulls sqft_above down within range while sqft_living stays put --
# not really constructible since sqft_above<=sqft_living always, so instead
# verify sqft_above alone (sqft_living in range, sqft_above beyond ITS OWN
# range but under sqft_living) still fires on its own.
h_above_alone = dict(BASE_HOUSE, sqft_living=10040, sqft_above=7681, sqft_basement=2359)
_, flags_above_alone, _ = build_feature_row(h_above_alone)
fired_alone = flags_above_alone.get("beyond_training_range_fields", [])
check("sqft_above alone (sqft_living in range) still fires on its own",
      "sqft_above" in fired_alone, str(fired_alone))
check("sqft_living does NOT fire in this case (it's in range)",
      "sqft_living" not in fired_alone, str(fired_alone))

print()
print("=" * 78)
print("SECTION 3: basement_ratio (derived field, item 7's own example)")
print("=" * 78)
# 90% basement ratio, matching PACKAGING_TODO item 7's own example --
# sqft_basement=1800, sqft_living=2000 -> basement_ratio=0.9, well past the
# train max of ~0.595. Needs sqft_above = sqft_living - sqft_basement = 200
# to keep the GAP1/GAP2 relational checks happy.
h_ratio = dict(BASE_HOUSE, sqft_living=2000, sqft_basement=1800, sqft_above=200)
_, flags_ratio, _ = build_feature_row(h_ratio)
check("90% basement ratio -> basement_ratio flag fires",
      "basement_ratio" in flags_ratio.get("beyond_training_range_fields", []))
result_ratio = interpret_house(h_ratio)
check("90% basement ratio house is still accepted (soft flag, not a reject)", result_ratio.ok)
if result_ratio.ok:
    check("90% basement ratio -> confidence is 'limited evidence'",
          result_ratio.confidence["level"] == "limited evidence")
    check("90% basement ratio -> reason mentions 'basement'",
          any("basement" in r.lower() for r in result_ratio.confidence["reasons"]))

# Just inside the train max ratio -> must NOT fire.
br_lo, br_hi = TRAINING_OBSERVED_RANGES["basement_ratio"]
sqft_living_ok = 2000
sqft_basement_ok = int(sqft_living_ok * br_hi)  # just at/under the train max ratio
h_ratio_ok = dict(BASE_HOUSE, sqft_living=sqft_living_ok, sqft_basement=sqft_basement_ok,
                   sqft_above=sqft_living_ok - sqft_basement_ok)
_, flags_ratio_ok, _ = build_feature_row(h_ratio_ok)
check(f"basement_ratio at/under train max ({br_hi:.4f}) -> no flag",
      "basement_ratio" not in flags_ratio_ok.get("beyond_training_range_fields", []),
      f"ratio={sqft_basement_ok/sqft_living_ok:.4f}")

print()
print("=" * 78)
print("SECTION 4: an entirely ordinary house -> zero new flags (regression guard)")
print("=" * 78)
result_base = interpret_house(BASE_HOUSE)
check("BASE_HOUSE is accepted", result_base.ok)
if result_base.ok:
    check("BASE_HOUSE has no beyond_training_range_fields flag",
          "beyond_training_range_fields" not in result_base.flags)
    check("BASE_HOUSE confidence reasons contain no beyond-range caveat",
          not any("extrapolating" in r for r in result_base.confidence["reasons"]))

print()
print("=" * 78)
print("SECTION 5: full training-split sweep -- must show ZERO false positives")
print("=" * 78)
# The bounds were computed as this split's own min/max -- every one of
# these 3,641 real rows must therefore land INSIDE every bound, by
# construction. A fire here would mean the bounds/comparison are wrong,
# not that these houses are "unusual" -- they're literally what defined
# "usual" for this check.
df = pd.read_csv("consolidated_output/enriched_dataset.csv")
train_df, test_df = group_train_test_split(df)

false_positive_rows = []
for idx, row in train_df.iterrows():
    h = {
        "bedrooms": row["bedrooms"], "bathrooms": row["bathrooms"], "floors": row["floors"],
        "waterfront": row["waterfront"], "view": row["view"], "condition": row["condition"],
        "sqft_living": row["sqft_living"], "sqft_lot": row["sqft_lot"],
        "sqft_above": row["sqft_above"], "sqft_basement": row["sqft_basement"],
        "yr_built": row["yr_built"], "yr_renovated": row["yr_renovated"],
        "city": row["city"], "statezip": row["statezip"], "country": "USA",
    }
    _, flags, _ = build_feature_row(h)
    if flags.get("beyond_training_range_fields"):
        false_positive_rows.append((idx, flags["beyond_training_range_fields"]))

check(f"zero false positives across all {len(train_df)} training-split rows",
      len(false_positive_rows) == 0, f"{len(false_positive_rows)} rows flagged: {false_positive_rows[:5]}")

print()
print("=" * 78)
print(f"TOTAL CHECKS: {checks}   FAILURES: {len(failures)}")
print("=" * 78)
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
else:
    print("ALL CHECKS PASSED")

# 2026-09-19 FIX (committee gap-review finding): this script printed a
# FAILURES summary but always exited 0 regardless -- a caller checking the
# exit code alone (e.g. `python3 test_beyond_training_range.py && echo
# PASS`, or any future CI wiring) would see success even when checks
# failed. The printed summary was always accurate; only the process exit
# code was wrong. Output above is unchanged either way.
sys.exit(1 if failures else 0)
