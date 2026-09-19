import sys
sys.path.insert(0, ".")

from step6_validation import validate_tier_a
from step6_interpretation import interpret_house

BASE_HOUSE = {
    "bedrooms": 3, "bathrooms": 2.0, "floors": 1.0,
    "waterfront": 0, "view": 0, "condition": 3,
    "sqft_living": 1800, "sqft_lot": 7500,
    "sqft_above": 1800, "sqft_basement": 0,
    "yr_built": 2000, "yr_renovated": 0,
    "city": "Bellevue", "statezip": "WA 98004", "country": "USA",
}

# (yr_built, yr_renovated) cases, including the exact gap the subagent found
# ("a 2020 build renovated in 2023 is currently inexpressible").
#
# 2026-09-18 UPDATE (validation-logic audit): `yr_renovated < yr_built` is
# now a Tier A HARD REJECT for live submissions (previously a silent
# soft-correct + `bad_renovation_date_ignored` flag) -- see
# step6_validation.py's own comment on this design change for the full
# reasoning. The two cases below that exercise this ((2020, 2014) and
# (1990, 1980)) are EXPECTED to flip from "PASSED with a flag" to
# "REJECTED" -- that is the intended behavior change this update verifies,
# not a regression.
CASES = [
    (1990, 0),      # not renovated at all -- baseline, must be unaffected
    (1990, 2000),   # ordinary pre-2014 renovation -- must be unaffected
    (1990, 2014),   # renovation right at the training edge
    (1990, 2020),   # pre-2014-built house, renovated after training window
    (1990, 2026),   # same, at the new outer bound
    (1990, 2027),   # must still be rejected (outside [0, 2026])
    (2020, 2014),   # built after 2014, "renovated" before it was built -- nonsense,
                     # now HARD REJECTED (was: silently caught, soft-flagged)
    (2020, 2023),   # THE gap case: post-2014 build, genuine post-2014 renovation -- must still pass
    (2020, 2020),   # renovated same year as built, post-training-window -- must still pass
    (1990, 1980),   # renovated before built -- now HARD REJECTED (was: bad-date rule, soft-flagged)
]

print("=" * 70)
print("TIER A VALIDATION CHECK (validate_tier_a only)")
print("=" * 70)
for yb, yr in CASES:
    h = dict(BASE_HOUSE, yr_built=yb, yr_renovated=yr)
    reasons = validate_tier_a(h)
    status = "REJECTED" if reasons else "PASSED"
    print(f"yr_built={yb:5d} yr_renovated={yr:5d} -> {status}  {reasons}")

print()
print("=" * 70)
print("FULL interpret_house() CHECK")
print("=" * 70)
for yb, yr in CASES:
    h = dict(BASE_HOUSE, yr_built=yb, yr_renovated=yr)
    try:
        result = interpret_house(h)
    except Exception as e:
        print(f"yr_built={yb:5d} yr_renovated={yr:5d} -> EXCEPTION: {type(e).__name__}: {e}")
        continue
    if not result.ok:
        print(f"yr_built={yb:5d} yr_renovated={yr:5d} -> REJECTED: {result.rejection_reasons}")
    else:
        print(f"yr_built={yb:5d} yr_renovated={yr:5d} -> OK  price=${result.predicted_price:,.0f}  "
              f"confidence={result.confidence['level']!r}  "
              f"flags={result.flags}")
        for r in result.confidence["reasons"]:
            print(f"    reason: {r}")
