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

TEST_YEARS = [1995, 2013, 2014, 2020, 2026, 2027, 2200]

print("=" * 70)
print("TIER A VALIDATION CHECK (validate_tier_a only)")
print("=" * 70)
for yr in TEST_YEARS:
    h = dict(BASE_HOUSE, yr_built=yr)
    reasons = validate_tier_a(h)
    status = "REJECTED" if reasons else "PASSED"
    print(f"yr_built={yr:5d} -> {status}  {reasons}")

print()
print("=" * 70)
print("FULL interpret_house() CHECK")
print("=" * 70)
for yr in TEST_YEARS:
    h = dict(BASE_HOUSE, yr_built=yr)
    try:
        result = interpret_house(h)
    except Exception as e:
        print(f"yr_built={yr:5d} -> EXCEPTION: {type(e).__name__}: {e}")
        continue
    if not result.ok:
        print(f"yr_built={yr:5d} -> REJECTED: {result.rejection_reasons}")
    else:
        print(f"yr_built={yr:5d} -> OK  price=${result.predicted_price:,.0f}  "
              f"interval=[${result.cqr_lo:,.0f}, ${result.cqr_hi:,.0f}]  "
              f"confidence={result.confidence['level']!r}  "
              f"reasons={result.confidence['reasons']}  "
              f"flags={result.flags}")
