"""Re-measure the confidence-tier distribution across the 908 held-out
test houses, per step6_confidence.py's own documented methodology ("run
directly through interpret_house() end-to-end"), so its docstring's
18.9% / 69.8% / 11.2% claim can be checked and corrected if this item 7
change moved it -- this project's own established discipline (see e.g.
SA26's docstring correction in the same file) is to only ever state
MEASURED numbers, never re-derive by hand.

Also reports, separately, how many of the 908 houses were pushed into
"limited evidence" specifically BY item 7's new trigger (as opposed to a
pre-existing trigger) -- the number that matters for judging the size of
this change's effect on what users see.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd

from step6_interpretation import interpret_house
from consolidated_pipeline import group_train_test_split

df = pd.read_csv("consolidated_output/enriched_dataset.csv")
train_df, test_df = group_train_test_split(df)
print(f"held-out test rows: {len(test_df)}")

levels = {"limited evidence": 0, "typical": 0, "well-supported": 0}
rejected = 0
exceptions = 0
new_trigger_only = 0  # limited evidence SOLELY because of the new beyond-range reason
new_trigger_any = 0   # limited evidence WITH the new beyond-range reason present at all

for idx, row in test_df.iterrows():
    h = {
        "bedrooms": row["bedrooms"], "bathrooms": row["bathrooms"], "floors": row["floors"],
        "waterfront": row["waterfront"], "view": row["view"], "condition": row["condition"],
        "sqft_living": row["sqft_living"], "sqft_lot": row["sqft_lot"],
        "sqft_above": row["sqft_above"], "sqft_basement": row["sqft_basement"],
        "yr_built": row["yr_built"], "yr_renovated": row["yr_renovated"],
        "city": row["city"], "statezip": row["statezip"], "country": "USA",
    }
    try:
        result = interpret_house(h)
    except Exception:
        exceptions += 1
        continue
    if not result.ok:
        rejected += 1
        continue
    levels[result.confidence["level"]] += 1
    reasons = result.confidence["reasons"]
    has_new = any("extrapolating beyond anything the model has actually seen" in r for r in reasons)
    if has_new:
        new_trigger_any += 1
        if len(reasons) == 1:
            new_trigger_only += 1

total = sum(levels.values())
print(f"exceptions: {exceptions}   rejected (Tier A): {rejected}   scored: {total}")
print()
for level, count in levels.items():
    pct = 100 * count / total if total else 0
    print(f"  {level:16s} {count:4d}   {pct:5.1f}%")
print()
print(f"houses where the NEW item-7 reason fired at all: {new_trigger_any} ({100*new_trigger_any/total:.1f}%)")
print(f"houses pushed into 'limited evidence' SOLELY by the new reason "
      f"(would otherwise have been typical/well-supported): {new_trigger_only} ({100*new_trigger_only/total:.1f}%)")
print()
print("Existing docstring claim (step6_confidence.py, pre-item-7): "
      "18.9% limited-evidence / 69.8% typical / 11.2% well-supported")
