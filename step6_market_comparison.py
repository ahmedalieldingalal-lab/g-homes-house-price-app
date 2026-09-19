"""Phase A, steps 8 (part 1) -- Tier 2 comparable-sales retrieval and the
point_position / interval_vs_comps / market_divergence market-comparison
statements (Decision 3/3b/4, and the critical market-verdict-degeneracy
fix logged under "THE critical finding (round 3)" in PROJECT_TODO.md).

Tier 2 retrieval (Decision 4): structured pandas filtering, no embeddings.
Fallback ladder widens GEOGRAPHY, not the similarity window, to reach the
target comp count: same zip -> same city -> same Tier-1 combined_category
-> dataset-wide, each filtered to sqft_living within +/-20% of the
subject house (a standard real-estate "comp" window; not previously
pinned to an exact number in the design log, so chosen here and measured
against real data in the accompanying test rather than assumed). If even
the dataset-wide + sqft-window tier can't reach the floor of 5,
`sklearn.neighbors.NearestNeighbors` over standardized [sqft_living,
bedrooms, bathrooms, house_age] (whole dataset, no sqft window) supplies
the last-resort closest-match set -- this is what makes "insufficient"
evidence genuinely rare rather than common, which is the entire point of
building a ladder instead of a single same-zip-or-nothing lookup.

`comp_evidence` (insufficient/limited/strong) is computed from the final
comp count actually reached (whichever tier that took), while
`tier2_tier_used` records WHICH tier supplied it -- so a later confidence
computation (Phase A step 9, not yet built) can penalize "reached the
NearestNeighbors last resort" even when the raw count looks "strong".

Market-comparison statements:
- `point_position`: where the PREDICTED PRICE sits among the Tier-2 comps'
  price percentiles (below <p10, lower p10-p25, central p25-p75, upper
  p75-p90, above >p90) -- the PRIMARY, informative market statement (see
  "THE critical finding" -- the previously-proposed interval-vs-comps rule
  was measured degenerate, returning "in line" for 100% of real houses).
- `interval_vs_comps`: whether the model's own 90% CQR interval overlaps
  the comps' interquartile range (overlapping/strictly_above/
  strictly_below). AUDIT FIX 2026-09-09 (SA5, verification pass): this
  used to be described here as "a fixed, always-rendered HONESTY CAVEAT
  about the model's uncertainty being wide" -- that was never accurate.
  Both narrative paths (`step6_narrative.py`, `step6_llm_narrative.py`)
  gate this sentence on `interval_vs_comps == "overlapping"`, so it is
  conditional, not always-rendered (confirmed: 2 of 908 held-out houses
  render no such sentence at all, `strictly_above`/`strictly_below`
  cases). It is also no longer framed around "uncertainty being wide" --
  the SA5 fix already reworded the sentence itself to state only that the
  two ranges intersect, since "overlapping" says nothing about which
  range is wider (4 of 908 houses had a CQR interval actually NARROWER
  than the comps' spread while the old wording still claimed "wider
  than"). This entry now matches both the gating condition and the
  current wording instead of describing neither correctly.
- `tier1_position` / `market_divergence`: whether the predicted price's
  standing relative to its own Tier-1 category's mean (below/central/
  above, +/-15% band) agrees with `point_position`'s coarse bucket --
  Decision 6b's "show both lenses, flagged, when they disagree".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# AUDIT FIX 2026-09-09 (SB4, final audit): anchored to this file's own
# location instead of the process cwd -- see step6_classify_house.py's
# identical fix for the full rationale.
REPO_ROOT = Path(__file__).resolve().parent
ENRICHED_DATASET_PATH = REPO_ROOT / "consolidated_output" / "enriched_dataset.csv"

SQFT_SIMILARITY_PCT = 0.20
TARGET_COMPS = 10
FLOOR_COMPS = 5
DISPLAY_CAP = 15
TIER1_DIVERGENCE_BAND = 0.15  # +/-15% of the category mean price

# AUDIT FIX 2026-09-09 (SA20, final audit): was a single `pd.DataFrame |
# None` slot, so a first call with the default path cached under that
# path forever, and a LATER call with a *different* `path` silently
# returned the first path's already-cached DataFrame instead of loading
# the new one -- confirmed live: `load_enriched_dataset()` then
# `load_enriched_dataset(Path('sample_data_SYNTHETIC_FOR_TESTING_ONLY.csv'))`
# returned the SAME object (`a is b` -> True), not the sample file.
# Harmless in production (the app only ever calls this with the default
# path), but a real footgun for tests wanting an isolated fixture path.
# Keyed by the resolved path string instead, so distinct paths get their
# own cache slot while repeated calls with the SAME path still hit cache.
_ENRICHED_DF_CACHE: dict[str, pd.DataFrame] = {}


def load_enriched_dataset(path: Path = ENRICHED_DATASET_PATH) -> pd.DataFrame:
    key = str(Path(path).resolve())
    if key not in _ENRICHED_DF_CACHE:
        _ENRICHED_DF_CACHE[key] = pd.read_csv(path)
    return _ENRICHED_DF_CACHE[key]


def _sqft_filter(df: pd.DataFrame, sqft_living: float) -> pd.DataFrame:
    lo, hi = sqft_living * (1 - SQFT_SIMILARITY_PCT), sqft_living * (1 + SQFT_SIMILARITY_PCT)
    return df[(df["sqft_living"] >= lo) & (df["sqft_living"] <= hi)]


def get_tier2_comps(engineered: dict, combined_category: str, df: pd.DataFrame | None = None,
                     target: int = TARGET_COMPS, floor: int = FLOOR_COMPS, cap: int = DISPLAY_CAP) -> dict:
    df = df if df is not None else load_enriched_dataset()
    sqft = engineered["sqft_living"]

    ladder = [
        ("zip", df[df["statezip"] == engineered["statezip"]]),
        ("city", df[df["city"] == engineered["city"]]),
        ("category", df[df["combined_category"] == combined_category]),
        ("dataset_wide", df),
    ]

    chosen_tier, comps = None, pd.DataFrame()
    for tier_name, subset in ladder:
        filtered = _sqft_filter(subset, sqft)
        if len(filtered) >= target:
            chosen_tier, comps = tier_name, filtered
            break
        if len(filtered) > len(comps):
            chosen_tier, comps = tier_name, filtered  # keep the best-so-far in case nothing reaches target

    flags: dict = {}
    if chosen_tier != "zip":
        flags["tier2_location_fallback"] = chosen_tier

    if len(comps) < floor:
        # Last resort: nearest neighbors over standardized numeric fields,
        # whole dataset, NO sqft window -- a sharper closest-match set than
        # "give up", per Decision 4.
        feature_cols = ["sqft_living", "bedrooms", "bathrooms", "house_age"]
        pool = df.dropna(subset=feature_cols).reset_index(drop=True)
        means, stds = pool[feature_cols].mean(), pool[feature_cols].std().replace(0, 1)
        pool_std = (pool[feature_cols] - means) / stds
        query = pd.DataFrame([{c: engineered[c] for c in feature_cols}])
        query_std = (query - means) / stds
        n_neighbors = min(target, len(pool))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(pool_std.values)
        _, idx = nn.kneighbors(query_std.values)
        comps = pool.iloc[idx[0]]
        chosen_tier = "nearest_neighbors"
        flags["tier2_location_fallback"] = "nearest_neighbors"

    n_found = len(comps)
    if n_found >= target:
        comp_evidence = "strong"
    elif n_found >= floor:
        comp_evidence = "limited"
    else:
        comp_evidence = "insufficient"

    comps_display = comps.sort_values(
        by="sqft_living", key=lambda s: (s - sqft).abs(),
    ).head(cap)
    comp_prices = comps["price"].values  # stats use ALL retrieved comps, not just the display-capped subset

    return {
        "tier2_tier_used": chosen_tier,
        "n_comps_found": int(n_found),
        "comp_evidence": comp_evidence,
        "comp_prices": comp_prices,
        "comps_display": comps_display[["price", "sqft_living", "bedrooms", "bathrooms", "city", "statezip"]].to_dict("records"),
        "flags": flags,
    }


def _percentile_bucket(value: float, series: np.ndarray) -> str:
    # AUDIT FIX 2026-09-09 (SA31f, final audit): the strict/non-strict
    # bounds below are asymmetric (`<p10`, `<p25`, `<=p75`, `<=p90`) --
    # flagged as an inconsistency, but it is cosmetic, not a bug: this is
    # a sequential if-ladder, not independent range checks, so every real
    # number lands in exactly one bucket regardless of which comparisons
    # are strict (a value exactly at p10 falls through to "lower", not
    # "below"; a value exactly at p75 or p90 is included in "central"/
    # "upper" rather than the bucket above it) -- there is no gap and no
    # double-count either way. Documented as an intentional (if arbitrary)
    # tie-breaking choice rather than "fixed" into artificial symmetry
    # that would change which bucket boundary values land in for no
    # behavioral benefit.
    p10, p25, p75, p90 = np.percentile(series, [10, 25, 75, 90])
    if value < p10:
        return "below"
    if value < p25:
        return "lower"
    if value <= p75:
        return "central"
    if value <= p90:
        return "upper"
    return "above"


def _coarsen(position: str) -> str:
    return {"below": "below", "lower": "below", "central": "central", "upper": "above", "above": "above"}[position]


# AUDIT FIX 2026-09-09 (SA12, final audit): these two display-phrase maps
# used to exist ONLY in step6_narrative.py -- the LLM-assisted narrative
# (step6_llm_narrative.py's assemble_llm_narrative()) printed the raw
# `tier2_tier_used`/`point_position` enum values instead ("...reads as
# 'upper' vs. the local market", with no comp-tier phrase at all), so the
# two narrative "sources" disagreed on how to say the same fact depending
# purely on whether Gemini answered. Moved here -- next to the functions
# that actually produce `tier2_tier_used`/`point_position` -- so both
# narrative paths share one translation instead of the LLM path silently
# leaking a raw enum (rule 11).
TIER2_TIER_PHRASES = {
    "zip": "in the same zip code", "city": "in the same city",
    "category": "in the same market category", "dataset_wide": "across the full dataset",
    "nearest_neighbors": "closest-matching, based on size and features",
}
POINT_POSITION_PHRASES = {
    "below": "well below", "lower": "somewhat below", "central": "right in the typical range for",
    "upper": "somewhat above", "above": "well above", "unknown": "not directly comparable to",
}


def tier2_tier_phrase(tier2_tier_used: str) -> str:
    return TIER2_TIER_PHRASES.get(tier2_tier_used, "")


def point_position_phrase(point_position: str) -> str:
    return POINT_POSITION_PHRASES[point_position]


def compute_market_comparison(predicted_price: float, cqr_lo: float, cqr_hi: float,
                               tier2: dict, tier1_profile: dict) -> dict:
    comp_prices = tier2["comp_prices"]
    if len(comp_prices) < 2:
        # Degenerate case (should be rare given the NN fallback, but a
        # NearestNeighbors pool smaller than 2 is theoretically possible on
        # a tiny dataset) -- no percentile statement can be made honestly.
        point_position = "unknown"
        interval_vs_comps = "unknown"
    else:
        point_position = _percentile_bucket(predicted_price, comp_prices)
        p25, p75 = np.percentile(comp_prices, [25, 75])
        if cqr_lo > p75:
            interval_vs_comps = "strictly_above"
        elif cqr_hi < p25:
            interval_vs_comps = "strictly_below"
        else:
            interval_vs_comps = "overlapping"

    mean_price = tier1_profile["mean_price"]
    ratio = predicted_price / mean_price if mean_price else 1.0
    if ratio < 1 - TIER1_DIVERGENCE_BAND:
        tier1_position = "below"
    elif ratio > 1 + TIER1_DIVERGENCE_BAND:
        tier1_position = "above"
    else:
        tier1_position = "central"

    if point_position == "unknown":
        market_divergence = "unknown"
    else:
        market_divergence = "aligned" if _coarsen(point_position) == tier1_position else "diverges"

    return {
        "point_position": point_position,
        "interval_vs_comps": interval_vs_comps,
        "tier1_position": tier1_position,
        "market_divergence": market_divergence,
    }
