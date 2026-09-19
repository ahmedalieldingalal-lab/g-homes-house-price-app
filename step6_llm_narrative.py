"""Phase B (steps 13-17) + Phase C (steps 18-19) -- the Gemini prompt/
response contract that PREPENDS/interleaves LLM connective prose around
the same code-owned facts the template narrative (`step6_narrative.py`)
already renders. The template path is NOT replaced -- per Decision 2's
"deterministic spine + LLM connective tissue" architecture, this module
IS the spine (all facts, all numbers) plus LLM prose slotted in at fixed
points; on ANY failure (no API key, timeout, malformed response, a
disallowed number, an ungrounded feature name, an inappropriate certainty
claim), the caller falls back to the plain template untouched.

Step 13 -- number-free prompt invariant: the LLM receives ONLY word-level
facts (feature display names, direction/magnitude words, verdict/
confidence enums, fired-flag labels) -- never a numeral, percent, or
dollar amount, and never the raw `combined_category` string (which
contains digits, e.g. "Physical Group 0 - Price Band 4") -- see
`digit_free_category_label()` below, which looks up that same category's
canonical, already-digit-free `friendly_category` string (AUDIT FIX
2026-09-09, SA11, final audit -- this used to invent its own, disagreeing
naming scheme instead of using the one already shipped for this exact
purpose) so a downstream regex check for "any digit in the LLM's OWN
output" is sound AND complete, not just heuristic: if the model never
received a
digit, any digit in its response is definitionally invented.

Step 14 -- two fixed system instructions (confident/hedged), selected by
`confidence["level"]`.

Step 15 -- one pydantic model, used as both Gemini's `response_schema`
and the inbound validator. No `caveats`/`confidence`/ordering fields --
those stay code-owned.

Step 16 -- generation config: JSON mime type, minimal "thinking" effort
(Gemini has thinking on by default and can return `finish_reason=
MAX_TOKENS` with EMPTY text if the thinking budget eats the whole output
allowance -- a real, specific failure mode, not a hypothetical one), a
generous `max_output_tokens`, one pinned temperature. UI_AMENDMENTS.md
entry 13: originally `thinking_budget=0` (the Gemini 2.5-era numeric
field) -- after moving to `gemini-3.6-flash` (entry 4), that legacy field
started 400'ing outright, since Gemini 3.x replaced it with a
`thinking_level` enum (`thinking_budget` and `thinking_level` cannot both
be sent, and the SDK/backend rejects the legacy field on 3.x models with
a generic "invalid argument" 400). Switched to
`thinking_level="minimal"`, the 3.x equivalent of "as little thinking as
possible."

Step 17 -- `city`/`statezip` are NEVER interpolated into the prompt as
free text; only word-level facts derived from them (e.g. which location
fallback tier was used) are ever sent.

Step 18 -- validation, in order, abort-to-spine-only on first failure:
finish_reason == STOP and non-empty -> JSON parses + pydantic-valid ->
digit/number-word regex reject on all 4 fields -> raw-feature-name leak
check (AUDIT FIX 2026-09-09, SA17, final audit -- this was previously
mis-described here as a "vocabulary check" that verified every
capitalized feature-like token against the supplied display names; no
such check exists or ever existed. What `_contains_ungrounded_feature_
reference()` actually does is substring-match the response against the
raw, underscore-containing engineered-feature names (e.g. 'sqft_living',
'condition_x_age') to catch one specific, concrete failure mode -- a raw
column name leaking into the LLM's prose, which would only happen from a
stale cached response or similar drift. It deliberately does NOT attempt
to detect a fabricated new display name, since display names are free-
form natural-language phrases, not a closed vocabulary to check
membership against.) -> certainty-lexicon ban when confidence !=
"well-supported".
"""
from __future__ import annotations

import json
import math
import os
import threading
import re
from typing import Optional

from pydantic import BaseModel, Field

# =============================================================================
# Digit-free display layer (step 13's grounding for combined_category)
# =============================================================================
def digit_free_category_label(combined_category: str, class_tables: dict) -> str:
    """Returns the canonical, human-readable `friendly_category` string for
    this `combined_category` (e.g. 'Physical Group 0 - Price Band 4' ->
    'Untouched Classics -- High-End Tier') -- already digit-free, so this
    is a lookup, not a transformation.

    AUDIT FIX 2026-09-09 (SA11, final audit): this used to INVENT a fourth
    naming scheme for these same 18 categories ("Group A, Premium tier",
    built from a lettered group + a hand-maintained tier-word ramp via the
    now-removed `_band_tier_word`/`GROUP_LETTERS`/`_TIER_WORDS_BY_K`) that
    disagreed with the project's own canonical `friendly_category` column
    for 9 of the 18 real categories -- most seriously, Group 0's top two
    price bands were shifted one tier word too high ("Premium"/"Luxury"
    here vs. the real "High-End"/"Premium"). `friendly_category` was
    already digit-free (confirmed for all 18 real categories) and was
    already the label shown everywhere else in this project (notebooks,
    Word doc, `consolidated_pipeline.py`) -- it needed to be looked up
    here, not reinvented. `step6_build_classification_lookups.py` now
    carries it on every `tier1_profiles` entry for exactly this lookup.
    This also makes SA4's bounds-clamping fix to the old tier-word ramp
    moot (that code is deleted along with the scheme it clamped), since
    a dict lookup by a real `combined_category` key can't index out of
    range the way `words[band]` could."""
    return class_tables["tier1_profiles"][combined_category]["friendly_category"]


# 2026-09-18 ADDITION (Item 5 -- independent "Property Type" badge): the
# physical-cluster HALF of `digit_free_category_label()`'s already-digit-
# free `friendly_category` string, without its price-tier suffix (e.g.
# 'Untouched Classics -- High-End Tier' -> 'Untouched Classics'). This is
# a pure string operation on the ONE canonical `friendly_category` value
# above -- never a second, independently-maintained cluster-name mapping
# (rule 11) -- so a future rename of the 4 cluster names only ever has to
# touch `classification_tables.json` and every surface that reads it,
# this one included, stays in sync automatically.
#
# Verified 2026-09-18: all 18 real `friendly_category` values use the
# same ' -- ' separator between property type and price tier, and every
# price band within the same physical cluster shares the identical
# property-type half (e.g. all 6 of Physical Group 0's bands ->
# 'Untouched Classics'), so this is a safe, order-independent split for
# every category the model can ever return -- not a heuristic that could
# disagree with the badge shown elsewhere for the same house.
def digit_free_property_type_label(combined_category: str, class_tables: dict) -> str:
    """Returns just the property-type name (e.g. 'Untouched Classics'),
    dropping the ' -- <Tier>' price-tier suffix that
    `digit_free_category_label()` includes. Intended for a standalone
    "Property Type" badge shown next to the price estimate, where the
    price tier is already conveyed separately by the Low/Estimate/High
    range -- repeating it in this badge too would be redundant."""
    return digit_free_category_label(combined_category, class_tables).split(" -- ", 1)[0]


# =============================================================================
# Feature display names (35 engineered features -> human-readable labels).
#
# AUDIT FIX 2026-09-09 (SA17, verification pass): this used to describe
# the dict below as Phase C step 18's "vocabulary allowlist: any
# feature-like name the LLM's response mentions that ISN'T a value in
# this dict is grounds for rejection" -- SA18's fix (see
# `_contains_ungrounded_feature_reference`'s own docstring below) already
# established that no such allowlist check exists or ever did: that
# function only flags a RAW, underscore-containing feature KEY (e.g.
# 'sqft_living') appearing verbatim, and explicitly does NOT try to detect
# a fabricated display name that isn't one of this dict's VALUES -- so a
# hallucinated but plausible-sounding feature name would pass. This
# comment was the sibling copy SA18's fix missed (same rule-11 class of
# miss as SA1/SA4); corrected here to describe what the code actually
# does, not what the module docstring higher up in this file once claimed.
# =============================================================================
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "bedrooms": "number of bedrooms", "bathrooms": "number of bathrooms", "floors": "number of floors",
    "waterfront": "waterfront access", "view": "view quality", "condition": "overall condition",
    "sqft_living": "living area size", "sqft_lot": "lot size", "sqft_above": "above-ground living area",
    "sqft_basement": "basement area", "yr_built": "year built", "yr_renovated": "renovation year",
    "house_age": "age of the house", "was_renovated": "whether it was renovated",
    "total_rooms": "total room count", "years_since_renovation": "time since renovation",
    "renovation_recency_ratio": "how recently it was renovated", "has_basement": "presence of a basement",
    "basement_ratio": "basement proportion of living area", "lot_utilization": "how much of the lot is built on",
    "bath_bed_ratio": "bathroom-to-bedroom ratio", "avg_room_size": "average room size",
    "premium_outlook_score": "premium outlook (view/waterfront)", "condition_x_age": "condition relative to age",
    "log_sqft_lot": "lot size", "sale_month": "time of year sold", "sale_month_sin": "seasonal timing",
    "sale_month_cos": "seasonal timing", "city_sale_volume_safe": "how active the city's market is",
    "zip_sale_volume_safe": "how active the local market is", "location_price_per_sqft": "location/neighborhood value",
    "age_bucket_100y+": "age category (very old)", "age_bucket_11-25y": "age category (young-to-mid-age)",
    "age_bucket_26-50y": "age category (mid-age)", "age_bucket_51-100y": "age category (older)",
}


def feature_label(feature: str) -> str:
    """The one place raw engineered feature names (e.g. 'location_price_
    per_sqft', 'city_sale_volume_safe', 'bath_bed_ratio') get translated
    to plain English before reaching a customer. UI_AMENDMENTS.md entry
    14: `build_prompt_facts()` below already did this translation for the
    facts fed TO Gemini, but the CODE-RENDERED driver lines in both
    `assemble_llm_narrative()` (this file) and `build_template_narrative()`
    (step6_narrative.py) were printing `f['feature']` raw and unchanged --
    found by the user pasting a narrative reading "driven mainly by:
    location_price_per_sqft (-39.6%), sqft_living (-6.9%), ...". Both call
    sites now go through this one shared helper instead of duplicating the
    `FEATURE_DISPLAY_NAMES.get(...)` lookup (rule 11)."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def group_top_features_by_label(top_features: list[dict]) -> list[dict]:
    """Several distinct engineered features can share the same human-
    readable display name (e.g. `sqft_lot` and `log_sqft_lot` both show as
    "lot size"; `sale_month_sin`/`sale_month_cos` both show as "seasonal
    timing"). Merges those onto one entry using the SAME log-space-
    additive math that makes the multiplicative identity exact elsewhere
    in this project (combined pct effect = exp(sum of the group's
    shap_log values) - 1, per step6_shap_contract.py's own derivation) --
    not an approximation, the exact combined effect of both features
    together.

    AUDIT FIX 2026-09-09 (SA1, final audit): this merge logic used to
    exist ONLY in step6_plots.py (`_group_top_features_by_label`,
    originally written there after "actually looking at a rendered chart,
    where it appeared as two same-labeled bars with overlapping text")
    and was never applied to either narrative path -- so the chart merged
    "lot size" into one bar while the narrative text directly above it,
    built independently by iterating `top_features` raw, still listed
    "lot size" twice with two different percentages (confirmed live on
    30% of a 200-house sample, and independently re-confirmed at 31.7% on
    a second, differently-seeded sample). Moved here -- the canonical
    home for feature-display-name logic, alongside FEATURE_DISPLAY_NAMES
    and feature_label() -- so step6_plots.py, step6_narrative.py, and
    this file's own assemble_llm_narrative()/build_prompt_facts() all
    share one implementation instead of the chart being the only correct
    one (rule 11)."""
    groups: dict[str, dict] = {}
    for f in top_features:
        label = feature_label(f["feature"])
        g = groups.setdefault(label, {"shap_log_sum": 0.0, "source_features": []})
        g["shap_log_sum"] += f["shap_log"]
        g["source_features"].append(f["feature"])

    grouped = [
        {
            "label": label if len(g["source_features"]) == 1 else f"{label} ({len(g['source_features'])} combined factors)",
            "pct_effect": math.exp(g["shap_log_sum"]) - 1.0,
            "abs_shap_log": abs(g["shap_log_sum"]),
            "source_features": g["source_features"],
        }
        for label, g in groups.items()
    ]
    grouped.sort(key=lambda g: -g["abs_shap_log"])
    return grouped


def _magnitude_word(pct_effect: float) -> str:
    a = abs(pct_effect)
    if a < 0.02:
        return "slightly"
    if a < 0.05:
        return "modestly"
    if a < 0.15:
        return "noticeably"
    if a < 0.30:
        return "substantially"
    return "dramatically"


def _count_word(n: int) -> str:
    if n <= 0:
        return "no"
    if n <= 4:
        return "a few"
    if n <= 15:
        return "several"
    return "many"


FLAG_LABELS = {
    "beyond_training_age_range": "unusually old for this model's training data",
    "bad_renovation_date_ignored": "an inconsistent renovation date was ignored",
    "age_bucket_fallback_used": "age category could not be determined precisely",
    "city_unseen_in_training": "this city was not seen during training",
    "zip_unseen_in_training": "this zip code was not seen during training",
    "city_unseen_for_classification": "this city was not seen during training",
    "price_band_clipped": "the price category required a small correction",
    "tier1_snapped_to_nearest_band": "the price category required a small correction",
}

# AUDIT FIX 2026-09-09 (SA24, final audit): `tier2_location_fallback`
# used to have ONE fixed label in `FLAG_LABELS` above ("comparable sales
# came from a broader area than the immediate neighborhood"), but the
# flag's VALUE is one of four different fallback tiers
# (`step6_market_comparison.get_tier2_comps`), and two of them are not
# geographic at all: "category" (a price/type bucket, not an area) and
# "nearest_neighbors" (whole-dataset feature similarity, not location).
# A 908-house run showed 61 houses (6.7%) received a geographic caveat
# for a non-geographic fallback. Now keyed by the actual value so each
# tier gets an accurate sentence. The "nearest_neighbors" wording matches
# `step6_confidence.py`'s identical-fact trigger verbatim (rule 12 --
# same fact, same words, wherever it's surfaced).
TIER2_LOCATION_FALLBACK_LABELS = {
    "city": "comparable sales came from the broader city rather than the immediate neighborhood",
    "category": "comparable sales came from the same market category rather than the immediate neighborhood",
    "dataset_wide": "comparable sales came from across the full dataset rather than the immediate neighborhood",
    "nearest_neighbors": "the comparable sales we found weren't close matches by location, so we used the closest homes by size and features instead",
}


def compute_fired_flag_labels(flags: dict) -> list[str]:
    """The one shared implementation of "which flags produce a
    user-facing caveat sentence" -- `build_prompt_facts()` below and
    `app.py`'s degradation-reasons display used to each hand-roll their
    own copy of this comprehension (rule 11). AUDIT FIX 2026-09-09
    (SA24, final audit): the old comprehension was `FLAG_LABELS[k] for k
    in flags if k in FLAG_LABELS` -- checking KEY PRESENCE only, blind to
    the value, so a flag that was ever explicitly set to a falsy value
    (e.g. `False`) would still render as fired. Now checks truthiness
    too, and looks up `tier2_location_fallback` by its VALUE (one of
    four distinct fallback tiers) rather than a single one-size-fits-all
    label."""
    labels = set()
    for key, value in flags.items():
        if not value:
            continue
        if key == "tier2_location_fallback":
            label = TIER2_LOCATION_FALLBACK_LABELS.get(value)
        else:
            label = FLAG_LABELS.get(key)
        if label:
            labels.add(label)
    return sorted(labels)


def build_prompt_facts(interp) -> dict:
    """Every value here is a WORD, an ENUM, or a COUNT-WORD -- never a
    numeral. `interp` is an `InterpretationResult` (step6_interpretation)."""
    # AUDIT FIX 2026-09-09 (SA1, final audit): this used to build one entry
    # per raw feature, so two features sharing a display name (e.g.
    # sqft_lot/log_sqft_lot -> "lot size") sent Gemini two "top_drivers"
    # entries with the SAME display_name and two different magnitude
    # words -- confirmed on 27.3% of a 150-house sample -- which the model
    # then had no way to know was one merged factor, not two independent
    # ones. Now grouped the same way the chart and the no-AI narrative are.
    top_drivers = [
        {
            # AUDIT FIX 2026-09-09 (SA1 fix regression caught by this
            # project's own number-free-prompt test, tests/test_step6_
            # llm_narrative.py): `g["label"]` can read e.g. "lot size (2
            # combined factors)" -- that literal digit must never reach a
            # Gemini-facing field (the number-free-prompt invariant this
            # module exists to enforce). Gemini doesn't need the merge
            # count, only the display name, so strip the parenthetical
            # rather than converting it to a count word.
            "display_name": g["label"].split(" (")[0],
            "direction": "increases" if g["pct_effect"] > 0 else "decreases",
            "magnitude": _magnitude_word(g["pct_effect"]),
        }
        for g in group_top_features_by_label(interp.shap["top_features"])
    ]
    from step6_classify_house import load_classification_tables
    class_tables = load_classification_tables()
    category_label = digit_free_category_label(interp.combined_category, class_tables)

    fired_flag_labels = compute_fired_flag_labels(interp.flags)

    return {
        "top_drivers": top_drivers,
        "num_other_factors": _count_word(interp.shap["catchall_n_features"]),
        # AUDIT FIX 2026-09-09 (SA31d, final audit): this used to compute a
        # direction word unconditionally, even when `catchall_n_features
        # == 0` (i.e. `num_other_factors` above is literally "no") -- since
        # `facts` here is `json.dumps`-ed straight into the Gemini prompt
        # (see `assemble_llm_narrative()` below), the model was being told
        # a fabricated "direction" for a set of "other factors" that the
        # SAME payload says doesn't exist. Now "n/a" in that case, matching
        # the word-level-fact style of the rest of this payload (no JSON
        # `null`). Separately: an exact `catchall_pct_effect == 0.0` (only
        # possible with real SHAP values in the practically-impossible case
        # of the remaining factors summing to exactly zero net effect)
        # still maps to "decreases" -- an arbitrary but harmless default,
        # left as-is rather than adding a third direction word that would
        # break the system instruction's documented two-word contract
        # ("direction words (increases/decreases)").
        "other_factors_direction": (
            "n/a" if interp.shap["catchall_n_features"] == 0
            else ("increases" if interp.shap["catchall_pct_effect"] > 0 else "decreases")
        ),
        "category_label": category_label,
        "comp_evidence": interp.tier2["comp_evidence"],
        "num_comps": _count_word(interp.tier2["n_comps_found"]),
        "point_position": interp.market["point_position"],
        "interval_vs_comps": interp.market["interval_vs_comps"],
        "market_divergence": interp.market["market_divergence"],
        "category_average_position": interp.market["tier1_position"],  # renamed key: "tier1" itself contains a digit
        "confidence_level": interp.confidence["level"],
        "fired_flag_labels": fired_flag_labels,
        "imputed_fields_present": bool(interp.imputed_fields),
    }


_DIGIT_RE = re.compile(r"\d")
# AUDIT FIX 2026-09-09 (SA3, final audit): this used to also include "one",
# every ordinal ("first".."tenth"), and the fraction words "half"/"quarter"/
# "dozen" -- all of which double as ordinary English discourse markers with
# no numeric meaning ("One of the strongest influences...", "The first
# thing that stands out...", "The second most influential factor...", "A
# quarter of the drivers..."). Reproduced live: all 4 of those hand-written,
# number-free, well-grounded sentences were wrongly rejected, and the
# rejected text was then echoed back to the user verbatim (app.py's
# degradation-reasons display) -- making the AI narrative look broken far
# more often than it actually was. Removed the words demonstrated (or in
# the same discourse-marker class as what was demonstrated) to cause false
# positives; kept every word that functions as an actual, unambiguous
# numeric magnitude and would be genuinely suspicious in this model's
# number-free connective prose.
_NUMBER_WORDS = {
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")

# AUDIT FIX 2026-09-09 (SA19, final audit): removed "will definitely" -- a
# strict superset-match of the already-present "definitely" (any text
# containing "will definitely" necessarily contains "definitely" too, so
# it was a dead entry that could never independently change the result).
CERTAINTY_LEXICON = {
    "guaranteed", "guarantee", "certainly", "definitely", "undoubtedly", "assured",
    "without a doubt", "surely", "always", "never wrong", "perfect", "flawless", "risk-free",
}

SYSTEM_INSTRUCTION_CONFIDENT = (
    "You are writing four short pieces of connective prose for a house price explanation report. "
    "You will be given only qualitative facts (words, not numbers): feature names, direction words "
    "(increases/decreases), magnitude words, and category labels. All numbers, prices, and percentages "
    "are rendered separately by the application -- you must NEVER write any digit, number word, percent, "
    "or dollar amount anywhere in your response, and NEVER name a feature that was not given to you. "
    "Write in a clear, professional, confident tone (the evidence for this house is well-supported). "
    "Return ONLY the requested JSON fields, each 1-2 sentences of plain connective prose."
)
SYSTEM_INSTRUCTION_HEDGED = (
    "You are writing four short pieces of connective prose for a house price explanation report. "
    "You will be given only qualitative facts (words, not numbers): feature names, direction words "
    "(increases/decreases), magnitude words, and category labels. All numbers, prices, and percentages "
    "are rendered separately by the application -- you must NEVER write any digit, number word, percent, "
    "or dollar amount anywhere in your response, and NEVER name a feature that was not given to you. "
    "Write in a clear, professional tone, and use APPROPRIATELY HEDGED language (e.g. 'appears to', "
    "'suggests', 'is estimated to') since the evidence for this specific house is limited -- never claim "
    "certainty, and never use words like 'guaranteed', 'definitely', or 'certainly'. "
    "Return ONLY the requested JSON fields, each 1-2 sentences of plain connective prose."
)


class LLMNarrativeFields(BaseModel):
    headline: str = Field(..., description="One sentence introducing the price estimate, no numbers.")
    drivers_intro: str = Field(..., description="One sentence introducing the main price drivers, no numbers.")
    drivers_synthesis: str = Field(..., description="One sentence synthesizing what the drivers say together, no numbers.")
    market_framing: str = Field(..., description="One sentence framing how this compares to the market, no numbers.")


def _contains_disallowed_number(text: str) -> bool:
    if _DIGIT_RE.search(text):
        return True
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return bool(words & _NUMBER_WORDS)


def _contains_ungrounded_feature_reference(text: str) -> bool:
    """Heuristic leak check (step 18): flags a RAW engineered feature name
    (e.g. 'sqft_living', 'condition_x_age') leaking into the response
    verbatim -- the model was only ever given display names, so a raw
    snake_case feature name appearing is itself evidence of drift (e.g. a
    stale cached prompt/response), even before checking display names.
    Since display names are natural-language phrases, not a closed
    single-token vocabulary, this deliberately does not try to detect a
    fabricated NEW display name -- that risk is handled by never trusting
    the LLM with numbers in the first place (any number it invents is
    caught by the digit check above, which covers the majority of
    fabrication risk in this application).

    AUDIT FIX 2026-09-09 (SA18, final audit): dropped the unused
    `allowed_display_names` parameter -- this function never referenced
    it (it only ever checked raw underscore-containing names, never the
    display-name allowlist), so callers were computing and passing a set
    that had no effect on the result.

    UI_AMENDMENTS.md entry 19: only checks raw names containing an "_" --
    found (from the user pasting back a real, well-grounded Gemini
    response that got rejected anyway) that 6 of the project's 35 raw
    keys are single, ordinary English words with no underscore
    ('bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition'),
    and each one's OWN allowed display name legitimately contains that
    exact word (e.g. 'condition' -> 'overall condition', 'view' -> 'view
    quality') -- so a substring match on those 6 self-triggers on every
    correctly grounded mention of them, not just on a real raw-name leak.
    There is no reliable way to tell "the model correctly said
    'condition'" apart from "the model leaked the raw column named
    condition" by substring matching alone, since they're the same word --
    unlike a genuine multi-token raw name (e.g. 'sqft_living',
    'condition_x_age'), which could never appear in fluent prose by
    coincidence. The project's other 29 raw feature names all contain an
    underscore (AUDIT FIX 2026-09-09, SA22, final audit -- corrected from
    the "34"/"28" originally written here; `len(FEATURE_DISPLAY_NAMES)`
    is actually 35, split 6 without an underscore / 29 with one), so this
    restriction costs nothing against real leaks."""
    lowered = text.lower()
    for raw_name in FEATURE_DISPLAY_NAMES:
        if "_" in raw_name and raw_name in lowered:
            return True
    return False


def _contains_certainty_claim(text: str) -> bool:
    """AUDIT FIX 2026-09-09 (SA19, final audit): was plain substring
    matching, which was both over- and under-inclusive: (1) "reassured"
    was flagged because it contains "assured" as a literal substring, and
    "always-on" was flagged because it contains "always" -- neither is a
    certainty claim about the price estimate; (2) "It is not guaranteed."
    was flagged even though the "not" makes it hedging, the exact opposite
    of a certainty claim. Fixed with a word-boundary match (a hyphen or
    underscore counts as part of the "word" for this purpose, so
    "always-on" no longer matches standalone "always") plus a short
    look-back for a preceding negation word, which exempts a genuinely
    hedged phrase like "not guaranteed" or "never certain" from the ban."""
    lowered = text.lower()
    for phrase in CERTAINTY_LEXICON:
        pattern = r"(?<![a-z0-9_-])" + re.escape(phrase) + r"(?![a-z0-9_-])"
        for m in re.finditer(pattern, lowered):
            preceding = lowered[max(0, m.start() - 15):m.start()]
            if re.search(r"\b(not|never|no|n't)\s+$", preceding):
                continue
            return True
    return False


def validate_llm_response(fields: LLMNarrativeFields, confidence_level: str) -> list[str]:
    """Returns a list of validation failure reasons (empty = passes all of
    step 18's checks).

    AUDIT FIX 2026-09-09 (SA3, verification pass): these reasons are
    user-facing -- `app.py`'s "Why isn't this AI-written?" expander (and,
    before that, its cache-retry check) render `InterpretationResult.
    degradation_reasons` verbatim via `st.write()`. The original SA3 fix
    stopped an ordinary discourse word like "half" from being wrongly
    flagged as a number leak, but every reason string here still embedded
    the FULL flagged text via `{text!r}` -- so any real leak this function
    exists to catch (a raw dollar figure, an ungrounded feature name) was
    then echoed straight back to the end user by the very fallback screen
    meant to hide it, and even a false-positive case (e.g. "Three factors
    drive this estimate.") surfaced clearly-LLM-flavored prose on what's
    supposed to be a plain Machine-Statistics explanation admitting the AI
    path failed. Reproduced directly: `esc_dollars(r)` only escapes `$`
    signs, it does not remove digits or feature names. Reasons now name the
    field and the rule violated without quoting the disallowed text itself
    -- the full text is still available to a developer via
    `fields.model_dump()` at the call site if needed for debugging."""
    reasons = []
    for field_name, text in fields.model_dump().items():
        if not text or not text.strip():
            reasons.append(f"{field_name} is empty")
            continue
        if _contains_disallowed_number(text):
            reasons.append(f"{field_name} contains a digit or number word (not shown here to avoid leaking it)")
        if _contains_ungrounded_feature_reference(text):
            reasons.append(f"{field_name} references a raw/ungrounded feature name (not shown here to avoid leaking it)")
        if confidence_level != "well-supported" and _contains_certainty_claim(text):
            reasons.append(f"{field_name} makes an inappropriate certainty claim for confidence={confidence_level!r} (not shown here to avoid leaking it)")
    return reasons


GEMINI_MODEL = "gemini-3.6-flash"
# UI_AMENDMENTS.md entry 4: gemini-2.5-flash started 404'ing for this app's
# API key ("no longer available to new users") even though it's still
# listed as stable -- Google is rolling out usage-history-based access
# restrictions on the 2.5 series, not a full deprecation. gemini-3.6-flash
# is Google's own suggested replacement in that error message.
# Step 21: no token-bucket rate limiter needed (single-user demo app, per
# the sizing note) -- instead, retry ONLY on retryable transport failures
# (408/429/5xx), zero retries on a content/validation failure (those never
# reach this layer's retry logic at all -- they're caught downstream,
# after a successful HTTP response, by validate_llm_response()), and a
# hard overall wall-clock cap so a single slow/hanging call can never
# block the app indefinitely.
#
# UI_AMENDMENTS.md entry 12: the original 4000ms per-request timeout
# started failing outright with `400 INVALID_ARGUMENT: Manually set
# deadline 4s is too short. Minimum allowed deadline is 10s.` -- Google
# now server-side-enforces a 10s minimum on this parameter (undocumented
# at the time this was first written), so ANY call was rejected before
# even attempting a request, regardless of network speed. Raised to the
# new floor, and the wall-clock cap raised proportionally (originally
# sized to cover 2 attempts @ 4s + backoff; now covers 2 attempts @ 10s +
# backoff, same ratio/intent, just against the new minimum).
GEMINI_RETRY_ATTEMPTS = 2          # total attempts, including the first
GEMINI_PER_REQUEST_TIMEOUT_MS = 10_000  # Google's enforced minimum -- do not lower this
HARD_WALL_CLOCK_CAP_S = 22.0       # covers 2 attempts @ 10s + backoff (0.5-2s) under budget


def _do_generate_content(api_key: str, prompt: str, system_instruction: str):
    """The actual blocking network call, isolated into its own function so
    it can be run in a worker thread and hard-capped by `.result(timeout=)`
    below -- the SDK's own per-request timeout bounds each individual
    attempt, but only a wall-clock join on the whole call (including any
    SDK-internal retries) gives a real, unconditional upper bound."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_PER_REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=GEMINI_RETRY_ATTEMPTS,
                initial_delay=0.5, max_delay=2.0, exp_base=2.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],  # never retries a plain 4xx (bad request/schema)
            ),
        ),
    )
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=LLMNarrativeFields,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),  # step 16 / entry 13: 3.x's minimal-effort setting
            max_output_tokens=1024,
            temperature=0.4,  # pinned, not tuned -- see step 16's rationale
        ),
    )


def call_gemini_narrative(interp, api_key: Optional[str] = None) -> dict:
    """Returns {"fields": LLMNarrativeFields | None, "source": "llm"|"failed",
    "degradation_reasons": [...]}. NEVER raises -- any failure (missing
    key, import error, network error, timeout, malformed/invalid response)
    is caught and reported as a degradation reason so the caller falls
    back to the template narrative untouched."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"fields": None, "source": "failed", "degradation_reasons": ["no Gemini API key configured"]}

    try:
        from google.genai import errors as genai_errors
    except ImportError as e:
        return {"fields": None, "source": "failed", "degradation_reasons": [f"google-genai import failed: {e}"]}

    # AUDIT FIX 2026-09-09 (SA4, final audit): `build_prompt_facts(interp)`
    # used to run OUTSIDE any try/except here, so anything it raised
    # propagated straight out of this function -- breaking this
    # function's own docstring promise ("NEVER raises") and, one level up,
    # `interpret_house()`'s "any failure falls back to the template
    # untouched" guarantee. The original trigger was `_band_tier_word()`
    # indexing `_TIER_WORDS_BY_K[n_bands][band]` with no bounds check
    # (`IndexError` for band >= n_bands) -- that whole invented-tier-word
    # scheme is gone now (SA11's fix replaced it with a plain, safe dict
    # lookup), so the SPECIFIC repro no longer fires, but the finding's
    # real point stands independently of any one trigger: this function
    # calls into enough downstream code (feature-label lookups, flag-label
    # lookups, JSON serialization) that a FUTURE change anywhere in that
    # call graph could raise again, and this is the one place designed to
    # guarantee it never surfaces as an unhandled exception. Wrapping the
    # whole body from here on is the actual second line of defense the
    # finding asked for -- distinct from (and a tighter guarantee than)
    # SB4's broader app.py-level try/except, which would still fail the
    # ENTIRE estimate rather than gracefully falling back to the template
    # narrative the way this function's contract promises.
    try:
        facts = build_prompt_facts(interp)
        system_instruction = (
            SYSTEM_INSTRUCTION_CONFIDENT if interp.confidence["level"] == "well-supported" else SYSTEM_INSTRUCTION_HEDGED
        )
        prompt = (
            "Facts about this house's price estimate (qualitative only -- no numbers were given to you, "
            "and you must not invent any):\n" + json.dumps(facts, indent=2)
        )
    except Exception as e:  # noqa: BLE001 -- this function must never raise, per its own contract
        return {"fields": None, "source": "failed", "degradation_reasons": [f"building the Gemini prompt failed: {e}"]}

    # Deliberately a raw daemon `threading.Thread`, NOT
    # `concurrent.futures.ThreadPoolExecutor`: verified directly (see
    # PROJECT_TODO.md) that even calling `pool.shutdown(wait=False)` after
    # a timeout does NOT let the interpreter exit promptly -- `concurrent.
    # futures.thread` registers a module-level `atexit` hook
    # (`_python_exit`) that joins EVERY thread any ThreadPoolExecutor ever
    # started, process-wide, regardless of that pool's own shutdown call.
    # A genuinely hung Gemini call would silently block a clean process
    # shutdown/restart. A plain daemon thread has no such hook -- the
    # calling code proceeds the instant the timeout fires, and the
    # abandoned thread is dropped (not joined) when the process exits.
    result_box: dict = {}

    def _target():
        try:
            result_box["value"] = _do_generate_content(api_key, prompt, system_instruction)
        except Exception as e:  # noqa: BLE001 -- re-raised on the calling side below
            result_box["error"] = e

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=HARD_WALL_CLOCK_CAP_S)

    if worker.is_alive():
        return {"fields": None, "source": "failed",
                "degradation_reasons": [f"Gemini call exceeded the {HARD_WALL_CLOCK_CAP_S}s hard wall-clock cap"]}
    if "error" in result_box:
        e = result_box["error"]
        if isinstance(e, (genai_errors.ClientError, genai_errors.ServerError)):
            return {"fields": None, "source": "failed", "degradation_reasons": [f"Gemini API error (code={getattr(e, 'code', '?')}): {e}"]}
        return {"fields": None, "source": "failed", "degradation_reasons": [f"Gemini call failed: {e}"]}
    response = result_box["value"]

    finish_reason = getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
    if str(finish_reason) not in ("STOP", "FinishReason.STOP", "1"):
        return {"fields": None, "source": "failed", "degradation_reasons": [f"finish_reason={finish_reason!r} (not STOP)"]}

    raw_text = response.text
    if not raw_text or not raw_text.strip():
        return {"fields": None, "source": "failed", "degradation_reasons": ["empty response text"]}

    try:
        parsed = json.loads(raw_text)
        fields = LLMNarrativeFields.model_validate(parsed)
    except Exception as e:  # noqa: BLE001
        return {"fields": None, "source": "failed", "degradation_reasons": [f"JSON/schema validation failed: {e}"]}

    reasons = validate_llm_response(fields, interp.confidence["level"])
    if reasons:
        return {"fields": None, "source": "failed", "degradation_reasons": reasons}

    return {"fields": fields, "source": "llm", "degradation_reasons": []}


def assemble_llm_narrative(interp, llm_fields: LLMNarrativeFields) -> str:
    """Step 19's fixed assembly order: price+interval -> drivers intro ->
    driver lines (code-rendered, |SHAP|-rank order, with a combined lead
    sentence if the top 2 are within ~15% relative magnitude of each
    other, per the plan's "never a forced false ranking") -> synthesis ->
    market framing -> point_position + interval_vs_comps (+divergence) ->
    caveats -> disclaimer. The LLM's 4 fields are slotted in as connective
    prose AROUND these code-rendered facts -- it never states a number
    itself; every number below comes straight from `interp`, unchanged
    from the template path, preserving the single-source-of-truth
    guarantee even in the LLM-assisted narrative."""
    # AUDIT FIX 2026-09-09 (SA31c, final audit): deliberately a LOCAL
    # (function-body) import, not a module-level one. step6_narrative.py
    # imports FROM this module at ITS module level (FEATURE_DISPLAY_NAMES,
    # feature_label, group_top_features_by_label), so a module-level
    # import here in the opposite direction would be a real circular
    # import -- Python only gets away with the pair as written because
    # this one is deferred until this function actually runs, by which
    # point both modules have already finished loading. Document, not
    # untangle: reordering which module owns `_fmt_money`/`_fmt_pct`
    # would be a bigger, riskier change for a cosmetic quality item, and
    # the current arrangement already works correctly (verified: both
    # import directions run clean).
    from step6_narrative import _fmt_money, _fmt_pct  # reuse the exact same formatters as the template path

    lines = []
    price, lo, hi = interp.predicted_price, interp.cqr_lo, interp.cqr_hi
    lines.append(llm_fields.headline.strip())
    # UI_AMENDMENTS.md entry 18: warmer phrasing for this code-rendered
    # fact line (same numbers, unchanged) -- matches the template path's
    # rewrite so the two narrative sources read consistently.
    lines.append(f"We estimate this home is worth {_fmt_money(price)}, with a likely range of {_fmt_money(lo)} to {_fmt_money(hi)}.")
    lines.append("")

    lines.append(llm_fields.drivers_intro.strip())
    # AUDIT FIX 2026-09-09 (SA1, final audit): this iterated the raw,
    # un-merged `top_features` list, so two features sharing a display
    # name (e.g. sqft_lot/log_sqft_lot -> "lot size") produced two driver
    # lines with the same label and two different percentages -- the
    # entry-18 comment directly above claims this narrative "reads
    # consistently" with the template path, which was false whenever this
    # collision fired (confirmed on 30% of real houses). Grouped the same
    # way the chart and the no-AI narrative are.
    top = group_top_features_by_label(interp.shap["top_features"])
    driver_lines = []
    for i, g in enumerate(top):
        if i == 1 and abs(top[0]["pct_effect"]) > 0 and abs(g["pct_effect"] - top[0]["pct_effect"]) / abs(top[0]["pct_effect"]) < 0.15:
            driver_lines.append(
                f"  - {g['label']} and {top[0]['label']} contribute "
                f"comparably ({_fmt_pct(g['pct_effect'])} and {_fmt_pct(top[0]['pct_effect'])})"
            )
        else:
            driver_lines.append(f"  - {g['label']} ({_fmt_pct(g['pct_effect'])})")
    lines.extend(driver_lines)
    if interp.shap["catchall_n_features"]:
        lines.append(f"  - {interp.shap['catchall_n_features']} other smaller factors combined ({_fmt_pct(interp.shap['catchall_pct_effect'])})")
    lines.append("")
    lines.append(llm_fields.drivers_synthesis.strip())
    lines.append("")

    # AUDIT FIX 2026-09-09 (SA12, final audit): this LLM-assembled narrative
    # used to be a strict INFORMATION SUBSET of the template narrative --
    # it dropped the entire Tier-1 profile paragraph (category name,
    # min/max/mean price, comp count) entirely, contradicting entry 18's
    # own claim (quoted in this module's docstring history) that the
    # rewrite made "the two narrative sources read consistently". Per
    # Decision 2's "deterministic spine + LLM connective tissue"
    # architecture, this code-owned fact belongs in BOTH paths -- it's
    # ported here verbatim from step6_narrative.py (including SA13's
    # p10-p90 fix, so both paths get the corrected range at once).
    from step6_market_comparison import point_position_phrase, tier2_tier_phrase
    import numpy as np
    t1 = interp.tier1_profile
    cat_prices = np.asarray(t1["sorted_prices"], dtype=float)
    p10, p90 = (float(v) for v in np.percentile(cat_prices, [10, 90]))
    # 2026-09-18 FIX: this line was ported verbatim from step6_narrative.py
    # (see SA12 above) BEFORE that file's raw-`combined_category`-leak bug
    # (see step6_interpretation.py's `category_label` comment) was found and
    # fixed there -- so the porting silently reproduced the same bug here,
    # independently, in the one narrative path (`narrative_source="llm"`)
    # that is actually live for users with a working Gemini key configured.
    # Confirmed live: with the cluster rename deployed, this exact sentence
    # still rendered the raw internal key ("...we call this group 'Physical
    # Group 0 - Price Band 1'...") on the running app, one paragraph above
    # the correctly-renamed "Untouched Classics" text the LLM prose itself
    # used (`build_prompt_facts()` above already computes `category_label`
    # correctly -- this hand-assembled sentence just wasn't reading it).
    # Same fix, same canonical lookup, computed locally since this function
    # receives the raw `InterpretationResult`, not the `category_label`-
    # enriched `narrative_input` dict the template path gets.
    from step6_classify_house import load_classification_tables
    category_label = digit_free_category_label(interp.combined_category, load_classification_tables())
    lines.append(
        f"This home fits the profile of similar properties in our data (we call this group "
        f"'{category_label}') -- homes like it typically sell for "
        f"{_fmt_money(p10)} to {_fmt_money(p90)}, averaging around "
        f"{_fmt_money(t1['mean_price'])}, based on {t1['count']} similar sales."
    )
    lines.append("")

    lines.append(llm_fields.market_framing.strip())
    n_comps = interp.tier2["n_comps_found"]
    # AUDIT FIX 2026-09-09 (SA12, final audit): this used to print the RAW
    # `point_position` enum ("...reads as 'upper' vs. the local market")
    # and never mentioned WHICH comps ("in the same zip code" etc.) --
    # both facts the template path already stated in plain English. Now
    # shares the same translation dicts (step6_market_comparison.py) the
    # template path uses, so the two agree instead of one leaking an enum.
    tier_label = tier2_tier_phrase(interp.tier2["tier2_tier_used"])
    position_phrase = point_position_phrase(interp.market["point_position"])
    # AUDIT FIX 2026-09-09 (SA31g, final audit): same fix as step6_
    # narrative.py's identical sentence (rule 11) -- built conditionally
    # so an unmapped tier degrades gracefully instead of leaving a double
    # space and a floating comma.
    tier_clause = f" {tier_label}" if tier_label else ""
    lines.append(
        f"(Based on {n_comps} comparable sales{tier_clause}, this estimate comes in "
        f"{position_phrase} what similar homes have actually sold for.)"
    )
    if interp.market["interval_vs_comps"] == "overlapping":
        # AUDIT FIX 2026-09-09 (SA5, final audit): see step6_narrative.py's
        # identical fix comment -- the old "a bit wider than" claim wasn't
        # actually tested by the "overlapping" condition it was gated on,
        # and was demonstrably false for 4 of 908 real houses.
        lines.append(
            "Worth noting: our confidence range overlaps with the typical spread of nearby sale prices, "
            "so treat that range -- not just the single estimate above -- as the more honest picture of "
            "how much this could vary."
        )
    if interp.market["market_divergence"] == "diverges":
        # AUDIT FIX 2026-09-09 (SA8, final audit): see step6_narrative.py's
        # identical fix -- fires for 56.1% of real houses, so it's reworded
        # to read as routine and expected rather than as a notable exception.
        # AUDIT FIX 2026-09-09 (SA12, final audit): this used to say the
        # picture "reads a little differently" without ever saying HOW --
        # the template path states the actual `tier1_position` word; this
        # path dropped it, another instance of the LLM narrative being a
        # strict information subset of the template one. Included below
        # for parity.
        lines.append(
            f"For additional context, by a different measure (this home's own price-category average), "
            f"the picture reads as '{interp.market['tier1_position']}', which differs somewhat from the "
            f"comparable-sales view above. That's common and expected -- both are legitimate ways to look "
            f"at the same home; they're just measuring slightly different things."
        )
    lines.append("")

    conf = interp.confidence
    lines.append(f"Overall, we'd call this a '{conf['level']}' confidence estimate.")
    if conf["reasons"]:
        lines.append("That's mainly because:")
        for r in conf["reasons"]:
            lines.append(f"  - {r}")
    lines.append("")

    if interp.imputed_fields:
        # AUDIT FIX 2026-09-09 (SA10, final audit): same fix as
        # step6_narrative.py's identical block -- raw column names (e.g.
        # "sqft_above") were being shown to the customer instead of the
        # display names already used elsewhere on this exact screen.
        friendly = [FEATURE_DISPLAY_NAMES.get(f, f) for f in interp.imputed_fields]
        lines.append(
            f"A quick note: you didn't provide {', '.join(friendly)}, so we filled those "
            f"in using typical values from our data -- treat the estimate as a bit more approximate "
            f"for those details."
        )
        lines.append("")

    # UI_AMENDMENTS.md entry 15: closing appraisal/financial-advice
    # disclaimer line removed from the narrative text itself, per explicit
    # instruction -- the separate graduation-project brand disclaimer
    # (entry 3) already lives in the page footer, unrelated to this line.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
