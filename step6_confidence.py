"""Phase A, step 9 -- confidence tiers, RENAMED honestly rather than tuned
to look better (per "THE critical finding"'s sibling issue in
PROJECT_TODO.md): low -> "limited evidence", medium -> "typical", high ->
"well-supported". Measured real distribution on all 908 held-out test
houses, run directly through `interpret_house()` end-to-end (AUDIT FIX
2026-09-09, SA16, final audit -- replaces this docstring's earlier claim
of "~22% / ~67% / ~10%", which did not reproduce and turned out to
mis-cite a DIFFERENT, 100-house sample logged in PROJECT_TODO.md as if it
were the full 908): 18.9% limited-evidence / 69.8% typical / 11.2%
well-supported -- a median CQR relative width of 0.672 makes the strict
"well-supported" gate (<=0.5) genuinely rare, and that is reported as-is
rather than relaxed to manufacture a fuller top bucket.

2026-09-18 RE-MEASURE (validation-logic audit + PACKAGING_TODO item 7):
the figures above are the ORIGINAL SA16 measurement and are kept for
history, but no longer describe this file's current behavior, for two
independent reasons layered on top of each other:
  1. The validation-logic audit's Tier A fix for `yr_renovated < yr_built`
     now hard-rejects 44 of the 908 held-out houses that used to reach
     this function at all -- MEASURED (via `step6_validation.validate_
     tier_a` directly against this split), all 44 of 44 are this one
     rejection reason; none of the sqft_above/sqft_basement relational
     checks or the bathroom-ratio floor fire on this particular 908-row
     split. The denominator for "scored" houses is now 864, not 908.
  2. Item 7 (`beyond_training_range_fields`, just below) adds one more
     trigger to the list.
  Re-measured the same way (`interpret_house()` end-to-end over the same
  908-row split): **19.1% limited-evidence / 70.4% typical / 10.5%
  well-supported** (of the 864 actually scored; 44 are now Tier A
  rejections, 0 raised exceptions). Isolating item 7's OWN marginal
  contribution specifically: its new trigger fired on 4 of the 864 scored
  houses (0.5%), and all 4 were ALREADY "limited evidence" for a
  pre-existing reason -- MEASURED (not assumed; an earlier draft of this
  note wrongly guessed `yr_built_after_training_window`, which is
  impossible on this dataset since every row's `yr_built` predates the
  2014 sale_year by construction): all 4 already had `relative_width >
  0.9` (1.36-1.81), and 3 of the 4 additionally hit the `nearest_
  neighbors` Tier 2 fallback. So item 7 added an accurate caveat SENTENCE
  to those 4 houses' explanations, but moved zero houses across a tier
  boundary by itself on this held-out set. The ~0.2-point rise in the
  limited-evidence share versus a naive "add 19.1 - 18.9" comparison is
  therefore almost entirely attributable to reason 1 (the smaller,
  harder-rejected denominator), not to item 7.

Trigger list (any ONE fires "limited evidence", per round-2's list, plus
item 7's addition below):
  - Tier 2 comp_evidence == "insufficient" (<5 comps found)
  - location resolved via the global/dataset-wide fallback (not zip/city)
  - predicted price falls outside the TRAINING SPLIT's 1st-99th percentile
    ($146,200 - $2,049,800 -- measured from the 3,641-row training split
    produced by `regression_pipeline.group_train_test_split`, NOT the
    full 4,549-row enriched_dataset.csv, which would leak the 908 held-out
    test rows into a threshold used to grade those same rows -- AUDIT FIX
    2026-09-09, SA7, final audit)
  - price_band was clipped by band_validity_guard (a structurally
    impossible classifier output that had to be corrected)
  - an IMPUTED optional field (Decision 3d) landed in the SHAP top-K
    drivers -- price attributed to a value the user never actually
    supplied (the exact cross-decision interaction Opus round 3 caught)
  - CQR relative interval width > 0.9
  - Tier 2 comps came from the NearestNeighbors last-resort fallback
    (`tier2_tier_used == "nearest_neighbors"`) -- AUDIT FIX 2026-09-09
    (SA9, final audit): `step6_market_comparison.py`'s own docstring
    promises this step penalizes "reached the NearestNeighbors last
    resort" even when the raw comp count looks "strong" (it always does:
    `n_neighbors = min(target, len(pool))` == target), but this module
    never actually read `tier2_tier_used` until now.
  - one or more of bedrooms/bathrooms/floors/sqft_living/sqft_lot/
    sqft_above/sqft_basement/basement_ratio falls outside the range this
    model actually trained on, even though it passed Tier A's deliberately
    wider bounds (2026-09-18 ADDITION, PACKAGING_TODO item 7 -- see
    `step6_inference_contract.TRAINING_OBSERVED_RANGES`)
  - house_age exceeds `max_trained_house_age` (the oldest house this
    model actually trained on), i.e. `beyond_training_age_range` is set
    (2026-09-19 FIX, post-delivery code-logic review -- this flag was
    already raised by step6_inference_contract.py's age_bucket clamp
    guard, per that module's own docstring, but this function never
    read it until now)

"well-supported" is coded as ALL of: zero limited-evidence triggers, Tier
2 comp count >= 10, location resolved at the zip level, and relative
width <= 0.5. AUDIT FIX 2026-09-09 (SA26, final audit): on the real
908-house held-out set (908/908 scored, before the validation-logic audit
added any Tier A rejections), two of those four conjuncts were
non-binding -- `location_fallback == "zip"` held for 908/908 houses and
`n_comps_found >= 10` for 907/908 -- so in practice this gate collapsed to
`relative_width <= 0.5` alone (measured: 102/908 houses passed the
relative-width condition, and exactly 102/908 were "well-supported" --
not merely close, an exact match).

2026-09-18 RE-MEASURE: re-run against the current 864 actually-scored
houses (908 minus the 44 the validation-logic audit now hard-rejects, per
the re-measure note above) the same exact-match property still holds:
91/864 pass the relative-width condition, and exactly 91/864 are
"well-supported". The code is written as a four-way gate and is left that
way (each condition is still a genuine, separately justified requirement,
and a future dataset/model with more location or comp-count variance
would make them binding again); what changed is only this docstring,
which used to present the gate as if the data made all four conjuncts
equally load-bearing today. Everything that fails "well-supported" is
"typical".
"""
from __future__ import annotations

# AUDIT FIX 2026-09-09 (SA7, final audit): these were the FULL 4,549-row
# enriched_dataset.csv's percentiles (148_000 / 2_016_440), not the
# training split's -- so a price that is well within the model's actual
# training range could get flagged "falls outside the typical price range"
# using a threshold computed partly from the very rows held out to grade
# it. Recomputed from the training split only:
#   train_df, test_df = regression_pipeline.group_train_test_split(df)
#   np.percentile(train_df["price"], [1, 99]) -> (146_200.0, 2_049_799.99...)
TRAINING_PRICE_P1 = 146_200.0
TRAINING_PRICE_P99 = 2_049_800.0
WELL_SUPPORTED_MAX_RELATIVE_WIDTH = 0.5
# AUDIT FIX 2026-09-09 (SA31e, final audit): the trigger below is
# `relative_width > LIMITED_EVIDENCE_MIN_RELATIVE_WIDTH` -- a strictly-
# greater-than (exclusive) comparison, so a width of EXACTLY 0.9 does
# NOT trigger "limited evidence" despite the "MIN" in this constant's
# name suggesting an inclusive floor (confirmed: `relative_width=0.9` ->
# "typical"; `0.9001` -> "limited evidence"). Documented rather than
# changed to `>=`: 0.9 landing on the "typical" side is the behavior
# this module's own trigger list and every existing test already assume,
# and flipping it would be a real behavior change smuggled into a
# naming-cleanup finding.
LIMITED_EVIDENCE_MIN_RELATIVE_WIDTH = 0.9


def compute_confidence(engineered_flags: dict, tier2: dict, relative_width: float,
                        predicted_price: float, imputed_fields_in_top_k: list[str]) -> dict:
    # UI_AMENDMENTS.md entry 6: reason strings rewritten in warmer, plainer
    # language (same underlying facts/triggers, same trigger logic below --
    # only the customer-facing wording changed) so the main "Estimate" page
    # reads like confident commercial copy rather than technical hedging.
    reasons_limited = []

    if tier2["comp_evidence"] == "insufficient":
        # AUDIT FIX 2026-09-09 (SA25, final audit): "nearby" was actively
        # misleading here -- this branch is only reachable once even the
        # whole-dataset NearestNeighbors fallback (get_tier2_comps) still
        # returned fewer than FLOOR_COMPS rows, i.e. by construction these
        # comps are NOT nearby when this text is shown. "sale(s)" was also
        # a programmer's plural, at odds with UI_AMENDMENTS entry 6's
        # rewrite of this module's copy into plain commercial English.
        n_comps = tier2["n_comps_found"]
        sale_word = "sale" if n_comps == 1 else "sales"
        reasons_limited.append(
            f"we could only find {n_comps} comparable {sale_word} to check this estimate against, even "
            f"after widening the search across the whole dataset"
        )
    # AUDIT FIX 2026-09-09 (SA27, final audit): was `engineered_flags.get(
    # "location_fallback")`, which would silently return "typical" (never
    # "well-supported", with no error) if this key were ever absent --
    # inconsistent with `tier2["comp_evidence"]` below, a bare (loud)
    # index lookup that raises `KeyError` on a partial dict. Unlike the
    # genuinely OPTIONAL flags in this dict (e.g. "price_band_clipped",
    # only present when it fires), `location_fallback` is set on every
    # single code path through `step6_inference_contract.py`'s zip ->
    # city -> global lookup (confirmed by reading all three branches --
    # not assumed) -- so a missing key here means a real caller-contract
    # break, and this function should fail loudly on that, exactly like
    # the `tier2` access does, rather than silently degrading a result's
    # confidence tier with no signal that anything went wrong.
    location_tier = engineered_flags["location_fallback"]
    if location_tier == "global":
        reasons_limited.append(
            "we don't have local pricing data for this exact area yet, so this leans on broader market data instead"
        )
    # 2026-09-17 DESIGN CHANGE: yr_built moved from a Tier A hard-reject to
    # this Tier B soft flag (step6_validation.py, step6_inference_contract.py).
    # A house built after 2014 (the most recent year in the training data)
    # still gets a real prediction -- house_age is clamped to 0, i.e. priced
    # as if built in 2014 -- but that assumption gets its own specific,
    # plainly-worded reason here, rather than being folded into the generic
    # "wide interval" or "outside typical range" messages below.
    if engineered_flags.get("yr_built_after_training_window"):
        reasons_limited.append(
            "this home was built after 2014, the most recent year in our training data, so we're pricing it as "
            "if it were built in 2014, which may not reflect newer construction standards, materials, or buyer "
            "expectations since then -- treat this estimate with extra caution"
        )
    # 2026-09-17 DESIGN CHANGE: same pattern as yr_built_after_training_window
    # above, for a renovation dated past 2014 (step6_validation.py's
    # yr_renovated bound moved to 2026 alongside yr_built; see
    # step6_inference_contract.py for the matching flag/clamp).
    if engineered_flags.get("yr_renovated_after_training_window"):
        reasons_limited.append(
            "this home's renovation happened after 2014, the most recent year in our training data, so we're "
            "pricing the renovation as if it happened in 2014, which may not reflect newer renovation standards "
            "or buyer expectations since then -- treat this estimate with extra caution"
        )
    # 2026-09-19 FIX (post-delivery code-logic review): step6_inference_
    # contract.py's own docstring (Step 3, age_bucket upper-edge guard)
    # promises that when a house is older than anything this model
    # actually trained on (house_age > max_trained_house_age, currently
    # 114 years / yr_built before 1900), it clamps the house into the top
    # age_bucket AND "a `beyond_training_age_range` flag is raised for
    # the confidence/caveat layer built in a later Phase A step" -- that
    # later step is this function, and until this fix it never actually
    # read the flag. Net effect before this fix: a very-old-house
    # prediction got the same clamped-bucket treatment as every other
    # beyond-range extrapolation, but silently kept whatever confidence
    # tier its comps/interval-width happened to produce, with no caveat
    # telling the user the model is extrapolating past its oldest
    # training example -- the one beyond-range condition in this whole
    # file that didn't demote confidence or add a reason string. Same
    # trigger family as yr_built_after_training_window above (both flag
    # a house whose age falls outside what the model actually trained
    # on), just via the opposite direction (older, not newer) and a
    # different upstream mechanism (age_bucket clamp, not a hard reject).
    if engineered_flags.get("beyond_training_age_range"):
        reasons_limited.append(
            "this home is older than any house in our training data, so we're extrapolating beyond the "
            "oldest properties the model has actually seen -- treat this estimate with extra caution"
        )
    # 2026-09-18 ADDITION (PACKAGING_TODO item 7 -- the Tier B "outside
    # training range" disclosure this module's sibling, step6_validation.py,
    # had documented as existing but never actually built): same family as
    # `yr_built_after_training_window`/`yr_renovated_after_training_window`
    # above, generalized to every other field where Tier A's hard-reject
    # bound is deliberately much wider than what this model actually
    # trained on (see step6_inference_contract.TRAINING_OBSERVED_RANGES for
    # the exact, train-split-only bounds and how they were derived). Folded
    # into ONE combined sentence naming every affected field, rather than
    # one reason per field, using the same "combine into one sentence"
    # pattern as the imputed-fields case below -- a submission with several
    # unusual fields at once gets one clear caveat, not a wall of near-
    # identical ones.
    beyond_range_fields = engineered_flags.get("beyond_training_range_fields")
    if beyond_range_fields:
        from step6_llm_narrative import FEATURE_DISPLAY_NAMES
        friendly = [FEATURE_DISPLAY_NAMES.get(f, f) for f in beyond_range_fields]
        if len(friendly) == 1:
            field_phrase, verb, it_them = friendly[0], "is", "it"
        else:
            field_phrase = ", ".join(friendly[:-1]) + f" and {friendly[-1]}"
            verb, it_them = "are", "them"
        reasons_limited.append(
            f"this home's {field_phrase} {verb} outside the range of houses in our training data, so "
            f"we're extrapolating beyond anything the model has actually seen for {it_them} -- treat this "
            f"estimate with extra caution"
        )
    if predicted_price < TRAINING_PRICE_P1 or predicted_price > TRAINING_PRICE_P99:
        reasons_limited.append(
            f"this estimate (${predicted_price:,.0f}) falls outside the typical price range we usually see "
            f"(${TRAINING_PRICE_P1:,.0f}-${TRAINING_PRICE_P99:,.0f}), so treat it as a rough ballpark"
        )
    if engineered_flags.get("price_band_clipped"):
        reasons_limited.append("we had to make a small correction to this home's price category")
    if imputed_fields_in_top_k:
        # Step 7 polish (found by actually reading a rendered app screen):
        # show the human-readable feature name here, not the raw internal
        # column name (e.g. "above-ground living area", not "sqft_above") --
        # the same display-name map already used for the LLM-facing prompt,
        # reused rather than duplicated (rule 11). AUDIT FIX 2026-09-09
        # (SA17, verification pass): dropped this comment's "and vocabulary
        # allowlist" clause -- FEATURE_DISPLAY_NAMES is not used as an
        # allowlist anywhere; see the corrected comment above its
        # definition in step6_llm_narrative.py for what the actual
        # ungrounded-feature check does.
        from step6_llm_narrative import FEATURE_DISPLAY_NAMES
        friendly = [FEATURE_DISPLAY_NAMES.get(f, f) for f in imputed_fields_in_top_k]
        it_they = "it" if len(friendly) == 1 else "they"
        reasons_limited.append(
            f"we filled in {', '.join(friendly)} using typical values, and {it_they} turned out "
            f"to matter for this estimate"
        )
    if relative_width > LIMITED_EVIDENCE_MIN_RELATIVE_WIDTH:
        reasons_limited.append(f"our confidence range for this estimate is wider than usual ({relative_width:.0%} of the price)")

    # AUDIT FIX 2026-09-09 (SA9, final audit): `comp_evidence` alone can't
    # tell "strong" apart from "reached the NearestNeighbors last resort" --
    # `get_tier2_comps()` sets `n_neighbors = min(target, len(pool))`, which
    # is always `target` in practice, so the last-resort fallback is
    # indistinguishable from a genuine same-area "strong" comp set by count
    # alone. `tier2_tier_used` is the one field that actually distinguishes
    # them, and step6_market_comparison.py's own docstring already promised
    # this step would read it -- it never did until now.
    if tier2.get("tier2_tier_used") == "nearest_neighbors":
        reasons_limited.append(
            "the comparable sales we found weren't close matches by location, so we used the closest "
            "homes by size and features instead"
        )

    if reasons_limited:
        return {"level": "limited evidence", "reasons": reasons_limited}

    is_well_supported = (
        tier2["n_comps_found"] >= 10
        and location_tier == "zip"
        and relative_width <= WELL_SUPPORTED_MAX_RELATIVE_WIDTH
    )
    if is_well_supported:
        return {"level": "well-supported", "reasons": []}

    return {"level": "typical", "reasons": []}
