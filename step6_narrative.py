"""Phase A, step 11 -- the no-AI template narrative. Per PROJECT_TODO.md:
"Ship + demo the no-AI template path FIRST -- it IS the spine; the AI path
is spine + LLM prose. Satisfies PDF's Deployment criterion with zero API
dependency before any Gemini code exists."

Everything here is CODE-COMPUTED and safe to print verbatim -- unlike
Phase B's future LLM prompt (which must never receive raw numerals, per
the number-free-prompt invariant), this template has no hallucination risk
at all, so it states dollar amounts and percentages directly. This is
`narrative_source="template"`; a later Gemini-backed path
(`narrative_source="llm"`/`"llm_partial"`) will PREPEND connective prose
around these same code-owned facts, never replace them.
"""
from __future__ import annotations

import numpy as np

from step6_llm_narrative import FEATURE_DISPLAY_NAMES, feature_label, group_top_features_by_label
from step6_market_comparison import point_position_phrase, tier2_tier_phrase


def _fmt_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p*100:.1f}%"


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def build_template_narrative(interp: dict) -> str:
    """UI_AMENDMENTS.md entry 18: rewritten in warmer, first-person "we"
    language (per feedback that the previous key/value-style lines --
    "Predicted price: ...", "This estimate is driven mainly by: ...",
    "Confidence: X." -- read as a technical printout, not something a
    customer would enjoy reading). Every FACT and NUMBER below is
    unchanged from before (still code-computed, still exact) -- only the
    sentence-level phrasing wrapping those facts changed."""
    lines = []

    price = interp["predicted_price"]
    lo, hi = interp["cqr_lo"], interp["cqr_hi"]
    lines.append(
        f"Based on the details you gave us, we estimate this home is worth {_fmt_money(price)}, "
        f"with a likely range of {_fmt_money(lo)} to {_fmt_money(hi)}."
    )
    lines.append("")

    shap = interp["shap"]
    # UI_AMENDMENTS.md entry 14: was printing the raw engineered feature
    # name here (e.g. "location_price_per_sqft"), unlike every other
    # customer-facing surface in this app -- now goes through the same
    # `feature_label()` translation as the LLM-assisted narrative.
    #
    # AUDIT FIX 2026-09-09 (SA1, final audit): that fix translated each
    # feature's name but never merged features that translate to the SAME
    # name (e.g. sqft_lot/log_sqft_lot both -> "lot size"), so this line
    # could -- and on 30% of real houses, did -- print "lot size (+4.2%),
    # lot size (+3.9%)": the same label twice with two different
    # percentages, while the chart directly above already showed one
    # merged bar. Now goes through the same grouping helper the chart
    # uses, so the two agree.
    driver_parts = [
        f"{g['label']} ({_fmt_pct(g['pct_effect'])})"
        for g in group_top_features_by_label(shap["top_features"])
    ]
    if shap["catchall_n_features"]:
        driver_parts.append(f"{shap['catchall_n_features']} other smaller factors combined ({_fmt_pct(shap['catchall_pct_effect'])})")
    lines.append("The biggest factors behind this number are " + ", ".join(driver_parts) + ".")
    lines.append("")

    t1 = interp["tier1_profile"]
    # AUDIT FIX 2026-09-09 (SA13, final audit): this used to present the
    # category's full MIN-to-MAX price range as what homes "typically sell
    # for" -- for Group 0 Band 4 that rendered as "typically sell for
    # $235,000 to $26,590,000", and Group 0 Band 1's min is $7,800. A full
    # min-max span is the opposite of "typically"; `sorted_prices` (already
    # stored on every tier1_profile for the percentile gauge chart) gives a
    # real p10-p90 range directly, with no pipeline rebuild needed.
    cat_prices = np.asarray(t1["sorted_prices"], dtype=float)
    p10, p90 = (float(v) for v in np.percentile(cat_prices, [10, 90]))
    # 2026-09-18 FIX: was interpolating the raw `combined_category` key
    # (e.g. "Physical Group 0 - Price Band 1") here -- see
    # `step6_interpretation.py`'s comment on `category_label` for why
    # this is now the same digit-free, human-readable name every other
    # surface in the app already uses for this category.
    lines.append(
        f"This home fits the profile of similar properties in our data (we call this group "
        f"'{interp['category_label']}') -- homes like it typically sell for "
        f"{_fmt_money(p10)} to {_fmt_money(p90)}, averaging around "
        f"{_fmt_money(t1['mean_price'])}, based on {t1['count']} similar sales."
    )
    lines.append("")

    n_comps = interp["tier2"]["n_comps_found"]
    tier_label = tier2_tier_phrase(interp["tier2"]["tier2_tier_used"])
    position_phrase = point_position_phrase(interp["market"]["point_position"])
    # AUDIT FIX 2026-09-09 (SA14, final audit): this used to call the comps
    # "recent" -- every sale in the dataset is actually from May-July 2014
    # (`step6_inference_contract.py`'s own docstring already says a house
    # scored today "would sit ~12 years outside anything the model has ever
    # seen"), so the narrative was asserting the opposite of what the
    # codebase already knows. Dropped the false claim rather than replacing
    # it with a specific date that would itself go stale.
    # AUDIT FIX 2026-09-09 (SA31g, final audit): `tier2_tier_phrase()`
    # returns "" via `.get(tier2_tier_used, "")` for any value it doesn't
    # recognize (all 5 current values are mapped, so this is latent
    # today) -- embedding that directly as `f"sales {tier_label}, this"`
    # would render a double space and a floating comma ("sales , this")
    # rather than degrading gracefully. Built conditionally instead.
    tier_clause = f" {tier_label}" if tier_label else ""
    lines.append(
        f"Looking at {n_comps} comparable sales{tier_clause}, this estimate comes in "
        f"{position_phrase} what similar homes have actually sold for."
    )
    if interp["market"]["interval_vs_comps"] == "overlapping":
        # AUDIT FIX 2026-09-09 (SA5, final audit): this used to claim the
        # confidence range is "a bit wider than" the comps' typical spread,
        # but the condition it's gated on ("overlapping") only means the
        # two ranges intersect at all -- it does not test which one is
        # wider, and in 4 of 908 real held-out houses the CQR interval was
        # actually NARROWER than the comps' interquartile range while this
        # sentence still rendered, making it a literally false statement
        # shown to a customer. Reworded to state only what "overlapping"
        # actually establishes.
        lines.append(
            "Worth noting: our confidence range overlaps with the typical spread of nearby sale prices, "
            "so treat that range -- not just the single estimate above -- as the more honest picture of "
            "how much this could vary."
        )
    if interp["market"]["market_divergence"] == "diverges":
        # AUDIT FIX 2026-09-09 (SA8, final audit): matches app.py's identical
        # fix -- this condition fires for 56.1% of the 908 held-out test
        # houses (reproduced directly), so it is not an exception worth
        # singling out; it's the routine result of comparing against nearby
        # comps vs. a category average pulled around by outlier sale prices.
        # Worded to say so plainly instead of implying something notable
        # happened with this particular house.
        lines.append(
            f"For additional context, by a different measure (this home's own price-category average), "
            f"the picture reads as '{interp['market']['tier1_position']}', which differs somewhat from "
            f"the comparable-sales view above. That's common and expected -- both are legitimate ways "
            f"to look at the same home, just measuring different things."
        )
    lines.append("")

    conf = interp["confidence"]
    lines.append(f"Overall, we'd call this a '{conf['level']}' confidence estimate.")
    if conf["reasons"]:
        lines.append("That's mainly because:")
        for r in conf["reasons"]:
            lines.append(f"  - {r}")
    lines.append("")

    if interp.get("imputed_fields"):
        # AUDIT FIX 2026-09-09 (SA10, final audit): this printed raw internal
        # column names (e.g. "sqft_above, waterfront") instead of the
        # human-readable display names step6_confidence.py already uses
        # correctly for the identical fact elsewhere on the same screen
        # (rule 11) -- reused here rather than reimplemented.
        friendly = [FEATURE_DISPLAY_NAMES.get(f, f) for f in interp["imputed_fields"]]
        lines.append(
            f"A quick note: you didn't provide {', '.join(friendly)}, so we filled "
            f"those in using typical values from our data -- treat the estimate as a bit more "
            f"approximate for those details."
        )
        lines.append("")

    # UI_AMENDMENTS.md entry 15: closing appraisal/financial-advice
    # disclaimer line removed from the narrative text itself, per explicit
    # instruction -- the separate graduation-project brand disclaimer
    # (entry 3) already lives in the page footer, unrelated to this line.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
