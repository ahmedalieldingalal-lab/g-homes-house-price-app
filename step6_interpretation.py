"""Phase A, step 6 (the actual deliverable) -- `InterpretationResult` +
`interpret_house(raw_input)`, the single source of truth every downstream
consumer (3 plots, the narrative, the Streamlit UI) reads from. Threads
together every piece built so far in Phase A:

  step6_validation      -> Tier A reject / Decision 3d optional defaults
  step6_inference_contract -> the 35-feature regression row (+ engineered dict)
  step6_cqr_interval    -> predicted_price + 90% CQR interval
  step6_shap_contract   -> multiplicative-percentage SHAP drivers
  step6_classify_house  -> physical_cluster / price_band / combined_category / Tier 1
  step6_market_comparison -> Tier 2 comps + point_position/interval_vs_comps/market_divergence
  step6_confidence      -> the renamed confidence tier
  step6_narrative       -> the no-AI template narrative (Phase B will add an LLM path)

`interpret_house()` never computes the predicted price a second time --
it is computed exactly once (by `step6_cqr_interval.predict_with_interval`)
and threaded through the SHAP explainer, the market comparison, the
confidence check, and the narrative unchanged (the single-source-of-truth
guarantee step 7 calls for; `tests/test_step6_interpretation.py` asserts
the narrative's own price string matches `predicted_price` exactly).
"""
from __future__ import annotations

from dataclasses import dataclass, field
# AUDIT FIX 2026-09-09 (SB21a, final audit): `from typing import Any` was
# unused -- `pyflakes` confirmed it, and confirmed no other unused
# imports in this file.

from step6_inference_contract import build_feature_row, load_inference_tables
from step6_classify_house import classify_house, load_classification_tables, load_classifiers
from step6_cqr_interval import load_cqr_models, predict_with_interval
from step6_shap_contract import build_explainer, explain_house
from step6_market_comparison import get_tier2_comps, compute_market_comparison, load_enriched_dataset
from step6_confidence import compute_confidence
from step6_validation import validate_tier_a, apply_optional_defaults
from step6_narrative import build_template_narrative
from step6_llm_narrative import digit_free_category_label, digit_free_property_type_label

# Raw optional field -> the engineered SHAP feature name(s) it can surface
# as (Decision 3d's imputed-field-in-top-K cross-check, Phase A step 10).
OPTIONAL_FIELD_TO_SHAP_FEATURES = {
    "floors": ["floors"],
    "waterfront": ["waterfront", "premium_outlook_score"],
    "view": ["view", "premium_outlook_score"],
    "condition": ["condition", "condition_x_age"],
    "sqft_lot": ["sqft_lot", "log_sqft_lot", "lot_utilization"],
    "sqft_above": ["sqft_above", "above_ratio"],
    "sqft_basement": ["sqft_basement", "basement_ratio", "has_basement"],
    "yr_renovated": ["yr_renovated", "was_renovated", "years_since_renovation", "renovation_recency_ratio"],
}


@dataclass
class InterpretationResult:
    ok: bool
    rejection_reasons: list[str] = field(default_factory=list)

    predicted_price: float | None = None
    cqr_lo: float | None = None
    cqr_hi: float | None = None
    relative_width: float | None = None

    physical_cluster: int | None = None
    price_band: int | None = None
    combined_category: str | None = None
    tier1_profile: dict | None = None

    shap: dict | None = None

    tier2: dict | None = None
    market: dict | None = None

    confidence: dict | None = None
    flags: dict = field(default_factory=dict)
    imputed_fields: list[str] = field(default_factory=list)

    narrative: str | None = None
    narrative_source: str = "template"
    degradation_reasons: list[str] = field(default_factory=list)
    # AUDIT FIX 2026-09-09 (SB21d, final audit): `app.py` currently
    # displays each chart from `build_all_plots()`'s OWN returned dict
    # (`plot_paths = build_all_plots(result, ...)`), never from this
    # field, even though `build_all_plots()` also writes the identical
    # dict here (confirmed: `grep -c "result\.plot_paths" app.py` -> 0).
    # Left in place rather than removed: it is this dataclass's own
    # documented, tested contract (`tests/test_step6_plots.py` asserts
    # `result.plot_paths == paths` for every house) for any caller that
    # only holds the `InterpretationResult` and not the separate return
    # value -- e.g. code that stores/passes the result object without
    # also threading the dict alongside it. `app.py` simply happens to
    # already have the more convenient local variable, not evidence this
    # field is unused everywhere.
    plot_paths: dict = field(default_factory=dict)
    # UI_AMENDMENTS.md entry 26: this house's percentile rank (0-100) within
    # its Tier 1 category's price distribution, computed once by
    # build_all_plots() (plot_tier1_range's percentile gauge) and stored
    # here so app.py's caption reuses the exact same number instead of
    # recomputing it independently (project rule 11: no duplicated lookups).
    tier1_percentile_rank: float | None = None
    # 2026-09-18 ADDITION (Item 5 -- independent "Property Type" badge):
    # the digit-free property-type name ALONE (e.g. "Untouched Classics"),
    # computed once below via `digit_free_property_type_label()` and
    # stored here so app.py's new badge widget reads the same value the
    # narrative's "we call this group '...'" sentence is built from,
    # rather than re-deriving it a second, independently-maintained way
    # (rule 11).
    property_type_label: str | None = None


class InterpretationContext:
    """Loads every model/table exactly once (they're each individually
    cached by their own module already, but this bundles the load calls
    so a caller -- e.g. a Streamlit app -- pays the cost a single time at
    startup, not on every `interpret_house()` call)."""

    def __init__(self):
        self.inference_tables = load_inference_tables()
        self.classification_tables = load_classification_tables()
        self.classifier_a, self.classifier_b = load_classifiers()
        self.cqr_models = load_cqr_models()
        self.enriched_df = load_enriched_dataset()
        self.shap_explainer = build_explainer(self.cqr_models["point_model"])


_DEFAULT_CONTEXT: InterpretationContext | None = None


def _default_context() -> InterpretationContext:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = InterpretationContext()
    return _DEFAULT_CONTEXT


def interpret_house(raw_input: dict, ctx: InterpretationContext | None = None,
                     sale_year: int | None = None, sale_month: int | None = None,
                     gemini_api_key: str | None = None) -> InterpretationResult:
    ctx = ctx if ctx is not None else _default_context()

    # --- Tier A: hard reject on the RAW input, before defaults are filled,
    # so a missing critical field is still caught (apply_optional_defaults
    # only ever fills OPTIONAL fields). Zero downstream calls if rejected.
    rejection_reasons = validate_tier_a(raw_input)
    if rejection_reasons:
        return InterpretationResult(ok=False, rejection_reasons=rejection_reasons)

    filled_input, imputed_fields = apply_optional_defaults(raw_input)

    # --- single feature-engineering pass, reused by regression + classification ---
    X_row, contract_flags, engineered = build_feature_row(
        filled_input, tables=ctx.inference_tables, sale_year=sale_year, sale_month=sale_month,
    )

    # --- price + interval, computed EXACTLY ONCE ---
    pred = predict_with_interval(X_row, models=ctx.cqr_models)

    # --- SHAP, built on the SAME X_row used for the prediction above ---
    shap_result = explain_house(X_row, ctx.cqr_models["point_model"], explainer=ctx.shap_explainer, top_k=8)
    if not shap_result["identity_holds"]:
        # Should be unreachable (verified on 50 real houses in
        # tests/test_step6_shap_contract.py) -- if it ever fires, the SHAP
        # breakdown is untrustworthy and must not be shown as fact.
        #
        # AUDIT FIX 2026-09-09 (SB14, final audit): `identity_holds` is
        # `additive_log_check < 1e-6 AND full_identity_diff < 1e-6 AND
        # grouped_identity_diff < 1e-6` (step6_shap_contract.py:99) -- all
        # three can independently fail it, but this message used to report
        # only the last two, omitting `additive_log_check_diff` -- the ONE
        # that actually fires on the failure mode this check exists to
        # catch (an explainer built on the wrong model): a mismatch test
        # (feeding `q_lo_model` to an explainer built on `point_model`)
        # measured `additive_log_check_diff = 0.4178` while the two
        # reported diffs told the debugger nothing about it. All three are
        # in the message now.
        raise AssertionError(
            f"SHAP multiplicative identity failed for this house (diffs: "
            f"additive_log_check={shap_result['additive_log_check_diff']}, "
            f"full_identity={shap_result['full_identity_diff']}, "
            f"grouped_identity={shap_result['grouped_identity_diff']}) -- refusing to "
            f"report an explanation that doesn't reconstruct the predicted price."
        )

    # --- classification (physical_cluster / price_band / Tier 1) ---
    cls = classify_house(engineered, tables=ctx.inference_tables, class_tables=ctx.classification_tables,
                          classifier_a=ctx.classifier_a, classifier_b=ctx.classifier_b)

    # --- Tier 2 comps + market comparison ---
    tier2 = get_tier2_comps(engineered, cls["combined_category"], df=ctx.enriched_df)
    market = compute_market_comparison(pred["predicted_price"], pred["cqr_lo"], pred["cqr_hi"],
                                        tier2, cls["tier1_profile"])

    # --- imputed-field-in-top-K cross-check (Phase A step 10) ---
    top_k_features = {f["feature"] for f in shap_result["top_features"]}
    imputed_in_top_k = [
        raw_field for raw_field in imputed_fields
        if top_k_features & set(OPTIONAL_FIELD_TO_SHAP_FEATURES.get(raw_field, [raw_field]))
    ]

    all_flags = {**contract_flags, **cls["flags"], **tier2["flags"]}

    confidence = compute_confidence(
        engineered_flags=all_flags, tier2=tier2, relative_width=pred["relative_width"],
        predicted_price=pred["predicted_price"], imputed_fields_in_top_k=imputed_in_top_k,
    )

    result = InterpretationResult(
        ok=True,
        predicted_price=pred["predicted_price"], cqr_lo=pred["cqr_lo"], cqr_hi=pred["cqr_hi"],
        relative_width=pred["relative_width"],
        physical_cluster=cls["physical_cluster"], price_band=cls["price_band"],
        combined_category=cls["combined_category"], tier1_profile=cls["tier1_profile"],
        shap=shap_result, tier2=tier2, market=market,
        confidence=confidence, flags=all_flags, imputed_fields=imputed_in_top_k,
        narrative_source="template",
    )

    # AUDIT FIX 2026-09-09 (SA10, final audit): this used to pass the FULL
    # `imputed_fields` list (every optional field the user left blank) to
    # the template narrative, while `InterpretationResult.imputed_fields`
    # (what the LLM path reads) is `imputed_in_top_k` -- only the imputed
    # fields that actually showed up among the SHAP top-K drivers. Same
    # house, same screen position, two different fact lists depending only
    # on whether Gemini answered. Both paths now read the same top-K-
    # filtered list, matching what step6_confidence.py already correctly
    # renders elsewhere on the same screen.
    # 2026-09-18 FIX: the template narrative used to interpolate
    # `combined_category` (a raw, digit-bearing internal key like
    # "Physical Group 0 - Price Band 1") straight into its first
    # customer-facing sentence -- the one user-visible spot in the whole
    # app that still leaked this key, discovered while visually
    # confirming the cluster-rename change below reached every real
    # display surface. Every other surface (the tier1 chart title, the
    # market-comparison paragraph, the LLM prompt) already goes through
    # this same `digit_free_category_label()` lookup (rule 11) for the
    # canonical, human-readable name -- the template narrative is now
    # the same, not a fourth, disagreeing way of describing the category.
    category_label = digit_free_category_label(result.combined_category, ctx.classification_tables)
    # 2026-09-18 ADDITION (Item 5): same canonical lookup, just without the
    # price-tier suffix -- see `digit_free_property_type_label()`'s own
    # docstring. Stored on `result` (not just `narrative_input`) since
    # app.py's new badge widget reads `result.property_type_label`
    # directly, the same object every other post-prediction widget reads
    # from, rather than threading a second dict through to it.
    result.property_type_label = digit_free_property_type_label(result.combined_category, ctx.classification_tables)
    narrative_input = {
        "predicted_price": result.predicted_price, "cqr_lo": result.cqr_lo, "cqr_hi": result.cqr_hi,
        "shap": result.shap, "combined_category": result.combined_category, "category_label": category_label,
        "tier1_profile": result.tier1_profile,
        "tier2": result.tier2, "market": result.market, "confidence": result.confidence,
        "imputed_fields": imputed_in_top_k,
    }

    # --- Phase B/C: attempt the Gemini-assisted narrative; ANY failure
    # (no key, network error, malformed/invalid response) falls back to
    # the plain template untouched -- this is what makes the app fully
    # functional and gradeable with no API key configured at all.
    from step6_llm_narrative import call_gemini_narrative, assemble_llm_narrative
    llm_response = call_gemini_narrative(result, api_key=gemini_api_key)
    if llm_response["fields"] is not None:
        try:
            result.narrative = assemble_llm_narrative(result, llm_response["fields"])
            result.narrative_source = "llm"
        except Exception as e:  # noqa: BLE001 -- assembly must never crash the app
            result.narrative = build_template_narrative(narrative_input)
            result.narrative_source = "template"
            result.degradation_reasons = [f"LLM narrative assembly failed: {e}"]
    else:
        result.narrative = build_template_narrative(narrative_input)
        result.narrative_source = "template"
        result.degradation_reasons = llm_response["degradation_reasons"]

    return result
