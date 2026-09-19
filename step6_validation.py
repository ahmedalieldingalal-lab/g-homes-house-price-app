"""Phase A -- Decision 3c/3d input validation (Tier A hard-reject +
Decision 3d's critical/optional field split), the first thing
`interpret_house()` runs before any model, SHAP, or RAG call.

Tier A (hard reject, ZERO downstream calls -- no SHAP/classifier/Gemini
work, so adversarial input never burns model time or API quota):
physically-impossible single-field values and relationship-nonsense that
a single field's bound would miss (e.g. a house whose bedroom count is
plausible alone but implies ~50 sqft per bedroom). These bounds are
DELIBERATELY wider/more permissive than the real training range --
"outside the range this model has actually seen" is Tier B's job (a soft
flag that still produces a prediction), not Tier A's. Tier A only fires
on things no real house could be, regardless of whether this particular
model happened to train on one.

2026-09-18 UPDATE (PACKAGING_TODO item 7): the Tier B layer referenced
above is now real, not aspirational -- see
`step6_inference_contract.TRAINING_OBSERVED_RANGES` (fixed, train-split-
only per-field bounds, hardcoded and documented the same way
`step6_confidence.TRAINING_PRICE_P1`/`P99` are) and
`step6_confidence.compute_confidence()`'s matching combined-caveat
sentence. Before this date, only house age and the two post-2014 year
flags actually disclosed anything outside Tier A's own (deliberately
wide) bounds; a submission like 20 bedrooms, a 5,000,000 sqft lot, or a
90% basement ratio -- all real values these bounds accept on purpose --
got a confident price with zero caveat. It doesn't cover every field
Tier A widens (waterfront/view/condition don't need it -- their training
min/max already equals Tier A's bound exactly), but it covers every
field where a real gap existed.

Decision 3d's critical/optional split (quoted directly from the decision
log rather than re-derived): critical = living area, bedrooms, bathrooms,
location (must be provided, hard reject if missing, never imputed);
optional = everything with a smaller typical price effect (may be left
blank, filled with a training-derived default, disclosed as an assumption
with a lower-confidence flag). `yr_built` is treated as critical here too
(house_age is a first-order price driver, same tier as living area) even
though the decision log's own examples ("e.g. renovation year, view")
only named smaller-effect fields explicitly.
"""
from __future__ import annotations

CRITICAL_FIELDS = ["sqft_living", "bedrooms", "bathrooms", "yr_built", "city", "statezip"]

# Training-derived defaults for OPTIONAL fields (Decision 3d) -- medians/
# modes of the TRAINING SPLIT of consolidated_output/enriched_dataset.csv's
# raw columns (`consolidated_pipeline.group_train_test_split`'s train_df,
# 3,641 of the 4,549 rows -- never the held-out test rows, per this
# project's standing leak-safety rule), computed once and hardcoded here
# rather than read live, since these are fixed, reviewable constants, not
# something that should silently drift if the dataset file changes on
# disk. (floors/waterfront/view/condition medians are the dataset's
# actual modal/median values; sqft_above/sqft_basement/yr_renovated
# likewise.)
#
# AUDIT FIX 2026-09-09 (SB11, final audit): `sqft_lot` used to be 7683.0
# with this same docstring claiming it came from enriched_dataset.csv --
# neither was true. 7683.0 is actually the median of the RAW, UNCLEANED
# `data.csv` (4,600 rows, before the cleaning pipeline runs at all).
# Verified: `pd.read_csv('data.csv').sqft_lot.median()` -> 7683.0 exactly.
# The enriched (post-clean, pre-split) median is 7680.0; the correct,
# leak-safe TRAIN-SPLIT-ONLY median -- matching every other statistic
# this final audit corrected to be train-split-only (SA7's confidence
# percentiles, this project's `TRAINING_PRICE_P1`/`P99`) -- is 7650.0,
# now used below. Numerical impact of the old value was negligible
# (~4e-4 on `log_sqft_lot`) but the correct, leak-safe number costs
# nothing to use instead of a wrong one from an entirely different file.
OPTIONAL_FIELD_DEFAULTS = {
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_lot": 7650.0,
    "sqft_above": None,   # derived: sqft_living - sqft_basement default
    "sqft_basement": 0,
    "yr_renovated": 0,
}

# Tier A hard bounds -- physically-impossible thresholds, wider than the
# real training range on purpose (see module docstring).
TIER_A_BOUNDS = {
    "bedrooms": (1, 20),
    "bathrooms": (0.5, 15.0),  # both float -- Streamlit's number_input requires value/min/max/step to share one type
    "floors": (1.0, 5.0),
    "waterfront": (0, 1),
    "view": (0, 4),
    "condition": (1, 5),
    "sqft_living": (100, 20000),
    "sqft_lot": (200, 5_000_000),
    "sqft_above": (0, 20000),
    "sqft_basement": (0, 10000),
    # 2026-09-17 DESIGN CHANGE: yr_built's upper bound was 2014 (the pinned
    # inference sale_year) -- any house built after training's most recent
    # year was a hard Tier A reject, with no prediction offered at all. Per
    # user decision, this is now a wide-but-real outer bound instead: Tier A
    # only catches a value no real house could ever have (a typo like 2200
    # or 9999), while "built after 2014" itself is handled as a Tier B soft
    # flag (see step6_inference_contract.py's `yr_built_after_training_
    # window` flag and step6_confidence.py's matching disclosure reason) --
    # the house still gets a real prediction, just with an explicit,
    # separately-named caveat. 2026 is the upper bound because a house
    # cannot really have been built after the actual present year; this is
    # a fixed constant, not computed from the system clock, and should be
    # revisited if this app is still in use in a later calendar year.
    "yr_built": (1800, 2026),
    # 2026-09-17 DESIGN CHANGE (same reasoning as yr_built above, raised in
    # the same pass after an Opus adversarial review flagged the gap this
    # left open): this used to stay capped at 2014 even after yr_built's
    # bound moved to 2026, so a house built in 2020 and genuinely renovated
    # in 2023 was inexpressible -- the renovation year would fail the
    # existing "yr_renovated < yr_built" check's inverse (it isn't less
    # than yr_built, so it wasn't rejected outright), but a value above 2014
    # was simply unreachable in the Streamlit widget, whose max_value reads
    # straight from this bound. Raised to match yr_built's outer limit for
    # the same reason: 2026 is the real present year, a fixed constant, not
    # computed from the system clock. `step6_inference_contract.py` clamps
    # any renovation year past the pinned sale_year (2014) to
    # years_since_renovation=0 (i.e. priced as if renovated in 2014, the
    # most recent the model has ever seen) and raises a matching
    # `yr_renovated_after_training_window` flag for the confidence layer to
    # disclose by name -- the same pattern as yr_built's own flag.
    "yr_renovated": (0, 2026),
}
MIN_SQFT_PER_BEDROOM = 100  # well below training's observed min (163.3) -- see PROJECT_TODO.md
# 2026-09-18 ADDITION (validation-logic audit, prompted by the user
# personally testing the app and finding it accepted a renovation year
# before the year built): `bathrooms` had no partner check at all, even
# though `bathrooms`' own Tier A bound goes up to 15.0 -- e.g.
# sqft_living=100 (the sqft_living floor), bathrooms=15.0 passed Tier A
# with zero rejections (6.7 sqft/bathroom, below the footprint of a single
# toilet). Verified against real training data: min sqft_living/bathrooms
# across all 4,549 rows is 300.0 -- this floor is a 10x margin below that,
# deliberately wide per this module's own "wider than training on purpose"
# rule, so it only ever fires on adversarial/nonsense input, never a real
# unusual house.
MIN_SQFT_PER_BATHROOM = 30


def apply_optional_defaults(raw_house: dict) -> tuple[dict, list[str]]:
    """Fills any missing OPTIONAL field with its training-derived default.
    Returns (filled_house, imputed_field_names) -- the caller discloses
    `imputed_field_names` in the narrative/confidence layer (Decision 3d /
    Phase A step 10), never silently.

    sqft_above/sqft_basement get special handling (UI_AMENDMENTS.md entry
    1, fixed here): sqft_living = sqft_above + sqft_basement is an exact
    training invariant, so whenever exactly one of the two is missing, it
    is reconstructed EXACTLY from the other two real values -- that's
    arithmetic on real inputs, not a guess, so it is deliberately NOT
    added to `imputed_field_names`. Only when BOTH are missing does
    reconstructing either one require an actual assumption (no basement),
    and only then are they flagged as imputed. Previously, sqft_above was
    marked "imputed" the instant it was `None` in the raw submission --
    which is always, for a UI that only ever collects sqft_basement
    directly -- even on the large majority of submissions where it was
    exact arithmetic from real inputs, overstating uncertainty that
    wasn't really there.
    """
    h = dict(raw_house)
    imputed: list[str] = []

    above_missing = h.get("sqft_above") is None
    basement_missing = h.get("sqft_basement") is None

    if above_missing and basement_missing:
        # Neither given -- assume no basement (the training default), then
        # derive sqft_above from that assumption. Both rest on a real guess.
        h["sqft_basement"] = OPTIONAL_FIELD_DEFAULTS["sqft_basement"]
        h["sqft_above"] = h.get("sqft_living", 0) - h["sqft_basement"]
        imputed.append("sqft_above")
        imputed.append("sqft_basement")
    elif above_missing:
        # sqft_basement is real -- sqft_above is exact arithmetic, not a guess.
        h["sqft_above"] = h.get("sqft_living", 0) - h["sqft_basement"]
    elif basement_missing:
        # sqft_above is real -- sqft_basement is exact arithmetic, not a guess.
        h["sqft_basement"] = h.get("sqft_living", 0) - h["sqft_above"]

    for field, default in OPTIONAL_FIELD_DEFAULTS.items():
        if field in ("sqft_above", "sqft_basement"):
            continue  # handled above with exact-derivation logic
        if h.get(field) is None:
            imputed.append(field)
            h[field] = default

    return h, imputed


def validate_tier_a(raw_house: dict) -> list[str]:
    """Returns a list of rejection reasons (empty = passes Tier A). Each
    reason is a specific, educational message naming the real bound
    exceeded, per Decision 3c -- never a generic "invalid input" error."""
    reasons = []

    for field in CRITICAL_FIELDS:
        if raw_house.get(field) in (None, ""):
            reasons.append(
                f"'{field}' is required (Decision 3d: living area, bedrooms, bathrooms, "
                f"location, and year built materially affect price and cannot be guessed)."
            )
    if reasons:
        return reasons  # can't check bounds on fields that were never provided

    for field, (lo, hi) in TIER_A_BOUNDS.items():
        value = raw_house.get(field)
        if value is None:
            continue  # optional field, not yet defaulted -- checked after apply_optional_defaults if desired
        if not (lo <= value <= hi):
            reasons.append(
                f"'{field}'={value} is outside the physically plausible range "
                f"[{lo}, {hi}] this model can ever meaningfully score."
            )

    sqft_living, bedrooms = raw_house.get("sqft_living"), raw_house.get("bedrooms")
    if sqft_living is not None and bedrooms and bedrooms > 0:
        ratio = sqft_living / bedrooms
        if ratio < MIN_SQFT_PER_BEDROOM:
            reasons.append(
                f"sqft_living/bedrooms = {ratio:.0f} sqft per bedroom, below the "
                f"{MIN_SQFT_PER_BEDROOM} sqft/bedroom floor no real house falls under -- "
                f"check the bedroom count and living area entered."
            )

    # 2026-09-18 ADDITION (validation-logic audit): `bathrooms` never had the
    # partner check `bedrooms` gets just above, even though `bathrooms`'
    # own Tier A bound goes up to 15.0 -- see MIN_SQFT_PER_BATHROOM's own
    # comment for the real-data floor this is checked against.
    bathrooms = raw_house.get("bathrooms")
    if sqft_living is not None and bathrooms and bathrooms > 0:
        bath_ratio = sqft_living / bathrooms
        if bath_ratio < MIN_SQFT_PER_BATHROOM:
            reasons.append(
                f"sqft_living/bathrooms = {bath_ratio:.0f} sqft per bathroom, below the "
                f"{MIN_SQFT_PER_BATHROOM} sqft/bathroom floor no real house falls under -- "
                f"check the bathroom count and living area entered."
            )

    sqft_above, sqft_basement = raw_house.get("sqft_above"), raw_house.get("sqft_basement")
    if sqft_living is not None and sqft_above is not None and sqft_basement is not None:
        if abs((sqft_above + sqft_basement) - sqft_living) > 1:
            reasons.append(
                f"sqft_above ({sqft_above}) + sqft_basement ({sqft_basement}) must equal "
                f"sqft_living ({sqft_living}) -- every real record in the training data "
                f"satisfies this exactly."
            )
        # 2026-09-18 ADDITION (validation-logic audit): the sum-equality
        # check just above can be satisfied by an impossible split (e.g.
        # sqft_above=0, sqft_basement=sqft_living -- sums correctly, but
        # asserts zero above-grade floor area on a house that
        # simultaneously claims `floors >= 1.0`, Tier A's own enforced
        # bound). A real house always has SOME above-grade area, however
        # small (training's own observed minimum is 370 sqft) -- this is
        # the "all three explicitly provided" counterpart to the two
        # one-sided checks below, which only run when one of the two is
        # left for `apply_optional_defaults` to derive.
        if sqft_above <= 0:
            reasons.append(
                f"sqft_above ({sqft_above}) must be greater than 0 -- every real house has "
                f"some above-ground floor area (this app already requires floors >= 1)."
            )

    # AUDIT FIX 2026-09-09 (SB1, final audit): the check above only fires
    # when `sqft_above` is already present on the RAW submission -- but the
    # deployed app (app.py) always sends `sqft_above=None` (it's derived
    # later, in apply_optional_defaults, from sqft_living - sqft_basement),
    # so that check was dead code in production: a basement bigger than the
    # house itself (e.g. sqft_living=100, sqft_basement=10000) passed Tier A
    # with zero rejection reasons, then silently produced sqft_above=-9900
    # downstream, and the app returned a confident price for a physically
    # impossible house. Fixed by checking the one relationship Tier A CAN
    # verify directly from the raw, always-provided fields (sqft_living and
    # sqft_basement -- app.py never sends sqft_basement as None, only 0 or a
    # real widget value), independent of whether sqft_above has been derived
    # yet. This does not replace the check above (which still catches a
    # caller that supplies all three fields directly, e.g. a future API
    # consumer) -- it closes the gap for the one caller (the deployed app)
    # that never does.
    #
    # 2026-09-18 FIX (validation-logic audit, GAP 1): this used to only
    # reject `sqft_basement > sqft_living` (strictly greater) -- EQUALITY
    # passed. A basement exactly equal to the total living area derives
    # sqft_above=0 in `apply_optional_defaults`, the same impossible
    # "zero above-grade area" case the check above catches for callers who
    # supply all three fields -- but this was the deployed app's own path
    # (app.py never sends sqft_above), so it reached a confident price with
    # ZERO flags, not even a disclosed one, on a house with no above-ground
    # floor at all. Verified reachable from the UI: tick "Has a basement",
    # set its area equal to the living area. `>=` closes it.
    if sqft_living is not None and sqft_basement is not None:
        if sqft_basement >= sqft_living:
            reasons.append(
                f"sqft_basement ({sqft_basement}) cannot be greater than or equal to "
                f"sqft_living ({sqft_living}) -- a basement smaller than the total living "
                f"area is required, since every real house has some above-ground floor area."
            )

    # 2026-09-18 ADDITION (validation-logic audit, GAP 2): the exact mirror
    # of the SB1 fix just above, for the OTHER one-sided case -- a caller
    # (not the deployed app today, which never sends `sqft_above`, but a
    # future API consumer per SB1's own comment) who supplies `sqft_above`
    # without `sqft_basement`. `apply_optional_defaults` would then derive
    # sqft_basement = sqft_living - sqft_above, which goes NEGATIVE the
    # moment sqft_above exceeds sqft_living -- the same class of "confident
    # price for a physically impossible house" SB1 already called out, just
    # on the other field. sqft_above == sqft_living is fine (it means no
    # basement, sqft_basement derives to exactly 0) -- only strictly
    # greater is impossible, so this is `>`, not `>=`.
    if sqft_living is not None and sqft_above is not None:
        if sqft_above > sqft_living:
            reasons.append(
                f"sqft_above ({sqft_above}) cannot exceed sqft_living ({sqft_living}) -- "
                f"the above-ground area is never larger than the home's total living area."
            )

    # 2026-09-18 DESIGN CHANGE (validation-logic audit, prompted by the user
    # personally testing the app and finding it accepted a renovation year
    # before the year built): this used to be a deliberate NON-reject --
    # `step6_inference_contract.py` silently treated the house as
    # "not renovated" instead (`bad_renovation_date_ignored` flag), on the
    # reasoning that training itself does the same thing for 193 messy
    # historical rows (`consolidated_pipeline.add_new_features`'s own
    # data-quality fix). That reasoning is right for TRAINING data (those
    # 193 houses were sold years ago; there is no one left to ask, so
    # best-effort cleaning is the only option) but wrong for a LIVE
    # submission: a person filling in this form right now can simply be
    # asked to fix their own input, and "yr_renovated < yr_built" is
    # exactly the "relationship-nonsense a single field's bound would
    # miss" this module's own docstring says Tier A exists to catch --
    # not a Tier B caveat. The old behavior was also the worst of both
    # worlds in practice: it silently discarded a field value the user had
    # to actively opt into providing (ticking "Has been renovated"), and
    # disclosed it only as one line buried in a "Good to know" bullet list
    # ("an inconsistent renovation date was ignored") rather than telling
    # them their input didn't make sense. `consolidated_pipeline.py`'s
    # training-time cleaning fix is UNCHANGED and untouched by this --
    # that's a separate, already-shipped decision about historical data,
    # not about what a live submission should be allowed to say today.
    #
    # `app.py`'s "Renovation year" widget is updated in the same pass to
    # bind its own min_value/default to the currently-entered `yr_built`,
    # so the UI can never even offer a value this new check would reject
    # (the same invariant every other widget in that file already
    # maintains) -- this reject is a backstop for a caller that bypasses
    # the UI (a future API consumer), not a trap sprung on a form user.
    yr_built, yr_renovated = raw_house.get("yr_built"), raw_house.get("yr_renovated")
    if yr_built is not None and yr_renovated:
        if 0 < yr_renovated < yr_built:
            reasons.append(
                f"'yr_renovated'={yr_renovated} is before 'yr_built'={yr_built} -- a home "
                f"cannot be renovated before it was built. Check the year built and "
                f"renovation year entered."
            )

    return reasons
