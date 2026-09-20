"""Step 7 (Deployment) -- the graduation project's Streamlit app.

Per the assignment PDF: "Build a Streamlit app or API + web application:
input house details, classify house into a category, predict price, LLM
interprets the result, display outputs in a user-friendly interface."

This file is a thin UI layer only -- every real computation (validation,
classification, price + interval, SHAP, comps, confidence, narrative, and
the 3 plots) is the exact same tested code from steps 1-7's Phase A-E
build, reached through the single `interpret_house()` entry point. Nothing
here re-implements or re-derives anything already built and verified
elsewhere (project rule 11: one implementation per piece of logic).

Gemini API key: read from the environment, NOT hardcoded.

FINAL AUDIT PANEL FIX 2026-09-10 (B3, found independently by two auditors):
this file previously contained a live Gemini API key as a string literal,
per an earlier decision (recorded in PROJECT_TODO.md's Step 7 log) to make
the AI narrative work with zero setup for anyone who ran the app. That
tradeoff is not acceptable for a submitted artifact: the key shipped inside
every copy of this file -- the submission archive, the deployed folder, and
anything published publicly. The old key has been removed from the source
and must be treated as compromised and revoked at Google AI Studio.

The app needs no key to work. With `GEMINI_API_KEY` unset it runs exactly
as before and renders the deterministic template narrative -- the
"Machine-Statistics (generated without AI)" path, which is the project's
designed spine and satisfies the deployment requirement on its own. Setting
the variable simply re-enables the LLM prose layer on top:

    export GEMINI_API_KEY=your-key-here     # macOS/Linux
    setx GEMINI_API_KEY your-key-here       # Windows (new shell after)
    streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from step6_interpretation import interpret_house, InterpretationContext, InterpretationResult
from step6_cache import canonical_cache_key, SimpleCache
from step6_plots import build_all_plots
# AUDIT FIX 2026-09-09 (SA1, final audit): the label-collision merge helper
# moved from step6_plots (`_group_top_features_by_label`, private) to
# step6_llm_narrative (`group_top_features_by_label`, public) so the chart,
# this widget, and both narrative paths all share one implementation
# instead of the chart being the only one that merged label collisions.
from step6_llm_narrative import group_top_features_by_label
from step6_llm_narrative import compute_fired_flag_labels
from step6_validation import OPTIONAL_FIELD_DEFAULTS, TIER_A_BOUNDS

# FINAL AUDIT PANEL FIX 2026-09-10 (B3): was a hardcoded literal key -- see
# this module's docstring. Empty string (not None) is deliberate: every
# downstream use below is a truthiness test (`bool(GEMINI_API_KEY)`,
# `GEMINI_API_KEY or None`), so an unset environment degrades to the
# template narrative path exactly as a falsy key always did.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

# AUDIT FIX 2026-09-09 (SB4, final audit): these were bare cwd-relative
# paths -- `streamlit run /abs/path/app.py` from any directory other than
# the repo root raised a raw `FileNotFoundError` before this even reached
# `interpret_house()`, since app.py itself couldn't find its own dataset.
# Anchored to this file's own location, matching every other step6_*.py
# module's identical fix.
REPO_ROOT = Path(__file__).resolve().parent
ENRICHED_DATASET_PATH = REPO_ROOT / "consolidated_output" / "enriched_dataset.csv"
PLOTS_DIR = REPO_ROOT / "app_plots"

# PACKAGING_TODO.md, post-delivery Amendment 1 (2026-09-20): the hero
# banner's photos and the brand fonts used to be hotlinked from
# images.pexels.com / fonts.googleapis.com -- both real, working features
# (verified rendering fine when reachable), but a live external dependency
# on first paint, which showed as a slow/blank banner on a first-time or
# slow connection (the trigger for this amendment). Bundled locally here
# instead: same photos (Pexels License: free for commercial/personal use,
# no attribution required) and same three Google Fonts (all OFL-licensed
# open source, pulled from Google's own canonical google/fonts source
# repo), shipped as local files under assets/ so the app has no runtime
# dependency on either host.
ASSETS_DIR = REPO_ROOT / "assets"
PHOTOS_DIR = ASSETS_DIR / "photos"
FONTS_DIR = ASSETS_DIR / "fonts"

# AUDIT FIX 2026-09-09 (SB15, final audit): this dict and the separate
# `{"well-supported": 90, "typical": 65, "limited evidence": 35}` fill-
# percentage dict inside `render_confidence_ring()` used to be two
# independent hardcoded copies of `step6_confidence`'s 3-value level enum
# (a third copy of the same enum lives in `step6_confidence.py` itself,
# as the return values of `compute_confidence()`). Merged into one dict
# here so there is only one place in app.py that lists the 3 levels and
# their display metadata together.
CONFIDENCE_DISPLAY = {
    "well-supported": {"color": "#22c55e", "fill_pct": 90},
    "typical": {"color": "#3b82f6", "fill_pct": 65},
    "limited evidence": {"color": "#f59e0b", "fill_pct": 35},
}
_CONFIDENCE_DISPLAY_FALLBACK = {"color": "#94a3b8", "fill_pct": 50}  # neutral grey -- see render_confidence_ring

# Header banner photos -- free-to-use stock photography (Pexels License: free
# for commercial/personal use, no attribution required). Picked by hand from
# Pexels' "house" / "modern house exterior" search results, checking each
# photo's own description first so the banner only shows real house
# exteriors (not interiors, dollhouses, or unrelated results that a plain
# keyword search can turn up). Bundled locally as of Amendment 1 (see
# ASSETS_DIR above) -- these are now local filenames under
# assets/photos/, not remote URLs; `_header_photo_data_uris()` below reads
# and base64-encodes them at runtime so `render_hero_header()` can keep
# using a plain `src="..."` <img> tag unchanged.
HEADER_PHOTOS = [
    "header_1.jpg",
    "header_2.jpg",
    "header_3.jpg",
    "header_4.jpg",
    "header_5.jpg",
    "header_6.jpg",
]


@st.cache_resource(show_spinner=False)
def _header_photo_data_uris(filenames: tuple[str, ...]) -> tuple[str, ...]:
    """Reads the local header photo files once (per session process,
    thanks to st.cache_resource -- Streamlit reruns this whole script on
    every interaction, and re-reading + re-encoding 6 images from disk on
    every rerun would be wasted work) and returns them as base64 data
    URIs, so `render_hero_header()` can embed them directly into its
    isolated iframe's HTML with a plain <img src="..."> -- the same
    approach already used for the local fonts below, and necessary for
    the same reason: components.html() renders into a sandboxed iframe
    with no access to Streamlit's own static file serving."""
    uris = []
    for name in filenames:
        data = (PHOTOS_DIR / name).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        uris.append(f"data:image/jpeg;base64,{b64}")
    return tuple(uris)


# Local, subsetted (Basic Latin + the handful of extra punctuation marks
# actually used: em/en dash, curly quotes, ellipsis, the gold "diamond"
# bullet) replacements for the same three Google Fonts previously pulled
# live from fonts.googleapis.com. Sourced from Google's own canonical
# google/fonts repo (OFL-licensed, free for any use) and subset+recompressed
# to .woff2 with fontTools -- ~192KB total across all 4 files, versus
# ~1.9MB for the unsubsetted originals, keeping the same footprint the
# live Google Fonts CDN would have served for this same character set.
# Playfair Display and Inter ship as variable fonts upstream, so one
# @font-face per style with a font-weight *range* covers every weight this
# app uses (600/700 normal + 500 italic for Playfair Display; 400/500/600
# for Inter) from a single file each -- simpler than one @font-face per
# weight and functionally identical to what the Google Fonts CSS2 API
# would have served for the same requested weights.
_LOCAL_FONT_FACES = (
    # (font-family, font-style, css "font-weight" value, filename)
    ("Playfair Display", "normal", "100 900", "playfair-display-wght.woff2"),
    ("Playfair Display", "italic", "100 900", "playfair-display-italic-wght.woff2"),
    ("Inter", "normal", "100 900", "inter-opsz-wght.woff2"),
    ("Great Vibes", "normal", "400", "great-vibes-regular.woff2"),
)


@st.cache_resource(show_spinner=False)
def _local_font_faces_css(family_names: tuple[str, ...]) -> str:
    """Builds @font-face CSS with base64-embedded local font files for the
    given family names (a subset of _LOCAL_FONT_FACES), replacing what
    used to be a live `@import url('https://fonts.googleapis.com/...')`.
    Takes a subset rather than always returning all 4 families so each of
    the 3 injection sites below only pays for the fonts it actually uses
    (e.g. the price headline iframe only ever needed Playfair Display,
    never Inter or Great Vibes) -- unchanged from each site's original,
    already-scoped @import. Cached like _header_photo_data_uris() above,
    for the same reason (avoid re-reading+re-encoding on every Streamlit
    rerun)."""
    rules = []
    for family, style, weight, filename in _LOCAL_FONT_FACES:
        if family not in family_names:
            continue
        data = (FONTS_DIR / filename).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        rules.append(
            f"@font-face {{ font-family: '{family}'; font-style: {style}; "
            f"font-weight: {weight}; font-display: swap; "
            f"src: url(data:font/woff2;base64,{b64}) format('woff2'); }}"
        )
    return "\n".join(rules)

# Real, previously-measured model accuracy numbers (regression_output/
# regression_report.md) -- one constant reused by both the hero header's
# honest accuracy line and the Market Insights tab's fuller technical
# breakdown, rather than two separately hardcoded copies (rule 11).
MODEL_ACCURACY_STATS = {
    "median_ape_pct": 10.15,
    "within_10pct_share_pct": 49.3,
    "cqr_coverage_pct": 90.6,
}

st.set_page_config(page_title="G-Homes | Instant Property Price Prediction", page_icon="\U0001F3DB️", layout="wide")

# G-Homes brand palette. Kept as one dict (rather than scattered hex
# literals) so the hero banner below and the global CSS injection stay in
# sync with the same navy/gold values -- and with .streamlit/config.toml,
# which carries the same colors into Streamlit's own native theming
# (buttons, inputs, sliders) so the brand is consistent everywhere, not
# just inside the hero banner.
BRAND = {
    "navy_deep": "#07142B",
    "navy": "#0A1930",
    "navy_light": "#16385E",
    "gold": "#D4AF37",
    "gold_soft": "#E8CA7B",
    "cream": "#F2ECDC",
}


def inject_global_brand_css() -> None:
    """Applies the G-Homes look to Streamlit's OWN chrome (section
    headings, the Predict button) -- not just the hero banner below, which
    is a self-contained iframe and can't reach outside itself. This runs
    once, early, as a page-level <style> block (st.markdown's sanitizer
    passes plain <style>/<link> tags through -- this is the standard,
    documented way to skin a Streamlit app beyond what config.toml alone
    can reach)."""
    st.markdown(
        _flatten_html_for_markdown(
            f"""
            <style>
              {_local_font_faces_css(("Playfair Display",))}

              h2, h3 {{
                font-family: 'Playfair Display', Georgia, serif !important;
                color: {BRAND['gold']} !important;
                letter-spacing: 0.2px;
              }}

              div.stButton > button[kind="primary"] {{
                font-family: 'Playfair Display', Georgia, serif;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-weight: 600;
                border: 1px solid {BRAND['gold']};
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def render_hero_header(
    photos: list[str] | tuple[str, ...],
    brand: str,
    slogan: str,
    headline: str,
    subtitle: str,
    features: list[tuple[str, str]],
    accuracy_lines: list[str],
    height: int = 460,
) -> None:
    """The G-Homes hero banner: real house photos cross-fading behind a
    navy scrim (same Ken-Burns technique as the original dynamic header --
    still real photography, still moving -- just re-tinted and rebuilt
    around the agency's brand rather than a plain "House Price Estimator"
    title), with the slogan, headline, 3 feature badges, and an honest
    accuracy line layered on top in the brand's gold/serif type.

    Rendered via components.v1.html (an isolated iframe): the same reason
    as before -- keyframe CSS and per-slide animation-delay are the kind
    of thing Streamlit's markdown sanitizer or its own theme CSS can
    clash with, and an iframe guarantees this renders exactly as written.
    """
    n = len(photos)
    slot_pct = 100.0 / n
    duration_s = 4.5 * n

    slides_html = "\n".join(
        f'<img class="hp-slide" style="animation-delay:{i * (duration_s / n):.2f}s" '
        f'src="{url}" alt="">'
        for i, url in enumerate(photos)
    )
    badges_html = "\n".join(
        f'<span class="hp-badge"><span class="hp-badge-icon">{icon}</span>{label}</span>'
        for icon, label in features
    )
    accuracy_html = '<span class="hp-dot">&#9670;</span>'.join(
        f"<span>{line}</span>" for line in accuracy_lines
    )

    html = f"""
    <style>
      {_local_font_faces_css(("Playfair Display", "Inter", "Great Vibes"))}

      * {{ box-sizing: border-box; }}
      .hp-wrap {{
        position: relative;
        width: 100%;
        height: {height}px;
        border-radius: 16px;
        overflow: hidden;
        background: {BRAND['navy']};
        font-family: 'Inter', "Segoe UI", sans-serif;
        border: 1px solid rgba(212,175,55,0.35);
      }}
      .hp-slide {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0;
        animation-name: hp-fade, hp-zoom;
        animation-duration: {duration_s:.2f}s, {duration_s:.2f}s;
        animation-timing-function: ease-in-out, linear;
        animation-iteration-count: infinite, infinite;
      }}
      @keyframes hp-fade {{
        0%   {{ opacity: 0; }}
        1.5% {{ opacity: 1; }}
        {slot_pct - 2.5:.2f}% {{ opacity: 1; }}
        {slot_pct:.2f}%       {{ opacity: 0; }}
        100% {{ opacity: 0; }}
      }}
      @keyframes hp-zoom {{
        0%   {{ transform: scale(1.0); }}
        100% {{ transform: scale(1.08); }}
      }}
      .hp-overlay {{
        position: absolute;
        inset: 0;
        background:
          linear-gradient(180deg, rgba(7,20,43,0.80) 0%, rgba(7,20,43,0.72) 35%, rgba(7,20,43,0.93) 100%);
      }}
      .hp-content {{
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 28px 32px;
        color: {BRAND['cream']};
      }}
      .hp-brand {{
        display: flex;
        align-items: baseline;
        justify-content: center;
        gap: 2px;
      }}
      .hp-brand-g {{
        font-family: 'Great Vibes', cursive;
        font-size: 3.4rem;
        line-height: 1;
        color: {BRAND['gold']};
      }}
      .hp-brand-homes {{
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700;
        font-size: 2.1rem;
        letter-spacing: 0.5px;
        color: {BRAND['gold']};
      }}
      .hp-rule {{
        width: 64px;
        height: 2px;
        background: {BRAND['gold']};
        margin: 10px auto 14px;
        opacity: 0.85;
      }}
      .hp-slogan {{
        font-family: 'Playfair Display', Georgia, serif;
        font-style: italic;
        font-weight: 500;
        font-size: 1.1rem;
        color: {BRAND['gold_soft']};
        margin: 0 0 18px 0;
        text-shadow: 0 1px 6px rgba(0,0,0,0.5);
      }}
      .hp-headline {{
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700;
        font-size: 2.5rem;
        line-height: 1.15;
        margin: 0 0 12px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
      }}
      .hp-subtitle {{
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 720px;
        opacity: 0.92;
        margin: 0 0 22px 0;
        text-shadow: 0 1px 4px rgba(0,0,0,0.4);
      }}
      .hp-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 12px;
        margin-bottom: 22px;
      }}
      .hp-badge {{
        border: 1px solid {BRAND['gold']};
        background: rgba(212,175,55,0.08);
        color: {BRAND['gold_soft']};
        padding: 7px 16px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        white-space: nowrap;
      }}
      .hp-badge-icon {{
        margin-right: 6px;
        color: {BRAND['gold']};
      }}
      .hp-accuracy {{
        border-top: 1px solid rgba(212,175,55,0.35);
        padding-top: 14px;
        font-size: 0.85rem;
        color: {BRAND['cream']};
        opacity: 0.9;
      }}
      .hp-accuracy .hp-dot {{
        color: {BRAND['gold']};
        margin: 0 14px;
        font-size: 0.6rem;
        vertical-align: middle;
      }}

      /* AUDIT FIX 2026-09-09 (SB8, final audit): `.hp-wrap` has a fixed
         height + overflow:hidden while `.hp-content` reflows with viewport
         width -- below ~500px the content (headline/subtitle/badges
         wrapping onto more lines) grew taller than the fixed box and was
         silently clipped top and bottom (vertically centered content
         overflows symmetrically), cutting the wordmark in half and hiding
         the accuracy line entirely. Reproduced with headless Chromium at
         16 widths before this fix (480px: +30.8px overflow; 420px:
         +38.3px; 320px: +125.8px, closely matching the audit's own
         independent measurement). These breakpoints compress the same
         content (never hide the accuracy line -- that's the one thing
         this fix must never clip) so it fits the existing fixed height
         instead of changing the height itself, which would leave a large
         empty gap in the banner at normal desktop widths. Re-measured
         after this fix at the same 16 widths down to 320px: zero overflow
         at every one, with headroom to spare. */
      @media (max-width: 640px) {{
        .hp-content {{ padding: 22px 24px; }}
        .hp-headline {{ font-size: 2.0rem; margin-bottom: 8px; }}
        .hp-subtitle {{ font-size: 0.88rem; margin-bottom: 16px; }}
        .hp-badges {{ margin-bottom: 16px; }}
      }}
      @media (max-width: 480px) {{
        .hp-content {{ padding: 16px 18px; }}
        .hp-brand-g {{ font-size: 2.4rem; }}
        .hp-brand-homes {{ font-size: 1.5rem; }}
        .hp-rule {{ margin: 6px auto 8px; }}
        .hp-slogan {{ font-size: 0.85rem; margin: 0 0 8px 0; }}
        .hp-headline {{ font-size: 1.5rem; line-height: 1.2; margin-bottom: 6px; }}
        .hp-subtitle {{ font-size: 0.76rem; line-height: 1.35; margin-bottom: 10px; }}
        .hp-badges {{ gap: 8px; margin-bottom: 10px; }}
        .hp-badge {{ padding: 4px 10px; font-size: 0.68rem; }}
        .hp-accuracy {{ padding-top: 8px; font-size: 0.7rem; }}
        .hp-accuracy .hp-dot {{ margin: 0 8px; }}
      }}
      @media (max-width: 360px) {{
        .hp-slogan {{ display: none; }}
        .hp-headline {{ font-size: 1.25rem; }}
        .hp-subtitle {{ display: none; }}
        .hp-badges {{ display: none; }}
      }}
    </style>
    <div class="hp-wrap">
      {slides_html}
      <div class="hp-overlay"></div>
      <div class="hp-content">
        <div class="hp-brand">{brand}</div>
        <div class="hp-rule"></div>
        <p class="hp-slogan">&ldquo;{slogan}&rdquo;</p>
        <h1 class="hp-headline">{headline}</h1>
        <p class="hp-subtitle">{subtitle}</p>
        <div class="hp-badges">
          {badges_html}
        </div>
        <div class="hp-accuracy">{accuracy_html}</div>
      </div>
    </div>
    """
    components.html(html, height=height + 10)


def esc_dollars(text: str) -> str:
    """Streamlit's markdown renderer (st.write/st.markdown/st.metric/
    st.error/st.info/st.caption) treats a MATCHED PAIR of literal '$'
    characters as inline LaTeX/KaTeX math -- found by actually looking at a
    rendered screenshot, where "Likely range (90%): $458,066 - $1,020,417"
    (two '$' signs) came out as a mangled, partially-dropped string (LaTeX
    also treats '%' as a comment character, so text between the two '$'s
    got silently truncated at the first '%'). Every dollar amount and the
    narrative text in this app can contain two or more '$' signs, so this
    is applied everywhere such text is displayed -- '\\$' is Streamlit's
    documented way to show a literal dollar sign.

    FINAL AUDIT PANEL FIX 2026-09-10: apply this ONLY to plain markdown
    text (`st.write`/`st.caption`/`st.error`), NEVER to a string that is
    about to be rendered as an HTML block via
    `st.markdown(..., unsafe_allow_html=True)`. Found by screenshotting the
    running app: the value-range gauge, the price-per-sqft bars and the
    comp cards were all displaying a literal backslash -- "\\$835,610"
    instead of "$835,610" -- on the app's most prominent numbers.

    The cause is a CommonMark rule: inside an HTML block, content is passed
    through verbatim, so neither backslash-escape processing nor KaTeX math
    parsing runs. Confirmed by isolating both cases in a live Streamlit
    render:

        HTML block,  "$458,066 - $1,020,417"   -> "$458,066 - $1,020,417" OK
        HTML block,  "\\$458,066"               -> "\\$458,066"  <- the bug
        plain md,    "$458,066 - $1,020,417"   -> "458,066 - 1,020,417"  <- KaTeX ate them
        plain md,    "\\$458,066 - \\$1,020,417" -> "$458,066 - $1,020,417" OK

    So the escape is REQUIRED in plain markdown and HARMFUL in an HTML
    block -- the very same immunity that makes HTML blocks safe from the
    KaTeX bug also stops them from un-escaping the backslash."""
    return text.replace("$", "\\$")


def _flatten_html_for_markdown(html: str) -> str:
    """Streamlit's markdown renderer treats any line indented 4+ spaces as
    a CommonMark INDENTED CODE BLOCK -- found by the user pasting back
    EXACTLY the raw, unrendered `<div>`/`<style>` markup (backslash-escaped
    dollar signs and all) from the comp-cards and "why this price" widgets
    below, instead of seeing them rendered. Both symptoms (literal HTML
    tags shown as text, and esc_dollars' '\\$' shown un-unescaped) are
    exactly what happens inside a code block: no HTML parsing, no markdown
    escape processing. The cause: those widgets build their HTML as
    triple-quoted f-strings that inherit several levels of Python source
    indentation, and CommonMark doesn't care that the indentation came from
    Python, not markdown.

    Stripping every line's leading whitespace is safe (HTML/CSS both
    ignore insignificant whitespace) and is applied right before every
    `st.markdown(..., unsafe_allow_html=True)` call in this file that
    builds a multi-line string. NOT needed for `components.html()` calls
    (the hero header) -- those inject raw HTML directly into an iframe via
    the browser's own HTML parser, never through Streamlit's markdown/
    CommonMark pipeline, so they were never affected by this."""
    return "\n".join(line.lstrip() for line in html.strip().split("\n"))


def render_loading_state(message: str) -> None:
    """UI_AMENDMENTS.md entry 28: branded loading icon -- a pulsing gold
    ◆ diamond in place of Streamlit's generic spinner ring, shown while
    `run_interpretation()` computes the estimate (entry 7's spinner
    message text is unchanged, just the visual around it). Reuses the
    SAME ◆ character already standardized as the app's icon motif
    elsewhere (the hero header badges, entry 21).

    AUDIT FIX 2026-09-09 (SB16, final audit): this docstring used to also
    claim the ◆ "replaced the tab icons... and sits at the center of the
    confidence ring -- entries 16/18/21", which is stale on both counts:
    entry 22 removed `st.tabs` entirely (confirmed: `grep -n "st.tabs"
    app.py` -> no matches), so there is no tab icon for the ◆ to have
    replaced anymore; and `render_confidence_ring()` renders
    `{level.upper()}` at the ring's center (e.g. "TYPICAL"), never a ◆.
    Corrected to describe only where the ◆ motif actually still appears.

    Pure CSS `@keyframes` (no `<script>` tag needed) -- unlike the
    animated price count-up and hero header, which both need
    `components.html()`'s isolated iframe because Streamlit's markdown
    pipeline never executes `<script>` tags, this animates fine through
    plain `st.markdown(unsafe_allow_html=True)`, the same way
    `render_value_gauge()`'s own `vg-slide` keyframe animation already
    does without an iframe."""
    html = f"""
    <style>
      .gh-loading-wrap {{ text-align: center; padding: 34px 10px 22px; }}
      .gh-loading-diamond {{
        display: inline-block; font-size: 2.6rem; color: {BRAND['gold']};
        animation: gh-loading-pulse 1.3s ease-in-out infinite;
      }}
      @keyframes gh-loading-pulse {{
        0%, 100% {{ transform: scale(0.85); opacity: 0.45; }}
        50% {{ transform: scale(1.15); opacity: 1; }}
      }}
      .gh-loading-text {{ margin-top: 14px; color: {BRAND['cream']}; font-size: 0.92rem; }}
    </style>
    <div class="gh-loading-wrap">
      <div class="gh-loading-diamond">◆</div>
      <div class="gh-loading-text">{message}</div>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Commercial main-page widgets (UI_AMENDMENTS.md entries 2 + 9)
#
# These replace the 3 analytical matplotlib plots on the main "Estimate"
# view with plain-language, animated equivalents aimed at a customer, not
# a technical audience. The ORIGINAL 3 plots (SHAP bar chart, Tier 1 range,
# Tier 2 comps scatter) are unchanged and still built via
# `step6_plots.build_all_plots` -- they just moved to the "Market Insights"
# tab instead of the main page (see the results section below).
#
# All three are rendered with `st.markdown(..., unsafe_allow_html=True)`
# rather than `components.html()` (unlike the hero header): a plain
# markdown injection flows naturally with the rest of the page (no fixed
# iframe height to get wrong), which matters here since these blocks'
# height varies with how many SHAP factors/comps a given house has.
#
# FINAL AUDIT PANEL FIX 2026-09-10: these blocks used to run their text
# through `esc_dollars` first, on the assumption that going through
# Streamlit's markdown renderer exposed them to the matched-pair-dollar
# KaTeX bug. Screenshotting the running app disproved that: because these
# are HTML BLOCKS, CommonMark passes their content through verbatim -- no
# KaTeX parsing (so no escape needed) and no backslash-escape processing
# (so the escape rendered as a visible "\$" on the app's headline
# numbers). The escape is now applied only to genuine plain-markdown text;
# see `esc_dollars`' docstring for the isolated proof of both halves.
# --------------------------------------------------------------------------- #
def render_why_this_price(shap: dict, top_n: int = 4) -> None:
    """Commercial replacement for the SHAP drivers bar chart: a "strength
    meter" list in plain language instead of a chart with an axis. Reuses
    the EXACT same grouped top-features data (and the SAME `pct_effect`
    numbers) as the analytical SHAP chart in step6_plots.py
    (`group_top_features_by_label`, rule 11 -- one grouping/labeling
    implementation, not two) so the two views can never silently disagree
    about which factors mattered most or by how much. UI_AMENDMENTS.md
    entry 18: the exact percentage is now shown beside each bar too
    (originally deliberately omitted here to keep this view "easy ones to
    be understood" -- per later explicit instruction, showing the number
    is preferred after all). Rows fade/slide in one at a time."""
    grouped = group_top_features_by_label(shap["top_features"])[:top_n]

    def _strength(pct_effect: float) -> tuple[str, int]:
        mag = abs(pct_effect)
        if mag >= 0.08:
            return "Strong", 100
        if mag >= 0.03:
            return "Moderate", 65
        return "Slight", 35

    rows_html = []
    for i, g in enumerate(grouped):
        up = g["pct_effect"] >= 0
        strength, fill = _strength(g["pct_effect"])
        arrow = "▲" if up else "▼"
        direction_word = "boosts" if up else "reduces"
        color = "#34d399" if up else "#f87171"
        pct_text = f"{'+' if up else ''}{g['pct_effect'] * 100:.1f}%"
        delay = i * 0.18
        rows_html.append(f"""
        <div class="wp-row" style="animation-delay:{delay:.2f}s">
          <div class="wp-arrow" style="color:{color}">{arrow}</div>
          <div class="wp-text">
            <div class="wp-label">{g['label'].capitalize()}</div>
            <div class="wp-sub">{strength} {direction_word} your estimated value</div>
          </div>
          <div class="wp-meter">
            <div class="wp-meter-fill" style="--fill:{fill}%; background:{color}; animation-delay:{delay + 0.15:.2f}s"></div>
          </div>
          <div class="wp-pct" style="color:{color}">{pct_text}</div>
        </div>
        """)

    html = f"""
    <style>
      .wp-wrap {{ display: flex; flex-direction: column; gap: 14px; padding: 10px 4px 4px; }}
      .wp-row {{
        display: flex; align-items: center; gap: 14px;
        opacity: 0; transform: translateY(8px);
        animation: wp-in 0.5s ease-out forwards;
      }}
      @keyframes wp-in {{ 0% {{ opacity: 0; transform: translateY(8px); }} 100% {{ opacity: 1; transform: translateY(0); }} }}
      .wp-arrow {{ font-size: 1.3rem; width: 26px; text-align: center; flex-shrink: 0; }}
      .wp-text {{ flex: 1 1 auto; min-width: 0; }}
      .wp-label {{ color: #F2ECDC; font-weight: 600; font-size: 0.95rem; }}
      .wp-sub {{ color: #B9C2D6; font-size: 0.8rem; margin-top: 1px; }}
      .wp-meter {{
        width: 100px; height: 6px; border-radius: 999px;
        background: rgba(255,255,255,0.12); flex-shrink: 0; overflow: hidden;
      }}
      .wp-meter-fill {{ height: 100%; width: 0; border-radius: 999px; animation: wp-fill 0.7s ease-out forwards; }}
      @keyframes wp-fill {{ 0% {{ width: 0; }} 100% {{ width: var(--fill); }} }}
      .wp-pct {{
        width: 56px; flex-shrink: 0; text-align: right;
        font-weight: 700; font-size: 0.85rem; font-variant-numeric: tabular-nums;
      }}
    </style>
    <div class="wp-wrap">
      {''.join(rows_html)}
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


def render_value_gauge(cqr_lo: float, predicted_price: float, cqr_hi: float) -> None:
    """Commercial replacement for the statistical range plot: a horizontal
    gauge bar with a pin marker that slides into position on load, plus
    big Low/High numbers underneath -- no "confidence interval"
    wording, same underlying cqr_lo/cqr_hi/predicted_price numbers already
    computed once by `predict_with_interval` and threaded through
    unchanged (never recomputed here).

    UI_AMENDMENTS.md entry 23a: the middle "Estimate $X" item was dropped
    from the row below the track -- entry 22 already put the animated
    price headline directly above this gauge, so the number was a
    duplicate a few pixels down. Low and High are untouched."""
    span = max(cqr_hi - cqr_lo, 1e-6)
    pin_pct = max(0.0, min(100.0, (predicted_price - cqr_lo) / span * 100.0))

    html = f"""
    <style>
      .vg-wrap {{ padding: 8px 6px 4px; }}
      .vg-heading {{
        color: #D4AF37; font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700; font-size: 1.15rem; margin-bottom: 30px;
      }}
      .vg-track {{
        position: relative; height: 10px; border-radius: 999px;
        background: linear-gradient(90deg, rgba(212,175,55,0.22), rgba(212,175,55,0.9));
        margin: 0 8px;
      }}
      .vg-pin {{
        position: absolute; top: 50%; left: 0%;
        transform: translate(-50%, -50%);
        width: 20px; height: 20px; border-radius: 50%;
        background: #D4AF37; border: 3px solid #F2ECDC;
        box-shadow: 0 0 0 4px rgba(212,175,55,0.25);
        animation: vg-slide 1.1s cubic-bezier(.22,1,.36,1) forwards;
        --pin-target: {pin_pct:.2f}%;
      }}
      @keyframes vg-slide {{ 0% {{ left: 0%; opacity: 0; }} 15% {{ opacity: 1; }} 100% {{ left: var(--pin-target); opacity: 1; }} }}
      .vg-pin-label {{
        position: absolute; top: -26px; left: 0%; transform: translateX(-50%);
        font-weight: 700; color: #F2ECDC; font-size: 0.85rem; white-space: nowrap;
        animation: vg-slide 1.1s cubic-bezier(.22,1,.36,1) forwards;
        --pin-target: {pin_pct:.2f}%;
      }}
      .vg-row {{ display: flex; justify-content: space-between; margin-top: 22px; }}
      .vg-item {{ text-align: center; flex: 1; }}
      .vg-item .vg-tag {{ color: #B9C2D6; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.6px; }}
      .vg-item .vg-num {{ color: #F2ECDC; font-size: 1.15rem; font-weight: 700; margin-top: 2px; }}
    </style>
    <div class="vg-wrap">
      <div class="vg-heading">Your Estimated Value Range</div>
      <div class="vg-track">
        <div class="vg-pin-label">${predicted_price:,.0f}</div>
        <div class="vg-pin"></div>
      </div>
      <div class="vg-row">
        <div class="vg-item"><div class="vg-tag">Low</div><div class="vg-num">${cqr_lo:,.0f}</div></div>
        <div class="vg-item"><div class="vg-tag">High</div><div class="vg-num">${cqr_hi:,.0f}</div></div>
      </div>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


def render_comp_cards(tier2: dict, predicted_price: float, top_n: int = 3) -> None:
    """Commercial replacement for the Tier 2 comps scatter plot: a small
    row of listing-style cards (closest-by-size comps first, same sort
    order `get_tier2_comps` already applies to `comps_display`) plus one
    comparison bar showing "Your estimate" vs. "Average of these homes".
    The full comps table (all of `comps_display`) stays available in the
    Market Insights tab's expander -- this only surfaces the top few.

    AUDIT FIX 2026-09-09 (SB7, final audit; corrected further in the
    2026-09-09 verification pass): "Average of these homes" used to
    average `tier2["comp_prices"]` -- ALL retrieved comps (55-175 of
    them, whatever `get_tier2_comps` actually found) -- while the label
    says "these homes", referring to the 3 cards actually shown, and while
    the adjacent `render_price_per_sqft_bars` widget ~40px away averages
    `comps_display` (capped at 15) for the exact same house. Two widgets
    on the same screen, two different comp populations, two different
    numbers, both captioned as if describing the same thing (confirmed
    live: a -6.1% to -9.4% mismatch between the label and the number it
    labels on real submissions). The first fix pass switched the SOURCE
    LIST from `comp_prices` to `comps_display`, matching the cross-widget
    population -- but left this function averaging the FULL (up to 15)
    `comps_display` list while the label still says "these homes" and only
    `top_n` (3) cards are drawn, so the same class of label/number mismatch
    survived (reproduced: a +25.5% / +$143,257 gap on one real submission).
    "these homes" can only honestly mean the cards actually on screen, so
    the average is now over the same `comps[:top_n]` slice the cards are
    built from."""
    comps = tier2["comps_display"][:top_n]
    comp_prices = [c["price"] for c in comps]
    avg_price = float(np.mean(comp_prices)) if len(comp_prices) else predicted_price

    cards_html = []
    for i, c in enumerate(comps):
        badge = '<div class="cc-badge">Most similar</div>' if i == 0 else ""
        cards_html.append(f"""
        <div class="cc-card" style="animation-delay:{i * 0.15:.2f}s">
          {badge}
          <div class="cc-price">${c['price']:,.0f}</div>
          <div class="cc-meta">{c['bedrooms']:g} bd &middot; {c['bathrooms']:g} ba &middot; {c['sqft_living']:,.0f} sqft</div>
          <div class="cc-loc">{c['city']}, {c['statezip']}</div>
        </div>
        """)

    max_val = max(predicted_price, avg_price, 1.0)
    your_pct = predicted_price / max_val * 100
    avg_pct = avg_price / max_val * 100

    html = f"""
    <style>
      .cc-grid {{ display: flex; gap: 14px; flex-wrap: wrap; padding: 8px 2px 4px; }}
      .cc-card {{
        position: relative; flex: 1 1 160px; min-width: 160px;
        background: rgba(212,175,55,0.06); border: 1px solid rgba(212,175,55,0.35);
        border-radius: 12px; padding: 14px 16px;
        opacity: 0; transform: translateY(8px);
        animation: cc-in 0.5s ease-out forwards;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }}
      .cc-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.35); }}
      @keyframes cc-in {{ 0% {{ opacity: 0; transform: translateY(8px); }} 100% {{ opacity: 1; transform: translateY(0); }} }}
      .cc-badge {{
        position: absolute; top: -10px; right: 12px;
        background: #D4AF37; color: #07142B; font-size: 0.68rem; font-weight: 700;
        padding: 3px 10px; border-radius: 999px; letter-spacing: 0.3px;
      }}
      .cc-price {{ color: #D4AF37; font-weight: 700; font-size: 1.15rem; }}
      .cc-meta {{ color: #F2ECDC; font-size: 0.85rem; margin-top: 4px; }}
      .cc-loc {{ color: #B9C2D6; font-size: 0.78rem; margin-top: 2px; }}
      .cc-compare {{ margin-top: 20px; padding: 0 2px; }}
      .cc-compare-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
      .cc-compare-label {{ width: 150px; flex-shrink: 0; color: #B9C2D6; font-size: 0.82rem; }}
      .cc-bar {{ flex: 1 1 auto; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.12); overflow: hidden; }}
      .cc-bar-fill {{ height: 100%; width: 0; border-radius: 999px; animation: cc-bar-fill 0.9s ease-out forwards; }}
      @keyframes cc-bar-fill {{ 0% {{ width: 0; }} 100% {{ width: var(--target); }} }}
      .cc-bar-you {{ background: #D4AF37; }}
      .cc-bar-avg {{ background: #3E6491; }}
      .cc-compare-value {{ width: 100px; flex-shrink: 0; text-align: right; color: #F2ECDC; font-weight: 600; font-size: 0.88rem; }}
    </style>
    <div class="cc-grid">{''.join(cards_html)}</div>
    <div class="cc-compare">
      <div class="cc-compare-row">
        <div class="cc-compare-label">Your estimate</div>
        <div class="cc-bar"><div class="cc-bar-fill cc-bar-you" style="--target:{your_pct:.1f}%"></div></div>
        <div class="cc-compare-value">${predicted_price:,.0f}</div>
      </div>
      <div class="cc-compare-row">
        <div class="cc-compare-label">Average of these homes</div>
        <div class="cc-bar"><div class="cc-bar-fill cc-bar-avg" style="--target:{avg_pct:.1f}%"></div></div>
        <div class="cc-compare-value">${avg_price:,.0f}</div>
      </div>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Market Insights ("report" tab) widgets -- UI_AMENDMENTS.md entry 17.
#
# Per explicit instruction, these 4 client-friendly, animated visuals go on
# the Market Insights tab, NOT the Estimate tab (which stays exactly as it
# was). Each restates a fact already computed elsewhere in the pipeline --
# predicted_price, confidence level, market position, comp prices -- never
# a second, separately-derived copy of that fact (rule 11).
# --------------------------------------------------------------------------- #
def render_report_price_headline(predicted_price: float) -> None:
    """An animated count-up restating the SAME predicted_price already
    shown on the Estimate tab, so the technical report opens with one
    clear headline number instead of straight into charts. Needs a real
    JS loop to animate the digits (with comma formatting) -- Streamlit's
    markdown pipeline never executes <script> tags it's handed (the same
    reason the hero header uses components.html instead of st.markdown),
    so this goes through an isolated iframe too."""
    html = f"""
    <!doctype html><html><head><meta charset="utf-8">
    <style>
      {_local_font_faces_css(("Playfair Display",))}
      html, body {{ margin: 0; padding: 0; background: {BRAND['navy']}; }}
    </style>
    </head><body>
      <div style="padding:14px 6px 8px; font-family:'Inter','Segoe UI',sans-serif; text-align:center;">
        <div style="color:{BRAND['cream']}; opacity:0.75; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
          Estimated Market Value
        </div>
        <div id="rph-num" style="color:{BRAND['gold']}; font-family:'Playfair Display',Georgia,serif; font-weight:700; font-size:2.4rem;">
          $0
        </div>
      </div>
      <script>
        (function () {{
          var target = {predicted_price:.0f};
          var el = document.getElementById('rph-num');
          var duration = 1100;
          var start = null;
          function frame(ts) {{
            if (start === null) start = ts;
            var t = Math.min(1, (ts - start) / duration);
            var eased = 1 - Math.pow(1 - t, 3);
            var val = Math.round(target * eased);
            el.textContent = '$' + val.toLocaleString('en-US');
            if (t < 1) requestAnimationFrame(frame);
          }}
          requestAnimationFrame(frame);
        }})();
      </script>
    </body></html>
    """
    components.html(html, height=100)


def render_property_type_badge(property_type_label: str) -> None:
    """Item 5 -- an independent "Property Type" badge shown right next to
    the price estimate, restating the SAME renamed physical-cluster name
    (`result.property_type_label`, computed once in
    step6_interpretation.py via `digit_free_property_type_label()`) that
    the narrative's "we call this group '...'" sentence and the Tier-1
    chart title already use -- never a second, independently-maintained
    copy of the cluster name (rule 11).

    Deliberately just the property-type half (e.g. "Untouched Classics"),
    not the full price-tier-qualified category -- the Low/Estimate/High
    gauge directly below this badge already conveys the price tier, so
    repeating it here would say the same thing twice on one screen (the
    same reasoning UI_AMENDMENTS.md entry 22 already applied to drop the
    duplicate plain-text confidence badge next to the confidence ring).

    Plain st.markdown, not components.html: unlike the animated price
    headline above, this is static text with nothing to count up or
    animate, so it doesn't need an isolated iframe."""
    html = f"""
    <style>
      .pt-badge-wrap {{ text-align: center; margin: -4px 0 12px; }}
      .pt-badge {{
        display: inline-block;
        background: {BRAND['gold']}; color: {BRAND['navy_deep']};
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700; font-size: 0.78rem; letter-spacing: 0.4px;
        padding: 5px 16px; border-radius: 999px;
      }}
      .pt-badge-tag {{
        text-transform: uppercase; letter-spacing: 0.6px; opacity: 0.72; margin-right: 6px;
      }}
    </style>
    <div class="pt-badge-wrap">
      <span class="pt-badge"><span class="pt-badge-tag">Property Type</span>{property_type_label}</span>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


def render_market_thermometer(market: dict) -> None:
    """Client-friendly visual for `market['point_position']` -- already
    computed once by `step6_market_comparison.compute_market_comparison`,
    never recomputed here -- as a 5-segment horizontal strip with the
    current position highlighted, replacing the plain-text "in the
    central/typical range" phrasing with something scannable at a glance."""
    position = market.get("point_position", "unknown")
    segments = [
        ("below", "Well Below"), ("lower", "Somewhat Below"), ("central", "Typical"),
        ("upper", "Somewhat Above"), ("above", "Well Above"),
    ]
    # AUDIT FIX 2026-09-09 (SB15, final audit): "unknown" (genuinely too
    # few comps to place the house at all -- step6_market_comparison's own
    # documented enum value) and "any OTHER unrecognized value" used to
    # share one message, "Not enough comparable sales nearby..." -- true
    # for the first case, but a confidently WRONG explanation for the
    # second: a value this function has never heard of means an upstream
    # enum renamed or drifted, not that comps were scarce. Split so a real
    # drift is visibly flagged as unexpected instead of silently
    # misexplained as a comps-count problem.
    if position == "unknown":
        st.caption("Not enough comparable sales nearby to place this house on the market-position scale.")
        return
    if position not in dict(segments):
        st.caption(f"Unable to display the market-position scale (unrecognized position value: {position!r}).")
        return

    seg_html = "".join(
        f'<div class="mt-seg{" mt-seg-active" if key == position else ""}"><span>{label}</span></div>'
        for key, label in segments
    )
    html = f"""
    <style>
      .mt-wrap {{ padding: 6px 4px 2px; }}
      .mt-title {{ color: {BRAND['cream']}; font-weight:600; font-size:0.95rem; margin-bottom: 10px; }}
      .mt-track {{ display: flex; gap: 4px; }}
      .mt-seg {{
        flex: 1; text-align: center; padding: 10px 4px; border-radius: 8px;
        background: rgba(255,255,255,0.06); color: {BRAND['cream']}; opacity: 0.55;
        font-size: 0.72rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
      }}
      .mt-seg-active {{
        background: rgba(212,175,55,0.18); border-color: {BRAND['gold']}; color: {BRAND['gold']};
        opacity: 1; transform: translateY(-3px); box-shadow: 0 4px 10px rgba(212,175,55,0.25);
      }}
    </style>
    <div class="mt-wrap">
      <div class="mt-title">Where This Estimate Sits vs. Nearby Sales</div>
      <div class="mt-track">{seg_html}</div>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


def render_confidence_ring(confidence: dict) -> None:
    """CSS-only circular "confidence ring" (a conic-gradient fill)
    restating the SAME `confidence['level']` already computed by
    `step6_confidence.compute_confidence` and already shown as a plain
    badge on the Estimate tab -- a second, more visual read of the
    identical value for the technical report, not a second computation."""
    # AUDIT FIX 2026-09-09 (SB15, final audit): `.get(level, ...)` falling
    # back to a plausible-looking MIDDLE value (fill 50%, neutral grey)
    # used to mean a genuinely unrecognized `level` (an upstream enum
    # rename/typo -- should never happen, but silently swallowing it
    # would hide exactly that) rendered as an ordinary-looking ring
    # instead of visibly signaling something's wrong.
    level = confidence.get("level", "typical")
    display = CONFIDENCE_DISPLAY.get(level, _CONFIDENCE_DISPLAY_FALLBACK)
    fill_pct = display["fill_pct"]
    color = display["color"]
    reasons = confidence.get("reasons", [])
    reasons_html = "".join(f"<li>{r.capitalize()}</li>" for r in reasons)
    reasons_block = f'<ul class="cr-reasons">{reasons_html}</ul>' if reasons else ""

    html = f"""
    <style>
      .cr-wrap {{ display: flex; align-items: center; gap: 20px; padding: 8px 4px; flex-wrap: wrap; }}
      .cr-ring {{
        width: 92px; height: 92px; border-radius: 50%; flex-shrink: 0;
        background: conic-gradient({color} {fill_pct}%, rgba(255,255,255,0.10) {fill_pct}% 100%);
        display: flex; align-items: center; justify-content: center;
        animation: cr-in 0.7s ease-out;
      }}
      @keyframes cr-in {{ 0% {{ opacity: 0; transform: scale(0.8); }} 100% {{ opacity: 1; transform: scale(1); }} }}
      .cr-ring-inner {{
        width: 70px; height: 70px; border-radius: 50%; background: {BRAND['navy']};
        display: flex; align-items: center; justify-content: center;
        font-size: 0.62rem; font-weight: 700; color: {color}; text-align: center; padding: 4px;
      }}
      .cr-label {{ color: {BRAND['cream']}; font-weight: 700; font-size: 1.05rem; text-transform: capitalize; }}
      .cr-reasons {{ margin: 6px 0 0; padding-left: 18px; color: {BRAND['cream']}; opacity: 0.85; font-size: 0.83rem; }}
    </style>
    <div class="cr-wrap">
      <div class="cr-ring"><div class="cr-ring-inner">{level.upper()}</div></div>
      <div>
        <div class="cr-label">{level} confidence</div>
        {reasons_block}
      </div>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


def render_price_per_sqft_bars(predicted_price: float, sqft_living: float, tier2: dict) -> None:
    """Your home's own $/sqft vs. the average $/sqft of the SAME comps
    already retrieved by `step6_market_comparison.get_tier2_comps` (never
    a separately queried or recomputed comp set) -- price-per-sqft is
    often the first number a buyer/seller reaches for, and the existing
    comp-cards widget (Estimate tab) doesn't surface it directly."""
    comps = tier2.get("comps_display", [])
    per_sqft_values = [c["price"] / c["sqft_living"] for c in comps if c.get("sqft_living")]
    if not per_sqft_values or not sqft_living:
        return
    your_psf = predicted_price / sqft_living
    avg_psf = float(np.mean(per_sqft_values))

    max_val = max(your_psf, avg_psf, 1.0)
    your_pct = your_psf / max_val * 100
    avg_pct = avg_psf / max_val * 100

    html = f"""
    <style>
      .psf-wrap {{ padding: 8px 4px 4px; }}
      .psf-title {{ color: {BRAND['cream']}; font-weight:600; font-size:0.95rem; margin-bottom: 12px; }}
      .psf-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
      .psf-label {{ width: 160px; flex-shrink: 0; color: {BRAND['cream']}; opacity: 0.85; font-size: 0.82rem; }}
      .psf-bar {{ flex: 1 1 auto; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.12); overflow: hidden; }}
      .psf-bar-fill {{ height: 100%; width: 0; border-radius: 999px; animation: psf-fill 0.9s ease-out forwards; }}
      @keyframes psf-fill {{ 0% {{ width: 0; }} 100% {{ width: var(--target); }} }}
      .psf-bar-you {{ background: {BRAND['gold']}; }}
      .psf-bar-avg {{ background: #3E6491; }}
      .psf-value {{ width: 90px; flex-shrink: 0; text-align: right; color: {BRAND['cream']}; font-weight: 600; font-size: 0.85rem; }}
    </style>
    <div class="psf-wrap">
      <div class="psf-title">Price per Square Foot</div>
      <div class="psf-row">
        <div class="psf-label">Your home</div>
        <div class="psf-bar"><div class="psf-bar-fill psf-bar-you" style="--target:{your_pct:.1f}%"></div></div>
        <div class="psf-value">${your_psf:,.0f}/sqft</div>
      </div>
      <div class="psf-row">
        <div class="psf-label">Nearby homes (avg)</div>
        <div class="psf-bar"><div class="psf-bar-fill psf-bar-avg" style="--target:{avg_pct:.1f}%"></div></div>
        <div class="psf-value">${avg_psf:,.0f}/sqft</div>
      </div>
    </div>
    """
    st.markdown(_flatten_html_for_markdown(html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached, expensive-once resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_context() -> InterpretationContext:
    return InterpretationContext()


@st.cache_resource
def get_result_cache() -> SimpleCache:
    return SimpleCache(max_size=256)


@st.cache_data
def load_city_zip_map() -> tuple[list[str], dict[str, list[str]]]:
    df = pd.read_csv(ENRICHED_DATASET_PATH)
    mapping = (
        df.groupby("city")["statezip"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )
    cities = sorted(mapping.keys())
    return cities, mapping


NOT_LISTED = "My city isn't listed"


def _should_cache_interpretation(result: InterpretationResult) -> bool:
    """AUDIT FIX 2026-09-09 (SA2, final audit): don't permanently cache a
    result that fell back to the template narrative because a REAL Gemini
    attempt failed (timeout, API error, malformed response, etc.) -- only
    that specific case. A house that falls back because no API key is
    configured at all (`GEMINI_API_KEY` falsy) is fine to cache normally:
    nothing about a retry would change, since `gemini_enabled=False` is
    already baked into the cache key and no call is ever attempted for it.
    Without this, one transient Gemini hiccup permanently pinned that
    exact house's cache entry to "Machine-Statistics (generated without
    AI)" for the rest of the app's process lifetime, with no way for a
    user to get a fresh attempt by resubmitting the identical inputs."""
    if result.narrative_source == "template" and GEMINI_API_KEY and result.degradation_reasons:
        return False
    return True


def run_interpretation(raw_house: dict) -> InterpretationResult:
    """Wraps interpret_house() behind the canonical-input cache built in
    Step 6 (step6_cache.py) -- an identical submission (e.g. someone
    re-clicking Predict on the same values, or a demo re-run) returns the
    exact same result instantly, with no repeated Gemini call and no risk
    of a different narrative purely from LLM sampling noise.

    AUDIT FIX 2026-09-09 (SB21b, final audit): this and `_should_cache_
    interpretation()`'s return-type annotations used to be QUOTED string
    literals ("InterpretationResult") with the real class never imported
    into this file at all -- harmless only because `from __future__
    import annotations` (line 23) defers every annotation's evaluation,
    so nothing here ever actually tried to resolve the string. But
    `typing.get_type_hints()` on either function raised `NameError:
    name 'InterpretationResult' is not defined` (confirmed) -- a real gap
    for any future tool, doc generator, or test that introspects these
    signatures. Now a real import (see the `step6_interpretation` import
    line above) and a real, unquoted annotation."""
    ctx = get_context()
    cache = get_result_cache()
    key = canonical_cache_key(raw_house, gemini_enabled=bool(GEMINI_API_KEY))

    def _compute():
        return interpret_house(raw_house, ctx=ctx, gemini_api_key=GEMINI_API_KEY or None)

    return cache.get_or_compute(key, _compute, should_cache=_should_cache_interpretation)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
inject_global_brand_css()
render_hero_header(
    _header_photo_data_uris(tuple(HEADER_PHOTOS)),
    # UI_AMENDMENTS.md entry 10: no more building-icon emoji next to the
    # name -- a larger two-font wordmark instead (script "G", serif
    # "-Homes"), styled by .hp-brand-g / .hp-brand-homes above.
    brand='<span class="hp-brand-g">G</span><span class="hp-brand-homes">-Homes</span>',
    slogan="Know the value before the sign hit the lawn.",
    headline="Instant Property Price Prediction",
    subtitle=(
        "Enter your property details below to get a data-driven market estimate backed by "
        "real comparable sales, neighborhood trends, and price-per-sqft analysis — in seconds."
    ),
    # UI_AMENDMENTS.md entry 21: emoji replaced with the brand's gold ◆
    # (this is raw HTML inside the hero banner's own components.html
    # iframe, so .hp-badge-icon is colored gold directly, no CSS
    # first-letter workaround needed).
    #
    # AUDIT FIX 2026-09-09 (SB16, final audit): this comment used to also
    # say "unlike the ◆ tab labels in entries 16/18" -- entry 22 removed
    # `st.tabs` from this app entirely (confirmed: `grep -n "st.tabs"
    # app.py` -> no matches), so those ◆ tab labels, and the
    # `[data-testid="stTabs"] ... p::first-letter {{ color: gold; }}` CSS
    # rule entry 18 added for them, are both gone -- correctly, but this
    # was the only place that fact was ever recorded. `inject_global_
    # brand_css()` no longer has that rule either (confirmed by grep).
    features=[
        ("◆", "Neighborhood-Level Precision"),
        ("◆", "Professional Technical Analysis"),
        ("◆", "Confidence Scoring"),
    ],
    # AUDIT FIX 2026-09-09 (SB6, final audit): these used to be two
    # independently hardcoded string literals ("within 10%", "90.6%")
    # instead of reading `MODEL_ACCURACY_STATS`, contradicting the comment
    # right above that dict's own definition (rule 11: "one constant
    # reused... rather than two separately hardcoded copies"). The real
    # median APE is 10.15%, not 10% -- the hardcoded hero line rounded a
    # real number in the flattering direction, and the page stated two
    # different figures for the same statistic (10% here, 10.15% in the
    # Market Insights breakdown a few hundred lines below). Now both spots
    # read the same dict, so they can't drift apart again.
    accuracy_lines=[
        f"Median estimate within {MODEL_ACCURACY_STATS['median_ape_pct']:.2f}% of actual sale price",
        f"90% confidence interval, calibrated to {MODEL_ACCURACY_STATS['cqr_coverage_pct']:.1f}% real-world accuracy",
    ],
)

cities, city_to_zips = load_city_zip_map()

# --------------------------------------------------------------------------- #
# Input widgets
# --------------------------------------------------------------------------- #
# Deliberately NOT wrapped in st.form(): the cascading city -> ZIP picker and
# the "has a basement" / "has been renovated" conditional fields all need to
# react IMMEDIATELY to another widget's value -- found, by actually clicking
# through the app, that st.form() batches every widget change and only
# reruns the script on submit, so a form wrapper here would silently freeze
# the location picker and the conditional fields at their very first values
# and never update them no matter what the user picks. Plain widgets + a
# regular st.button re-run on every change, which is what this UI needs.
st.subheader("House details")

# Min/max on every widget below is pulled directly from
# step6_validation.TIER_A_BOUNDS -- the same hard-reject bounds
# interpret_house() itself enforces -- so the UI can never even offer a
# value Tier A would reject, and the two never drift apart (rule 11).
col1, col2, col3 = st.columns(3)
with col1:
    sl_lo, sl_hi = TIER_A_BOUNDS["sqft_living"]
    sqft_living = st.number_input(
        "Living area (sqft)", min_value=sl_lo, max_value=sl_hi, value=1800, step=50,
        help="Required. Total finished living area, above and below ground.",
    )
    bd_lo, bd_hi = TIER_A_BOUNDS["bedrooms"]
    bedrooms = st.number_input("Bedrooms", min_value=bd_lo, max_value=bd_hi, value=3, step=1, help="Required.")
    ba_lo, ba_hi = TIER_A_BOUNDS["bathrooms"]
    bathrooms = st.number_input(
        "Bathrooms", min_value=ba_lo, max_value=ba_hi, value=2.0, step=0.25, help="Required.",
    )
with col2:
    yb_lo, yb_hi = TIER_A_BOUNDS["yr_built"]
    yr_built = st.number_input(
        "Year built", min_value=yb_lo, max_value=yb_hi, value=1995, step=1,
        help="Required. This model was trained on sales through 2014. A house built after 2014 still gets "
             f"a real estimate -- priced as if it were built in 2014, with an added confidence caution -- "
             f"but a year after {yb_hi} is rejected as implausible.",
    )
    fl_lo, fl_hi = TIER_A_BOUNDS["floors"]
    floors = st.number_input(
        "Floors", min_value=fl_lo, max_value=fl_hi, value=1.0, step=0.5,
        help=f"Optional -- defaults to {OPTIONAL_FIELD_DEFAULTS['floors']} if left as-is.",
    )
    co_lo, co_hi = TIER_A_BOUNDS["condition"]
    # 2026-09-18 UI ADDITION: display-only labels beside each number, so a
    # first-time visitor knows what "3" means without guessing. `options`
    # is still the plain-int list TIER_A_BOUNDS defines, and `format_func`
    # only changes what's SHOWN -- st.selectbox still returns the raw int
    # option itself, so `condition` below is exactly the same value type
    # Tier A validation and the model have always received. See the App
    # Catalogue document for the full naming rationale.
    #
    # 2026-09-19: pulled forward from the live desktop copy (applied there
    # directly on 2026-09-18) to keep this sandbox copy in sync -- see
    # PACKAGING_TODO.md's "Comprehensive final code/logic verification
    # round" entry for why the two had drifted.
    CONDITION_LABELS = {1: "Poor", 2: "Fair", 3: "Average", 4: "Good", 5: "Very Good"}
    condition = st.selectbox(
        "Overall condition", options=list(range(co_lo, co_hi + 1)), index=2,
        format_func=lambda v: f"{v} — {CONDITION_LABELS.get(v, v)}",
        help=f"Optional ({co_lo}=worst, {co_hi}=best) -- defaults to {OPTIONAL_FIELD_DEFAULTS['condition']}.",
    )
with col3:
    vw_lo, vw_hi = TIER_A_BOUNDS["view"]
    # 2026-09-18 UI ADDITION: same display-only labeling as Overall
    # condition above -- see that comment for why this is safe.
    VIEW_LABELS = {0: "No View", 1: "Fair View", 2: "Average View", 3: "Good View", 4: "Premium View"}
    view = st.selectbox(
        "View quality", options=list(range(vw_lo, vw_hi + 1)), index=0,
        format_func=lambda v: f"{v} — {VIEW_LABELS.get(v, v)}",
        help=f"Optional ({vw_lo}=none, {vw_hi}=excellent) -- defaults to {OPTIONAL_FIELD_DEFAULTS['view']}.",
    )
    waterfront = st.checkbox("Waterfront property", value=False, help="Optional.")
    has_basement = st.checkbox("Has a basement", value=False, help="Optional.")
    sb_lo, sb_hi = TIER_A_BOUNDS["sqft_basement"]
    sqft_basement = st.number_input(
        "Basement area (sqft)", min_value=sb_lo, max_value=sb_hi, value=0, step=50,
        disabled=not has_basement,
        help="Only asked if 'Has a basement' is checked -- the above-ground area is worked out "
             "automatically from living area minus this.",
    )

st.markdown("---")
col4, col5 = st.columns(2)
with col4:
    # AUDIT FIX 2026-09-09 (SB5, final audit): this widget hardcoded
    # min_value=0, max_value=1_000_000 -- literals that contradicted this
    # exact section's own rule-11 claim two lines above ("Min/max on every
    # widget below is pulled directly from step6_validation.TIER_A_BOUNDS
    # ... so the UI can never even offer a value Tier A would reject, and
    # the two never drift apart") and disagreed with the real bound,
    # TIER_A_BOUNDS["sqft_lot"] = (200, 5_000_000), in both directions:
    # 0/100/199 were freely selectable here yet rejected by validate_tier_a,
    # and 5_000_000 was accepted by Tier A yet unreachable in this widget.
    sl_lo, sl_hi = TIER_A_BOUNDS["sqft_lot"]
    sqft_lot = st.number_input(
        "Lot size (sqft)", min_value=sl_lo, max_value=sl_hi, value=int(OPTIONAL_FIELD_DEFAULTS["sqft_lot"]),
        step=100, help=f"Optional -- defaults to {OPTIONAL_FIELD_DEFAULTS['sqft_lot']:.0f} if left as-is.",
    )
with col5:
    was_renovated = st.checkbox("Has been renovated", value=False, help="Optional.")
    yr_lo, yr_hi = TIER_A_BOUNDS["yr_renovated"]
    # 2026-09-18 FIX (validation-logic audit): `step6_validation.
    # validate_tier_a()` now hard-rejects `yr_renovated < yr_built` (a home
    # can't be renovated before it was built) -- but this widget's
    # min_value was still the bare TIER_A_BOUNDS floor (0) and its default
    # `value` a hardcoded 2000, both independent of whatever year the user
    # just entered above for "Year built". A user whose house was built
    # after 2000 who ticked "Has been renovated" and left this at its
    # default would have hit an instant hard reject from a value they
    # never typed -- exactly the "UI can never even offer a value Tier A
    # would reject" invariant this section's own comment claims (rule 11),
    # broken by the new reject unless this widget is bound the same way
    # every sibling widget already is. `yr_built` is a plain Python
    # variable from earlier in this same script run (col2, above), so
    # both bounds and the default track it directly with no extra state.
    ren_lo = max(yr_lo, yr_built)
    # 2026-09-19 FIX (post-delivery code-logic review): the fix above binds
    # this widget's min_value/default to yr_built, but Streamlit's own
    # number_input has a documented behavior this didn't account for: on a
    # rerun, if the widget's PREVIOUSLY held value falls below its NEWLY
    # computed min_value, Streamlit silently resets it to `value` -- no
    # warning, no visual flag beyond the number quietly changing. Concrete
    # failure: user sets Year built=1970, enters Renovation year=1998,
    # then goes back and corrects Year built to 2005 -- on that rerun
    # ren_lo becomes 2005, 1998 is now out of bounds, and Streamlit
    # silently substitutes 2005 for the user's actual 1998 with no
    # indication anything changed, so a submitted house can carry a
    # renovation year the user never entered. Fixed by giving the widget
    # an explicit key so its last value is readable from session_state
    # BEFORE Streamlit's own reset happens, and warning the user by name
    # exactly when that reset is about to silently occur, rather than
    # letting it pass unnoticed.
    _ren_key = "yr_renovated_input"
    _prev_ren = st.session_state.get(_ren_key)
    if _prev_ren is not None and _prev_ren < ren_lo:
        st.warning(
            f"Year built was raised to {yr_built}, which is after the renovation year you entered "
            f"({_prev_ren}). Renovation year has been reset to {max(2000, ren_lo)} below -- please "
            "re-enter the correct renovation year before predicting."
        )
    yr_renovated = st.number_input(
        "Renovation year", min_value=ren_lo, max_value=yr_hi, value=max(2000, ren_lo), step=1,
        disabled=not was_renovated, key=_ren_key,
        help="Only asked if 'Has been renovated' is checked. Must be on or after the year built above -- a "
             "home can't be renovated before it existed. This model was trained on renovations through 2014. "
             "A later renovation year still gets a real estimate -- priced as if the renovation happened in 2014, "
             f"with an added confidence caution -- but a year after {yr_hi} is rejected as implausible.",
    )

st.markdown("---")
st.subheader("Location")
city_options = cities + [NOT_LISTED]
city_choice = st.selectbox("City", options=city_options, index=0)

if city_choice == NOT_LISTED:
    col6, col7 = st.columns(2)
    with col6:
        free_city = st.text_input("City name", placeholder="e.g. Spokane")
    with col7:
        free_state_zip = st.text_input("State + ZIP", placeholder="e.g. WA 99201")
    st.info(
        "This location wasn't part of the training data, so the prediction will still be "
        "computed but flagged as lower-confidence -- shown clearly in the result, not hidden.",
        icon="ℹ️",
    )
    final_city, final_statezip = free_city.strip(), free_state_zip.strip()
else:
    zip_options = city_to_zips.get(city_choice, [])
    if len(zip_options) == 1:
        st.caption(f"ZIP code: {zip_options[0]} (only one on file for {city_choice})")
        zip_choice = zip_options[0]
    else:
        zip_choice = st.selectbox("ZIP code", options=zip_options)
    final_city, final_statezip = city_choice, zip_choice

submitted = st.button("Value It", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if submitted:
    raw_house = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "floors": floors,
        "waterfront": 1 if waterfront else 0, "view": view, "condition": condition,
        "sqft_living": sqft_living, "sqft_lot": sqft_lot,
        "sqft_above": None,  # always derived from sqft_living - sqft_basement (see apply_optional_defaults)
        "sqft_basement": sqft_basement if has_basement else 0,
        "yr_built": yr_built, "yr_renovated": yr_renovated if was_renovated else 0,
        "city": final_city or None, "statezip": final_statezip or None,
    }

    # UI_AMENDMENTS.md entry 7: branded copy instead of a generic
    # "Computing..." message, so the wait (Gemini call + building the
    # analytical plots isn't instant) feels intentional rather than stalled.
    # Entry 28: the generic spinner ring was replaced with a pulsing gold
    # ◆ (render_loading_state) -- st.empty() gives us a placeholder we can
    # clear the instant the real result is ready, same visible lifetime
    # `st.spinner()` used to manage for us automatically.
    loading_slot = st.empty()
    with loading_slot.container():
        render_loading_state("Analyzing comparable sales and neighborhood trends near you...")
    try:
        result = run_interpretation(raw_house)
        loading_slot.empty()

        st.markdown("---")

        if not result.ok:
            st.error("This house couldn't be evaluated:", icon="\U0001F6AB")
            for reason in result.rejection_reasons:
                st.write(f"- {esc_dollars(reason)}")
            st.caption(
                "This is a deliberate safety check (Tier A validation), not a bug -- it stops the model "
                "from being asked to score an input far outside anything real houses look like."
            )
        else:
            # UI_AMENDMENTS.md entry 22: the Estimate/Market Insights TAB split
            # from entries 2/5/9 is reversed here, per explicit later
            # instruction -- everything now renders on ONE page, titled
            # "Market Insights", instead of behind 2 tabs. Per the same
            # instruction: the confidence RING (below) is kept and the old
            # plain-text confidence badge is dropped (they said the same thing
            # twice once merged onto one page); for price, BOTH displays are
            # kept, with the animated headline number placed above the
            # Low/Estimate/High gauge scale.
            st.markdown("### Market Insights")
            # UI_AMENDMENTS.md entry 24: reworded per explicit user wording.
            st.caption(
                "Everything about your estimate in one place -- supported by AI methodologies."
            )

            # --- price, up top: animated headline, then the Low/Estimate/High scale ---
            render_report_price_headline(result.predicted_price)
            # Item 5: independent "Property Type" badge, right next to the
            # price estimate as requested -- between the headline number
            # and the Low/Estimate/High gauge, so it reads as a property of
            # the estimate rather than being buried lower on the page.
            render_property_type_badge(result.property_type_label)
            render_value_gauge(result.cqr_lo, result.predicted_price, result.cqr_hi)

            # --- confidence RING (kept) + market-position thermometer ---
            rc1, rc2 = st.columns(2)
            with rc1:
                render_confidence_ring(result.confidence)
            with rc2:
                render_market_thermometer(result.market)
            render_price_per_sqft_bars(result.predicted_price, sqft_living, result.tier2)

            # --- Tier B banners: never hide a caveat, show it plainly --
            # UI_AMENDMENTS.md entry 6: warmer wording (step6_confidence.py's
            # reason strings themselves were rewritten; the imputed-fields
            # note that used to be repeated separately right below this
            # container was removed as a duplicate -- the same fact already
            # appears as one of the reasons here whenever it applies).
            # AUDIT FIX 2026-09-09 (SA24, final audit): this used to
            # hand-roll its own copy of the same key-presence-only
            # comprehension that step6_llm_narrative.build_prompt_facts()
            # had (rule 11) -- both now go through one shared, value-aware
            # helper. See compute_fired_flag_labels()'s docstring.
            #
            # 2026-09-19 FIX (post-delivery code-logic review, panel finding):
            # three flags -- price_band_clipped, tier2_location_fallback ==
            # "nearest_neighbors", and beyond_training_age_range -- are each
            # ALSO turned into their own (differently-worded) sentence by
            # step6_confidence.py's reasons_limited below, unconditionally
            # whenever the flag fires (confirmed: no gate in either place
            # that could make one fire without the other). Left as-is, this
            # panel would show the same fact twice in a row with slightly
            # different wording, e.g. "The price category required a small
            # correction." immediately followed by "We had to make a small
            # correction to this home's price category." Filtered out of
            # THIS PANEL'S copy of the flags only -- deliberately NOT touching
            # `compute_fired_flag_labels()` itself or its `FLAG_LABELS`/
            # `TIER2_LOCATION_FALLBACK_LABELS` dicts, since that same function
            # also feeds `step6_llm_narrative.build_prompt_facts()`'s
            # `fired_flag_labels` fact list straight into the Gemini prompt --
            # removing these labels there would silently drop real facts the
            # AI narrative is supposed to know about, not just de-duplicate a
            # display. This is a display-only fix.
            _PANEL_DEDUP_FLAG_KEYS = ("price_band_clipped", "beyond_training_age_range")
            panel_flags = {k: v for k, v in result.flags.items() if k not in _PANEL_DEDUP_FLAG_KEYS}
            if result.tier2.get("tier2_tier_used") == "nearest_neighbors":
                panel_flags.pop("tier2_location_fallback", None)
            fired_flag_labels = compute_fired_flag_labels(panel_flags)
            if fired_flag_labels or result.confidence["reasons"]:
                with st.container(border=True):
                    st.markdown("**Good to know about this estimate:**")
                    for label in fired_flag_labels:
                        st.write(f"- {esc_dollars(label.capitalize())}.")
                    for reason in result.confidence["reasons"]:
                        st.write(f"- {esc_dollars(reason.capitalize())}.")

            # --- narrative -- UI_AMENDMENTS.md entry 5: no AI/non-AI tag ---
            st.subheader("Why This Estimate")
            st.write(esc_dollars(result.narrative))

            # --- commercial "why this price" + "comparable homes" (entries 2/9/18) ---
            st.subheader("Why This Price")
            render_why_this_price(result.shap)

            st.subheader("Comparable Homes Nearby")
            render_comp_cards(result.tier2, result.predicted_price)

            st.markdown("---")
            st.markdown("#### Charts")

            # UI_AMENDMENTS.md entry 17: house_id used to be built from only
            # 4 of the ~13 raw inputs (city/statezip/sqft_living/bedrooms), so
            # two DIFFERENT houses sharing those 4 values (e.g. same
            # city/zip/size/bedroom-count but different bathrooms, condition,
            # age, etc. -- easy to hit while tuning inputs to compare) wrote
            # their plots to the SAME 3 file paths, silently overwriting each
            # other on disk. Hashing the complete validated input dict makes
            # every distinct house's plot files unique, so this can't happen.
            house_id = hashlib.sha256(
                json.dumps(raw_house, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
            plot_paths = build_all_plots(result, PLOTS_DIR, house_id=house_id)

            # Full width, one at a time (previously 2 were squeezed side by
            # side into half-width columns) -- easier to read the category
            # labels/legends/annotations on each chart, plus a plain-English
            # caption under each explaining what it shows.
            st.image(plot_paths["shap_drivers"], use_container_width=True)
            st.caption(
                "**What this shows:** the specific features of this house that pushed the estimate up "
                "(green) or down (red), and roughly how much each one mattered -- the longest marks "
                "matter most."
            )

            st.image(plot_paths["tier1_range"], use_container_width=True)
            # UI_AMENDMENTS.md entry 26: static caption replaced with a dynamic
            # one built from `result.tier1_percentile_rank` (computed once by
            # build_all_plots() -- never recomputed here, per project rule 11).
            pct_rank = result.tier1_percentile_rank
            st.caption(
                f"**What this shows:** where this estimate (gold pin) falls among other homes of the "
                f"same general type and price tier. Priced higher than {pct_rank:.0f}% of similar homes."
            )

            st.image(plot_paths["tier2_comps"], use_container_width=True)
            # AUDIT FIX 2026-09-09 (SA14, final audit): "recent" was false --
            # every sale in the dataset is from May-July 2014, matching the
            # identical fix in step6_narrative.py.
            st.caption(
                "**What this shows:** this estimate against actual sale prices of similar homes "
                "nearby (blue dots), with the gold-shaded band marking this home's likely price range."
            )

            if result.market["market_divergence"] == "diverges":
                # AUDIT FIX 2026-09-09 (SA8, final audit): this banner used to open
                # with "Heads up:" and describe the two comparisons as pointing "in
                # different directions" -- framing that reads as an exception or a
                # problem with the estimate. Reproduced on the 908 held-out test
                # houses: this condition fires for 56.1% of them, because
                # `point_position` (a percentile rank among nearby comps) and
                # `tier1_position` (a +/-15% band around the home's price-category
                # MEAN, a statistic pulled around by a long right tail of luxury
                # outliers) are measuring genuinely different things and routinely
                # land in different buckets -- not because anything unusual
                # happened with this particular house. "A flag that fires on the
                # majority is not a flag," so the copy no longer calls it out as
                # noteworthy; it explains why two honest comparisons can legitimately
                # differ.
                st.info(
                    "This estimate is compared to the market two ways above: against similar sales "
                    "nearby, and against this home's overall price category. Landing in different spots on "
                    "each comparison is common and expected, not a sign anything is off -- the two are "
                    "measuring genuinely different things (specific nearby comps vs. a broader category "
                    "average). Both are shown so you get the fuller picture.",
                    icon="ℹ️",
                )

            st.markdown("#### Model accuracy")
            st.write(
                esc_dollars(
                    f"- Median estimate within {MODEL_ACCURACY_STATS['median_ape_pct']:.2f}% of the actual sale price\n"
                    f"- {MODEL_ACCURACY_STATS['within_10pct_share_pct']:.1f}% of estimates land within 10% of the actual sale price\n"
                    f"- The 90% likely range is calibrated to {MODEL_ACCURACY_STATS['cqr_coverage_pct']:.1f}% real-world coverage "
                    f"(measured across held-out sales, not this one house)"
                )
            )

            # AUDIT FIX 2026-09-09 (SA28, final audit): the narrative sentence
            # above ("Looking at N comparable sales...") reports the FULL
            # comp count the percentile/market-position statistics are
            # computed from, while this table only ever holds `comps_display`
            # -- capped at DISPLAY_CAP (15) sales. 83.3% of held-out houses
            # have N > 15, so most users saw an expander titled as if it held
            # everything the sentence just counted, when it holds only a
            # sample. The title and caption now say so explicitly instead of
            # letting the two numbers silently disagree.
            n_comps_found = result.tier2["n_comps_found"]
            n_comps_shown = len(result.tier2["comps_display"])
            comps_expander_label = (
                "See the individual comparable sales used"
                if n_comps_shown >= n_comps_found
                else f"See a sample of the comparable sales used ({n_comps_shown} of {n_comps_found} found)"
            )
            with st.expander(comps_expander_label):
                if n_comps_shown < n_comps_found:
                    st.caption(
                        f"Showing the closest {n_comps_shown} by size -- the estimate's statistics above are "
                        f"based on all {n_comps_found} comparable sales found, not just these."
                    )
                st.dataframe(pd.DataFrame(result.tier2["comps_display"]), use_container_width=True)

            # UI_AMENDMENTS.md entry 25: moved to the very last item in this
            # results section (was right after "Model accuracy"), and rendered
            # in a small caption font instead of a `####` heading -- per the
            # user's revised request. The non-AI label was also renamed from
            # "Plain-English" to "Machine-Statistics" (still with the same
            # clarifying parenthetical) since that's a more accurate
            # description of a narrative assembled from the code-owned
            # template rather than an LLM.
            source_label = "AI-written" if result.narrative_source == "llm" else "Machine-Statistics (generated without AI)"
            st.caption(f"Explanation source: {source_label}")
            # 2026-09-19 FIX (committee gap-review finding): ETHICS_AND_
            # LIMITATIONS.md's own "Third-party processing" section states
            # plainly that when the narrative is AI-written, "the property
            # details the user entered and the model's computed facts are
            # sent to Google's Gemini API to be written up as prose. Users
            # should be told this." -- until this fix, nowhere in the app
            # actually told them; the only visible signal was the neutral
            # "Explanation source: AI-written" label above, which never
            # names Gemini or Google or says any data left the machine.
            # Only shown when the LLM path actually ran (narrative_source
            # == "llm") -- the template fallback never calls Gemini, so
            # nothing left the machine in that case and no notice is shown.
            if result.narrative_source == "llm":
                st.caption(
                    "This explanation was generated using Google's Gemini API -- the property "
                    "details you entered and this estimate's computed facts were sent to Google to produce it."
                )
            if result.degradation_reasons:
                with st.expander("Why isn't this AI-written?"):
                    for r in result.degradation_reasons:
                        st.write(f"- {esc_dollars(r)}")
    except Exception as e:
        # AUDIT FIX 2026-09-09 (SB4, final audit): app.py used to have zero
        # exception handling in ~1200 lines -- an AssertionError from
        # step6_interpretation.py/step6_plots.py, a ValueError from SHAP or
        # a sklearn feature-contract drift, or any other unexpected failure
        # reached the end user as a raw Streamlit traceback exposing internal
        # file paths and code. This is the one broad safety net around the
        # actual computation + rendering (validation itself still raises its
        # own clear, user-facing rejection reasons via "result.ok", unaffected
        # by this -- this only catches genuinely UNEXPECTED failures).
        loading_slot.empty()
        st.error(
            "Something went wrong while building this estimate. Please try again, or "
            "adjust the inputs above and resubmit -- if the problem persists, it's a bug, "
            "not something wrong with your inputs.",
            icon="\U0001F6AB",
        )
        with st.expander("Technical details (for debugging)"):
            st.exception(e)

# UI_AMENDMENTS.md entry 3: the graduation-project disclaimer moved out of
# the hero area into a small, unobtrusive footer at the very bottom of the
# page (shown regardless of whether a prediction was submitted yet).
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:0.72rem; color:#8a94a6;'>"
    "G-Homes is a graduation-project demo brand built to showcase this pricing model -- "
    "not a licensed real-estate brokerage, and this estimate is not a professional appraisal."
    "</p>",
    unsafe_allow_html=True,
)
# UI_AMENDMENTS.md entry 18: a build fingerprint (hash of THIS file's own
# bytes, computed fresh at runtime -- not a manually-bumped number that's
# easy to forget) so a restart can be VERIFIED rather than assumed. After
# any deploy, compare this against the hash quoted alongside it -- if they
# don't match, the running Streamlit process is still serving an older
# app.py and needs a full restart (Ctrl+C, confirm no leftover python.exe
# in Task Manager, then re-run `streamlit run app.py` FROM THIS FILE'S OWN
# DIRECTORY), not just a browser refresh.
#
# AUDIT FIX 2026-09-09 (SB18, final audit): this comment used to say
# "relaunch run_app.bat" -- no such file, or any other launcher script,
# exists anywhere in this repo (`find . -name "run_app*"` and `find . -
# name "*.bat" -o -name "*.sh"` both return nothing outside `archive*`),
# so that instruction could never actually be followed. It also used to
# be the ONLY thing pinning the operator to the correct working
# directory -- since SB4's fix, every step6_*.py module and this file
# anchor their own paths to `REPO_ROOT = Path(__file__).resolve().parent`
# instead of the process's cwd, so `streamlit run app.py` now works
# correctly from any starting directory; naming this file's own
# directory here is just the plain, always-correct restart command, not
# a workaround for a cwd requirement that no longer exists.
st.markdown(
    f"<p style='text-align:center; font-size:0.62rem; color:#4a5468; margin-top:2px;'>"
    f"build {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]}"
    f"</p>",
    unsafe_allow_html=True,
)
