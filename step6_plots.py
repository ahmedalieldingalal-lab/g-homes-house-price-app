"""Step 7 (Deployment) -- the 3 end-user plots agreed under Decision 6a/6b
(PROJECT_TODO.md): (1) SHAP price-drivers bar chart, (2) Tier 1 category
price-range chart, (3) Tier 2 comparable-sales + CQR interval chart. Each is
explicitly labeled with what it's comparing against (Decision 6b) rather
than forced onto one shared baseline.

Non-negotiable requirement carried from that discussion: the ONE predicted
price computed by `step6_cqr_interval.predict_with_interval` (already
threaded, unchanged, through `InterpretationResult`) must appear verbatim
in every plot title here -- never recomputed. `build_all_plots()` checks
this itself before returning (an explicit `raise AssertionError`, not a
bare `assert`, so it cannot be stripped by `python -O` -- AUDIT FIX
2026-09-09, SB2 verification pass), and `tests/test_step6_plots.py` checks
it again independently by reading back each title's actual in-memory
Axes text via `ax.get_title()` (not by parsing anything from the saved PNG
files -- no PNG metadata is read anywhere) -- this is Step 24's
previously-deferred "plot-title half" of the price-consistency guard,
now that these plots actually exist.

Style: dpi=110, `fig.tight_layout()`, one fixed STEP6_PALETTE constant
(rather than any plotting library's default theme) -- originally matched
`regression_pipeline.py`'s light-background house look, then re-themed to
the G-Homes agency brand (navy background, gold accents) requested for
Step 7's UI, via `_style_dark_axes()`/`_style_legend()` so the 3 plots read
as part of the same branded page instead of plain white cards dropped onto
a navy app.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

from step6_llm_narrative import digit_free_category_label, feature_label, group_top_features_by_label
from step6_market_comparison import tier2_tier_phrase

REPO_ROOT = Path(__file__).resolve().parent
CLUSTER_PROFILE_STATS_PATH = REPO_ROOT / "saved_models" / "cluster_profile_stats.json"

# AUDIT FIX 2026-09-09 (SB10, final audit): `build_all_plots()` writes
# `result.plot_paths` and `result.tier1_percentile_rank` onto the
# `InterpretationResult` it's given -- and `app.py` shares ONE such result
# per house across every concurrent Streamlit session via a process-global
# `@st.cache_resource` cache (`SimpleCache`, see step6_cache.py's SA15 fix
# for the read side of this same sharing). Two sessions submitting the
# same house at the same time both get that one shared object and could
# both call `build_all_plots()` on it concurrently, racing on these writes
# with no lock -- SA11's fix already removed the one mutation site with
# real wrong-value-read potential (the temporary `tier1_profile` swap), but
# these two remaining writes were still unsynchronized. Since this is a
# "single-process demo app" (step6_cache.py's own sizing note) and plot
# generation is not the app's bottleneck, a single lock serializing the
# whole function is the simplest fix that is actually correct -- not just
# reducing the race window, eliminating it -- at a real but acceptable
# cost of serializing plot builds for DIFFERENT houses too.
_BUILD_ALL_PLOTS_LOCK = threading.Lock()

# AUDIT FIX 2026-09-09 (SB20, final audit): `app_plots/` (this module's
# `output_dir`) used to grow WITHOUT BOUND -- entry 17 changed `house_id`
# from a 4-field hash to a hash of the full validated input specifically
# so distinct houses stop overwriting each other's files, which is
# correct, but it turned an overwrite bug into unbounded accumulation:
# 3 PNGs per distinct submission, retained forever, no eviction/TTL/size
# cap anywhere in this file or app.py (confirmed: `grep -n "unlink\|
# rmtree\|glob(" step6_plots.py app.py` found no delete path at all
# before this fix) -- and each PNG is a rendering of a real visitor's own
# property details and price estimate, not disposable scratch output.
# `_evict_old_plot_groups()` below caps the directory to the most
# recently-written `MAX_CACHED_HOUSE_PLOTS` distinct houses' worth of
# files, run from inside `build_all_plots()`'s own lock so concurrent
# builds can't race on the eviction scan.
MAX_CACHED_HOUSE_PLOTS = 200
_PLOT_FILE_SUFFIXES = ("_shap_drivers.png", "_tier1_range.png", "_tier2_comps.png")


def _evict_old_plot_groups(output_dir: Path, keep_house_id: str, max_houses: int = MAX_CACHED_HOUSE_PLOTS) -> None:
    """Keeps at most `max_houses` distinct house_id groups of plot files in
    `output_dir`, deleting the LEAST RECENTLY WRITTEN groups first (by each
    group's newest file mtime) once that cap is exceeded. `keep_house_id`
    (the house about to be written) is never evicted, even if it isn't in
    `output_dir` yet."""
    groups: dict[str, list[Path]] = {}
    for f in output_dir.glob("*.png"):
        for suffix in _PLOT_FILE_SUFFIXES:
            if f.name.endswith(suffix):
                groups.setdefault(f.name[: -len(suffix)], []).append(f)
                break
    groups.setdefault(keep_house_id, [])

    if len(groups) <= max_houses:
        return

    def newest_mtime(house_id: str) -> float:
        files = groups[house_id]
        return max((f.stat().st_mtime for f in files), default=0.0)

    evictable = sorted((hid for hid in groups if hid != keep_house_id), key=newest_mtime)
    for hid in evictable[: len(groups) - max_houses]:
        for f in groups[hid]:
            f.unlink(missing_ok=True)

# One fixed palette for all 3 plots. G-Homes brand re-theme: "subject" IS
# the brand gold now (D4AF37) -- doing double duty as both the "this house"
# marker color and the plot title color, one constant instead of two so the
# brand accent can't drift out of sync between them (rule 11). The
# increase/decrease/category/comp-dot colors are the same green/red/purple/
# teal families as before, each brightened one Tailwind step (e.g. 500->400)
# so they still read clearly against the new dark navy background instead
# of the light background they were originally tuned for.
STEP6_PALETTE = {
    "subject": "#D4AF37",       # G-Homes brand gold -- "this house" marker AND plot titles
    "increase": "#34d399",      # SHAP: feature pushes price up
    "decrease": "#f87171",      # SHAP: feature pushes price down
    "neutral": "#94a3b8",       # catch-all / neutral bars, axis zero-line
    "range_fill": "#3E6491",    # Tier 1 min-max range bar fill -- muted steel-blue, deliberately NOT gold
                                 # (Decision 6b: Tier 1's category range and Tier 2's likely-range band are
                                 # two different comparison lenses and stay visually distinct on purpose)
    "category_marker": "#a78bfa",  # Tier 1 category mean marker
    "interval_fill": "#D4AF37",  # CQR interval shaded band -- soft brand gold glow (low alpha, see plot_tier2_comps)
    "comp_dot": "#22d3ee",      # Tier 2 individual comp markers
    "warning": "#fbbf24",       # thin-evidence / fallback annotation color
    "grid": "#25406b",          # gridlines/spines -- subtle on the dark navy background
}

# The dark-navy plot background itself (a shade lighter than the app page's
# own navy so each chart still reads as a distinct card) plus the two text
# tones used across all 3 plots -- cream for primary labels/ticks, a dimmer
# cream for secondary/muted annotations (subtitles, thin-evidence notes).
DARK_BG = "#0F2340"
DARK_TEXT = "#F2ECDC"
DARK_TEXT_MUTED = "#B9C2D6"

DPI = 110
# UI_AMENDMENTS.md entry 18: widened slightly (8.5->9.5in, 5->5.4in) for
# more breathing room around bar/category labels on the Market Insights
# tab, where charts now render full-width (entry 17) instead of squeezed
# into half-width columns.
FIGSIZE = (9.5, 5.4)
# UI_AMENDMENTS.md entry 27: `plot_tier1_range`'s percentile gauge (entry
# 26) has much sparser content than the other 2 charts -- one thin track
# plus a couple of short label lines -- so at the shared FIGSIZE height it
# rendered as a tall mostly-empty navy card that read as "stretched" once
# Streamlit scaled it to the full container width. A dedicated, shorter
# figure size lets that chart's actual content fill its frame instead.
TIER1_GAUGE_FIGSIZE = (9.5, 3.1)


def _style_dark_axes(fig, ax) -> None:
    """Applies the G-Homes navy/gold brand to one figure/axes: dark navy
    background, cream tick/axis-label text, gold bold title, subtle navy
    gridline/spine color. Called by all 3 plot functions right after the
    figure is built, so the brand can't drift out of sync between them
    (rule 11 -- one styling routine, not 3 copies of the same tweaks)."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(STEP6_PALETTE["grid"])
    ax.tick_params(colors=DARK_TEXT, labelsize=8.5)
    ax.xaxis.label.set_color(DARK_TEXT)
    ax.yaxis.label.set_color(DARK_TEXT)
    ax.title.set_color(STEP6_PALETTE["subject"])
    ax.title.set_fontweight("bold")


def _style_legend(leg) -> None:
    """Restyles a matplotlib legend to match the dark navy/gold theme --
    without this, `framealpha=0.9` on the default white legend background
    would leave a bright white rectangle sitting on top of the dark chart."""
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(DARK_BG)
    frame.set_edgecolor(STEP6_PALETTE["subject"])
    frame.set_alpha(0.95)
    for text in leg.get_texts():
        text.set_color(DARK_TEXT)


def _esc_mpl_dollars(text: str) -> str:
    """matplotlib enables mathtext parsing on any text containing a
    MATCHED PAIR of literal '$' characters (same mechanism as Streamlit's
    markdown renderer, found the same way -- by actually rendering a plot
    and reading it: the tier-2 legend's "$162,896-$610,843" came out as
    "162, 896 -" with mathtext's own comma/operator spacing rules replacing
    the intended plain currency formatting). '\\$' is matplotlib's
    documented escape for a literal dollar sign, same convention as LaTeX."""
    return text.replace("$", "\\$")


def _feature_label(feature: str) -> str:
    """Chart-specific presentation on top of the one canonical lookup
    (`step6_llm_narrative.feature_label`, lowercase, correct for
    mid-sentence narrative prose) -- charts read better with a capitalized
    first word. AUDIT FIX 2026-09-09 (SA23, final audit): this used to
    re-implement the `FEATURE_DISPLAY_NAMES.get(...)` lookup itself,
    creating a second, independently-maintained copy that a change-log
    entry incorrectly described as "one shared implementation, not
    three" -- there were two. Now a thin wrapper over the shared
    lookup, so there is genuinely only one."""
    return feature_label(feature).capitalize()


# AUDIT FIX 2026-09-09 (SB21f, final audit): this used to re-read and
# re-parse `cluster_profile_stats.json` from disk on EVERY chart render
# (called once per `plot_tier1_range()` call, i.e. once per house
# submission) -- the only loader in this file, or in step6_classify_
# house.py/step6_inference_contract.py/step6_market_comparison.py, that
# didn't cache in a module global. Now matches that same pattern.
_CLUSTER_PROFILE_STATS_CACHE: dict | None = None


def _load_cluster_profile_stats() -> dict:
    global _CLUSTER_PROFILE_STATS_CACHE
    if _CLUSTER_PROFILE_STATS_CACHE is None:
        _CLUSTER_PROFILE_STATS_CACHE = json.loads(CLUSTER_PROFILE_STATS_PATH.read_text())
    return _CLUSTER_PROFILE_STATS_CACHE


# UI_AMENDMENTS.md entry 20: extra vertical breathing room between rows in
# the redesigned lollipop chart below (one shared constant so the row
# spacing and the ylim padding that frames it can never drift apart).
SHAP_ROW_SPACING = 1.35


def plot_shap_drivers(result, save_path: Path) -> str:
    """Top price drivers, as multiplicative percentage effects (Decision 2's
    corrected, log-scale-safe presentation -- see step6_shap_contract.py's
    module docstring for why these are percentages, not dollar amounts, and
    why they still reconstruct the exact predicted price when combined).

    UI_AMENDMENTS.md entry 20: redesigned from a filled horizontal bar
    chart into a "lollipop" chart (a thin stem from zero to the value,
    with a circular dot at the tip) per explicit request for "your own
    made same charts but in neat design but with the same exact statistics
    and variables" -- every number/grouping here is UNCHANGED from the
    bar-chart version (same `_group_top_features_by_label` data, same
    catch-all bucket, same price-consistency-guarded title), only the
    visual mark type, row spacing, and gridline weight changed."""
    shap = result.shap
    # AUDIT FIX 2026-09-09 (SA1/SA23, final audit): grouping logic now
    # imported from step6_llm_narrative (the canonical home for feature-
    # display-name logic) instead of a chart-local copy -- labels come
    # back lowercase (correct for the narrative paths that share this same
    # function); `.capitalize()` here is the one piece of genuinely
    # chart-specific presentation.
    grouped = group_top_features_by_label(shap["top_features"])

    labels = [g["label"].capitalize() for g in grouped] + [
        f"All other factors combined (n={shap['catchall_n_features']})"
    ]
    pct_values = [g["pct_effect"] * 100 for g in grouped] + [shap["catchall_pct_effect"] * 100]
    colors = [
        STEP6_PALETTE["increase"] if v > 0 else STEP6_PALETTE["decrease"] if v < 0 else STEP6_PALETTE["neutral"]
        for v in pct_values[:-1]
    ] + [STEP6_PALETTE["neutral"]]

    # Reverse so the largest effect is at the TOP of the chart.
    labels, pct_values, colors = labels[::-1], pct_values[::-1], colors[::-1]
    y_pos = [i * SHAP_ROW_SPACING for i in range(len(labels))]

    # UI_AMENDMENTS.md entry 23b: the "+N.N%"/"-N.N%" figure moves from a
    # data-anchored label out along the lollipop's own line (entry 20's
    # `place_value_labels`, still used/tested elsewhere -- see its
    # docstring) into the row's own Y-AXIS LABEL, as a second line below
    # the feature name. This also sidesteps entry 20's overflow-avoidance
    # machinery entirely: a label anchored to the fixed-width tick-label
    # margin never depends on where its data point happens to land.
    tick_labels = [f"{label}\n{v:+.1f}%" for label, v in zip(labels, pct_values)]

    title = f"What drove this ${result.predicted_price:,.0f} estimate"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y, v, c in zip(y_pos, pct_values, colors):
        ax.hlines(y=y, xmin=0, xmax=v, color=c, linewidth=3, alpha=0.9, zorder=2, capstyle="round")
        ax.scatter([v], [y], color=c, s=95, zorder=3, edgecolors=DARK_BG, linewidths=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-SHAP_ROW_SPACING * 0.65, y_pos[-1] + SHAP_ROW_SPACING * 0.65)
    ax.axvline(0, color=STEP6_PALETTE["neutral"], linewidth=1)
    ax.set_xlabel("Effect on predicted price (%)")
    ax.set_title(title)
    _style_dark_axes(fig, ax)
    # Lighter gridlines than the old bar-chart version -- the lollipop
    # marks themselves already carry most of the visual weight.
    ax.grid(axis="x", color=STEP6_PALETTE["grid"], linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # AUDIT FIX 2026-09-09 (SB2, final audit): read the title back from the
    # Axes object matplotlib actually holds, instead of returning the local
    # `title` variable used to build it. Previously this function returned
    # `title` itself, so build_all_plots()'s "price-consistency guard"
    # (below) was comparing that same string against itself -- it could
    # never detect a real mismatch, e.g. a future edit that called
    # `ax.set_title(something_else)` without also updating this variable.
    # Reproduced live: monkeypatching `Axes.set_title` to draw a wrong
    # string while leaving `title` untouched made the old guard pass
    # silently. `ax.get_title()` reflects whatever was actually drawn.
    rendered_title = ax.get_title()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    return str(save_path), rendered_title


def place_value_labels(fig, ax, y_positions: list[float], pct_values: list[float]) -> None:
    """UI_AMENDMENTS.md entry 23b: no longer called by `plot_shap_drivers`
    (the percentage moved into the row's own y-axis tick label instead of
    a data-anchored annotation) -- kept as-is, still covered by its own
    regression test in `tests/test_step6_plots.py`, since the precise
    bbox-measurement overflow-avoidance technique here is independently
    useful and this project's rule 11 prefers keeping one well-tested
    implementation over deleting it just because its one caller changed.

    Draws a "+N.N%"/"-N.N%" label just OUTSIDE each row's data point --
    factored out of plot_shap_drivers so this specific behavior (and the
    bug below) is independently testable without building a whole
    InterpretationResult. UI_AMENDMENTS.md entry 20: generalized from
    `place_bar_value_labels(fig, ax, bars, pct_values)` (which read each
    row's center from a `Rectangle` bar patch) to take plain `y_positions`
    directly, so the exact same tested overflow-avoidance logic backs both
    the original bar chart and the new lollipop chart -- rule 11, one
    label-placement implementation, not two.

    A label placed just outside its value's tip with a small fixed offset
    looks fine for a short value, but for a value long enough that its tip
    already sits near the axis edge, the label's own width (not just its
    anchor point) can spill past that edge into the tick-label margin,
    since matplotlib doesn't clip ax.text by default. Found by actually
    rendering a chart with an 82%/-10% spread: the -10% row's label
    visually merged into its own row's category name -- and a first fix
    using a fixed magnitude threshold (e.g. "inside if abs(v) > 12% of the
    axis span") still missed that exact case, since the real overflow
    depends on the label's actual pixel width, not a guessed cutoff. This
    instead PRECISELY MEASURES each label after a real draw.

    UI_AMENDMENTS.md entry 20: the original fix (still visible in git
    history/this project's own log) repositioned an overflowing label to
    sit INSIDE the bar in dark-on-light text -- which read fine on a wide
    filled bar, but on this chart's new thin lollipop line there's no wide
    fill to place dark text ON, so that same fallback rendered as
    dark-on-navy text nearly invisible against the page background,
    visually smearing into the line and dot (found the same way as
    always: by actually rendering the redesigned chart and looking at it,
    not by assuming the old fix still applied). Replaced with a cleaner
    approach that works for BOTH chart styles: every label stays outside
    in the same light, always-legible color, and instead the AXIS ITSELF
    is widened by exactly the measured overflow (plus a small safety
    margin) so nothing needs to be repositioned or recolored at all."""
    # UI_AMENDMENTS.md entry 18: offset widened (0.3->0.5) as extra
    # breathing room around each percentage label -- an automated scan
    # (precise bbox measurement, the same technique this function already
    # uses, not a guessed threshold) across 150+ diverse real houses from
    # the test set found zero label/label or label/category-name overlaps
    # with the OLD offset already, so this is a safety margin on top of a
    # state that measured clean, not a reproduced-bug fix.
    xlim = ax.get_xlim()
    pending = []
    for y, v in zip(y_positions, pct_values):
        text_str = f"{v:+.1f}%"
        offset = 0.5 if v >= 0 else -0.5
        ha = "left" if v >= 0 else "right"
        t = ax.text(v + offset, y, text_str, va="center", ha=ha, fontsize=8, color=DARK_TEXT)
        pending.append((t, v))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_right_overflow = 0.0
    max_left_overflow = 0.0
    for t, v in pending:
        bbox_data = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        if v >= 0:
            max_right_overflow = max(max_right_overflow, bbox_data.x1 - xlim[1])
        else:
            max_left_overflow = max(max_left_overflow, xlim[0] - bbox_data.x0)

    if max_right_overflow > 0 or max_left_overflow > 0:
        safety = 1.25  # a little extra beyond the measured overflow itself
        ax.set_xlim(xlim[0] - max_left_overflow * safety, xlim[1] + max_right_overflow * safety)


def _percentile_rank(value: float, sorted_values: np.ndarray) -> float:
    """Standard 'mean' percentile-rank definition (average of the strict-
    below and at-or-below counts) computed with plain numpy, so this
    doesn't need scipy (`percentileofscore`) as a new project dependency
    just for one calculation. UI_AMENDMENTS.md entry 26."""
    arr = np.asarray(sorted_values, dtype=float)
    n = len(arr)
    if n == 0:
        return 50.0
    below = int(np.sum(arr < value))
    equal = int(np.sum(arr == value))
    return float((below + 0.5 * equal) / n * 100.0)


def plot_tier1_range(result, save_path: Path) -> tuple[str, str, float]:
    """This house's price positioned within its own Tier 1
    (combined_category) price distribution -- house-type/price-tier
    similarity, one of the two legitimately different comparison lenses
    from Decision 6b (the other being Tier 2's geographic comps, plotted
    separately below -- never forced onto one shared baseline).

    UI_AMENDMENTS.md entry 26: redesigned from a min-max range bar with a
    separate category-average diamond into a single percentile-position
    gauge, per the user's own design choices: (1) the gauge spans the
    category's 5th-95th percentile rather than raw min/max, so one outlier
    sale can't stretch the scale; (2) the average marker is dropped --
    this chart now shows exactly one thing, where THIS house sits; (3) the
    gold gradient track + gold-pin-with-cream-border styling deliberately
    matches the brand's Low/Estimate/High price gauge widget
    (`render_value_gauge` in app.py) even though this is a saved
    matplotlib PNG, not that HTML widget, so the report reads as one
    consistent visual system; (4) the "priced higher than X%" callout
    itself is NOT drawn on the chart -- it's returned here so the caller
    can put it in the caption text instead."""
    profile = result.tier1_profile
    # AUDIT FIX 2026-09-09 (SA11, final audit): `digit_free_category_label`
    # is now a lookup by `combined_category` into the classification
    # tables' `friendly_category` field (see step6_llm_narrative.py), not a
    # function of (physical_cluster, price_band, n_bands) -- this also
    # retires the n_bands plumbing this call site used to need (the
    # `build_all_plots()` caller no longer has to temporarily mutate
    # `result.tier1_profile` with a `_n_bands_for_label` hint just to get a
    # label rendered; see that function's own updated comment).
    from step6_classify_house import load_classification_tables
    class_tables = load_classification_tables()
    category_label = digit_free_category_label(result.combined_category, class_tables)

    cluster_stats = _load_cluster_profile_stats().get(str(result.physical_cluster))

    cat_prices = np.asarray(profile["sorted_prices"], dtype=float)
    p5, p95 = (float(v) for v in np.percentile(cat_prices, [5, 95]))
    pct_rank = _percentile_rank(result.predicted_price, cat_prices)
    if p95 - p5 < 1.0:
        # Degenerate/near-constant category price spread (possible for a
        # very small or unusually uniform category) -- pad so the gauge
        # still renders as a visible span instead of a zero-width line.
        #
        # AUDIT FIX 2026-09-09 (SB21e, final audit): confirmed unreachable
        # with the shipped data, not merely assumed -- measured p5-p95
        # directly for all 18 real Tier 1 categories in `classification_
        # tables.json`: smallest row count is 108 (`Physical Group 2 -
        # Price Band 4`), and the smallest p5-p95 SPAN across all 18 is
        # $242,002 (`Physical Group 2 - Price Band 0`, 179 rows) -- both
        # far above this $1.0 threshold. Kept as a defensive guard for a
        # future retrain/dataset that could produce a genuinely tiny or
        # near-uniform category, not because it fires today.
        pad = max(abs(p5) * 0.05, 1000.0)
        p5, p95 = p5 - pad, p95 + pad

    title = f"${result.predicted_price:,.0f} vs. similar-type homes ({category_label})"
    fig, ax = plt.subplots(figsize=TIER1_GAUGE_FIGSIZE)

    # Gold gradient track: same brand-gold color as the price gauge's CSS
    # track (rgba(212,175,55,0.22) -> rgba(212,175,55,0.9)), reproduced as
    # two solid blends of that gold over the dark navy background since a
    # LineCollection needs concrete RGB stops, not CSS alpha.
    def _blend(fg_hex: str, bg_hex: str, alpha: float) -> tuple:
        fg, bg = to_rgb(fg_hex), to_rgb(bg_hex)
        return tuple(alpha * f + (1 - alpha) * b for f, b in zip(fg, bg))

    track_cmap = LinearSegmentedColormap.from_list(
        "gauge_track", [_blend(STEP6_PALETTE["subject"], DARK_BG, 0.22),
                         _blend(STEP6_PALETTE["subject"], DARK_BG, 0.9)],
    )
    n_seg = 200
    xs = np.linspace(p5, p95, n_seg)
    pts = np.array([xs, np.zeros_like(xs)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(
        segments, colors=[track_cmap(t) for t in np.linspace(0, 1, len(segments) - 1)]
        if len(segments) > 1 else [track_cmap(0.5)],
        linewidths=14, capstyle="round", zorder=2,
    )
    ax.add_collection(lc)

    # Soft outer glow behind the pin, mimicking the widget's
    # `box-shadow: 0 0 0 4px rgba(212,175,55,0.25)` ring.
    ax.scatter([result.predicted_price], [0], s=1400, color=STEP6_PALETTE["subject"],
               alpha=0.18, zorder=3, linewidths=0)
    # The pin itself: gold fill, cream border -- exactly `.vg-pin`'s colors.
    ax.scatter([result.predicted_price], [0], s=340, color=STEP6_PALETTE["subject"],
               edgecolors=DARK_TEXT, linewidths=2.5, zorder=4)
    ax.text(result.predicted_price, 0.30, "This house", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=DARK_TEXT)

    # End labels -- echo the widget's Low/High tags. No middle "Estimate"
    # number here: entry 26 dropped the average marker so this chart shows
    # one thing only, this house's position in the spread. The "priced
    # higher than X%" figure itself deliberately does NOT appear on the
    # chart -- see the caller for the caption text using `pct_rank`.
    ax.text(p5, -0.30, f"Lower typical\n${p5:,.0f}", ha="left", va="top",
            fontsize=8, color=DARK_TEXT_MUTED)
    ax.text(p95, -0.30, f"Higher typical\n${p95:,.0f}", ha="right", va="top",
            fontsize=8, color=DARK_TEXT_MUTED)

    # The axis bounds must cover the PIN too, not just the p5-p95 track --
    # found by actually rendering a house whose price sits outside its own
    # category's 5th-95th percentile band (a real, unremarkable case: a
    # house can legitimately be pricier or cheaper than "typical" for its
    # category), which cut the pin and its "This house" label off the
    # right edge of the chart. Same overflow-handling philosophy as entry
    # 20's SHAP label fix -- widen the axis to fit what's actually being
    # drawn, rather than assuming the data always stays inside p5-p95.
    plot_lo = min(p5, result.predicted_price)
    plot_hi = max(p95, result.predicted_price)
    pad = max((plot_hi - plot_lo) * 0.10, 1.0)
    ax.set_xlim(plot_lo - pad, plot_hi + pad)
    # UI_AMENDMENTS.md entry 27: tightened from (-0.9, 0.9) -- that range
    # left a lot of dead navy space above "This house" and below the end
    # labels once combined with the new, shorter `TIER1_GAUGE_FIGSIZE`.
    # This range is pure layout (the y-axis carries no data meaning here,
    # same as Tier 2's comps chart), so it's free to tighten.
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(title)
    # AUDIT FIX 2026-09-09 (SB2, final audit): see plot_shap_drivers' fix
    # comment above -- read the title back from the Axes rather than
    # returning the pre-set_title local variable.
    rendered_title = ax.get_title()
    _style_dark_axes(fig, ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    subtitle_parts = [f"based on {profile['count']} homes in this category"]
    if cluster_stats:
        subtitle_parts.append(
            f"typically ~{cluster_stats['mean_sqft']:.0f} sqft, ~{cluster_stats['mean_age']:.0f} years old"
        )
    # UI_AMENDMENTS.md entry 27: moved up from -0.62 (which sat far below
    # the end labels, creating a big empty gap) to just beneath them.
    ax.text(0.5, -0.40, " -- ".join(subtitle_parts), transform=ax.transAxes, ha="center",
            fontsize=8.5, color=DARK_TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    return str(save_path), rendered_title, pct_rank


def plot_tier2_comps(result, save_path: Path) -> str:
    """This house's predicted price + its 90% CQR interval vs. real nearby
    comparable sales -- the geographic-similarity lens (Decision 6b's
    second, deliberately separate baseline from Tier 1's category lens
    above; a real divergence between the two is surfaced honestly, not
    hidden, per `market["market_divergence"]` in the narrative)."""
    tier2 = result.tier2
    comp_prices = np.asarray(tier2["comp_prices"], dtype=float)

    # AUDIT FIX 2026-09-09 (SB15, final audit): this used to be a FOURTH
    # independent copy of the tier2-tier-used phrase mapping (alongside
    # step6_market_comparison.TIER2_TIER_PHRASES and step6_llm_narrative's
    # TIER2_LOCATION_FALLBACK_LABELS/FLAG_LABELS), and its `.get(key, key)`
    # fallback silently printed the RAW enum string (e.g.
    # "nearest_neighbors") straight into a user-facing chart title if the
    # value were ever unrecognized -- exactly the kind of silent enum-
    # drift masking this finding is about. Now reuses the one canonical
    # mapping, and an unrecognized value says so explicitly instead of
    # rendering as if it were legitimate, human-facing copy.
    tier_phrase = tier2_tier_phrase(tier2["tier2_tier_used"])
    tier_used_label = tier_phrase if tier_phrase else f"unrecognized comparison basis: {tier2['tier2_tier_used']!r}"
    title = f"${result.predicted_price:,.0f} vs. {tier2['n_comps_found']} nearby comparable sales ({tier_used_label})"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(comp_prices))
    # Soft brand-gold "glow" for the likely-range band (low alpha -- this
    # used to be a solid pastel fill on a white background, but on the dark
    # navy theme a low-alpha gold wash reads as an elegant highlighted zone
    # instead of a flat block).
    ax.axvspan(result.cqr_lo, result.cqr_hi, color=STEP6_PALETTE["interval_fill"], alpha=0.18,
               label=_esc_mpl_dollars(f"This home's likely range (${result.cqr_lo:,.0f}-${result.cqr_hi:,.0f})"))
    ax.scatter(comp_prices, y_jitter, color=STEP6_PALETTE["comp_dot"], alpha=0.8, s=40,
               label=f"Nearby comparable sales (n={tier2['n_comps_found']})")
    ax.scatter([result.predicted_price], [0], color=STEP6_PALETTE["subject"], s=220, zorder=4,
               marker="*", edgecolors=DARK_BG, linewidths=0.6,
               label=f"This house's estimate (${result.predicted_price:,.0f})")

    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Price ($)")
    ax.set_title(title)
    # AUDIT FIX 2026-09-09 (SB2, final audit): see plot_shap_drivers' fix
    # comment above -- read the title back from the Axes rather than
    # returning the pre-set_title local variable.
    rendered_title = ax.get_title()
    _style_dark_axes(fig, ax)
    if tier2["comp_evidence"] != "strong":
        ax.text(0.5, -0.35, f"Note: comparable-sales evidence is '{tier2['comp_evidence']}' for this house",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=STEP6_PALETTE["warning"])
    _style_legend(ax.legend(loc="upper left", fontsize=8, framealpha=0.95))
    ax.grid(axis="x", color=STEP6_PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    return str(save_path), rendered_title


def build_all_plots(result, output_dir: Path, house_id: str = "house") -> dict:
    """Builds all 3 plots for one already-computed InterpretationResult,
    writes them to `output_dir`, sets `result.plot_paths`, and returns the
    same dict. Asserts the one non-negotiable requirement from Decision 6b
    before returning: every plot's title contains the EXACT predicted-price
    string, never a separately formatted/rounded one."""
    with _BUILD_ALL_PLOTS_LOCK:
        if not result.ok:
            raise ValueError("build_all_plots called on a rejected (ok=False) InterpretationResult")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # AUDIT FIX 2026-09-09 (SB20, final audit): evict old plot groups
        # before writing this house's, so the directory never accumulates
        # more than MAX_CACHED_HOUSE_PLOTS distinct houses' files.
        _evict_old_plot_groups(output_dir, keep_house_id=house_id)

        # AUDIT FIX 2026-09-09 (SA11, final audit): this used to temporarily
        # mutate `result.tier1_profile` (a dict borrowed from the shared,
        # `@st.cache_resource`-cached classification tables -- see SB10) just to
        # smuggle an `_n_bands_for_label` hint into `plot_tier1_range()` for the
        # old, invented tier-word label scheme. That scheme (and the n_bands it
        # needed) is gone -- `digit_free_category_label()` now looks up the
        # canonical `friendly_category` string directly by `combined_category`
        # -- so this mutate/restore dance is no longer needed at all, which
        # also removes one instance of SB10's shared-object-mutation risk
        # rather than just working around it here.
        shap_path, shap_title = plot_shap_drivers(result, output_dir / f"{house_id}_shap_drivers.png")
        tier1_path, tier1_title, tier1_pct_rank = plot_tier1_range(
            result, output_dir / f"{house_id}_tier1_range.png"
        )
        tier2_path, tier2_title = plot_tier2_comps(result, output_dir / f"{house_id}_tier2_comps.png")

        # UI_AMENDMENTS.md entry 26: the percentile-position gauge's rank is
        # computed once here and stored on the result, so app.py's caption
        # reuses this exact number instead of recomputing it independently.
        result.tier1_percentile_rank = tier1_pct_rank

        # Step 24's plot-title price-consistency guard: check the ACTUAL title
        # string each plot function put on its own axes (not a second,
        # independently-formatted copy of the price), so a future edit that
        # changes one title's formatting but not the price it embeds fails loudly.
        # AUDIT FIX 2026-09-09 (SB2, verification pass): this guard must fire
        # even when the interpreter is run with `-O` (which strips bare
        # `assert` statements entirely -- confirmed by reproduction that the
        # previous bare `assert` silently let a wrong-title PNG through under
        # `python3 -O`). Use an explicit check + raise so it can never be
        # optimized away, matching the choice already made for SB22's guard.
        price_str = f"${result.predicted_price:,.0f}"
        for name, title in [("shap_drivers", shap_title), ("tier1_range", tier1_title), ("tier2_comps", tier2_title)]:
            if price_str not in title:
                raise AssertionError(
                    f"price-consistency guard failed: {name}'s plot title {title!r} does not contain "
                    f"the exact predicted-price string {price_str!r}."
                )

        paths = {"shap_drivers": shap_path, "tier1_range": tier1_path, "tier2_comps": tier2_path}
        result.plot_paths = paths
        return paths
