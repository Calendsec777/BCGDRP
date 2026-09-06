"""Geometry and style contract for the BCGDRP figure set.

Two rules in here exist because breaking them is invisible until a production
editor catches it.

Authoring at final size.  `bbox_inches="tight"` couples the canvas size to the
font size, so raising a label widens the figure and the extra downscaling at
insertion cancels the gain exactly.  The previous builder used it, which is why
four of five figures rendered at 218-338 mm wide instead of 180.  Every save here
goes through `save()`, which disables it and asserts the rendered geometry.

Lightness is an ordered channel.  The nominal palette varies hue at fixed chroma
for unordered categories; the ladder varies lightness for ordered quantities.
Mixing them lets lightness smuggle an ordering into a categorical variable, so
the two sets are disjoint and `assert_palette_disjoint()` checks it.
"""
import os

import matplotlib
import numpy as np
from matplotlib.container import BarContainer
from matplotlib.text import Text

MM = 1.0 / 25.4

# Nature-family column widths.  Tolerance is 0.2 mm: 1 mm would pass a figure
# that is genuinely out of spec, and 0.015 mm is backend quantisation noise.
W_1COL, W_1HALF, W_2COL = 88.0, 135.0, 180.0
H_MAIN = 200.0          # 78% of the 255 mm text block; the longest caption is 46 mm
H_SUPP_MAX = 225.0
TOL_MM = 0.2

INK = "#28303F"
# Full-wheel categorical palette measured in the approved comparison figures.
# Category identity still receives a second channel (marker, line style, or
# hatch) wherever the plotted mark supports one.
NOMINAL = ["#9D5061", "#8A6635", "#69AF87", "#3CA6CF", "#80679F"]
NOMINAL_EXTENDED = NOMINAL + ["#999933"]
NEUTRAL = "#888888"
LADDER = ["#667DB8", "#CB7F9F", "#A4A8B6", "#CAAEDF", "#9CD0F3", "#F3CFED", "#DEEFFF"]

MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), ":"]
BAR_HATCH = ["", "///", "\\\\", "...", "xxx"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # mathtext has its own font stack and ignores font.sans-serif entirely, so a
    # figure that is Arial everywhere a person looks still renders $...$ in DejaVu.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    # journals want live, editable text; this is required here and banned at AAAI
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.0,
    "figure.dpi": 150,
}


def apply():
    matplotlib.rcParams.update(RC)


def _luminance(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def assert_palette_disjoint():
    """Mirror assertions: the ladder must survive greyscale, the nominal set must not order."""
    lum = [_luminance(c) for c in LADDER]
    gaps = np.diff(sorted(lum))
    assert gaps.min() >= 0.05, f"ordered ladder adjacent luminance gap {gaps.min():.3f} < 0.05"
    nl = [_luminance(c) for c in NOMINAL]
    spread = max(nl) - min(nl)
    assert spread <= 0.22, f"nominal luminance spread {spread:.3f} > 0.22, it encodes an order"
    assert not (set(NOMINAL) & set(LADDER)), "a colour appears in both palettes"


def figure_mm(fig):
    return tuple(v / MM for v in fig.get_size_inches())


def grid_mm(fig, nrows, ncols, left=15.0, right=3.0, top=6.0, bottom=13.0,
            wgap=5.0, hgap=5.0, width_ratios=None, height_ratios=None,
            rows_mm=None):
    """A gridspec whose inter-panel gaps are specified in millimetres.

    matplotlib's wspace is a fraction of the average axis width, so asking for a
    fixed gap means solving for it. Specifying gaps as a fraction is why every
    figure here used to sit at 21 mm: wspace=0.62 reads as modest and is not.

    A gap of 5 mm only works between panels that share an axis. A y-axis label
    plus its tick labels needs about 12 mm, so unshared neighbours need that much
    and `assert_layout` reports which is which.
    """
    W, H = figure_mm(fig)
    span_w = W - left - right
    span_h = H - top - bottom
    # Explicit row heights in millimetres. Ratios make a panel's size a consequence
    # of the other panels; a heatmap of six rows needs a stated height, not a share.
    if rows_mm is not None:
        assert len(rows_mm) == nrows, f"{len(rows_mm)} heights for {nrows} rows"
        used = sum(rows_mm) + (nrows - 1) * hgap
        assert used <= span_h + 0.05, (f"rows plus gaps need {used:.1f} mm, "
                                       f"only {span_h:.1f} mm available")
        bottom = bottom + (span_h - used)      # keep the block against the top
        span_h = used
        height_ratios = list(rows_mm)
    wspace = wgap / ((span_w - (ncols - 1) * wgap) / ncols) if ncols > 1 else 0.0
    hspace = hgap / ((span_h - (nrows - 1) * hgap) / nrows) if nrows > 1 else 0.0
    return fig.add_gridspec(nrows, ncols,
                            left=left / W, right=1 - right / W,
                            bottom=bottom / H, top=1 - top / H,
                            wspace=wspace, hspace=hspace,
                            width_ratios=width_ratios, height_ratios=height_ratios)


def headroom(ax, frac=0.13):
    """Open an empty band at the top of the axes for the panel letter.

    The letter belongs at the top left inside its own panel. When a plot fills
    that corner the letter has to step outside, and on a narrow panel outside
    means on top of the title. Giving the data a little headroom keeps the letter
    where it should be and costs nothing but a slightly wider y range.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * frac)
    return ax


def panel_label(fig, ax, letter, dx=1.4, dy=-1.4, fontsize=8, ha="left",
                va="top", auto=True):
    """The letter always sits at the top-left corner of its own axes.

    Uniform placement is the point: a reader should not have to hunt for the
    letter. It goes inside the axes at the top left, and when something is drawn
    there it moves outward along the same corner rather than jumping to a
    different one, so it may overhang slightly but never appears elsewhere.
    """
    t = None
    # inside first, then progressively outward from the same corner
    # Inside first, then progressively outward from the same corner. A panel that
    # is entirely filled, a heatmap being the usual case, has no clear interior at
    # all, so the last position sits fully outside the corner rather than on data.
    for ox, oy in ((dx, dy), (dx, 2.2), (-2.6, 2.2), (-4.2, 3.4), (-8.5, 8.5)):
        if t is not None:
            t.remove()
        t = ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(ox, oy), textcoords="offset points",
                        fontsize=fontsize, fontweight="bold", color=INK,
                        ha="left", va="top", zorder=12, annotation_clip=False)
        t.set_in_layout(False)
        if not auto:
            return t
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tb = t.get_window_extent(r)
        if _points_in_bbox(ax, tb):
            continue
        clash = False
        others = [o for o in ax.texts if o is not t and o.get_text().strip()]
        others += [lg for lg in ax.findobj(match=lambda x: hasattr(x, "get_texts"))
                   if getattr(lg, "axes", None) is ax]
        # Stepping outward puts the letter where the tick labels live, and those
        # are not in ax.texts, so without this the letter only moved from sitting
        # on the data to sitting on a tick label.
        others += [o for o in (ax.get_xticklabels() + ax.get_yticklabels())
                   if o.get_text().strip()]
        others += [o for o in (ax.xaxis.label, ax.yaxis.label, ax.title)
                   if o is not None and o.get_text().strip()]
        for other in others:
            try:
                ob = other.get_window_extent(r)
            except Exception:
                continue
            if tb.x0 < ob.x1 and ob.x0 < tb.x1 and tb.y0 < ob.y1 and ob.y0 < tb.y1:
                clash = True
                break
        if not clash:
            return t
    return t


def assert_layout(fig, max_gap_mm=5.0, verbose=True):
    """Measure the realised layout instead of trusting the parameters.

    Fails on a panel letter that escapes its axes or covers a drawn point. Gaps
    wider than the target are reported with whether the pair shares an axis, so
    an unshared 12 mm gap reads as a decision and a shared one reads as a bug.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = figure_mm(fig)
    axes = [a for a in fig.axes if a.get_visible()]
    problems, wide = [], []

    for ax in axes:
        ab = ax.get_window_extent(r)
        for t in ax.texts:
            s = t.get_text()
            if len(s) != 1 or not s.isalpha() or t.get_fontweight() != "bold":
                continue
            tb = t.get_window_extent(r)
            # points. A letter may overhang its own corner when the top-left of
            # the panel is occupied. A filled panel, a heatmap being the usual
            # case, has no clear interior at all, so the letter has to sit just
            # outside it; 22 pt is about 7.8 mm, which is a corner marker rather
            # than a letter adrift on the page.
            slack = 22
            if not (ab.x0 - slack <= tb.x0 and tb.x1 <= ab.x1 + 1
                    and ab.y0 - 1 <= tb.y0 and tb.y1 <= ab.y1 + slack):
                problems.append(f"panel letter '{s}' is far outside its axes")
            n = _points_in_bbox(ax, tb)
            if n:
                problems.append(f"panel letter '{s}' covers {n} drawn points")

    px_per_mm = fig.get_dpi() / 25.4
    boxes = [(a, a.get_window_extent(r)) for a in axes]

    def blocked(pa, pb, horizontal):
        """True when a third panel lies in the corridor between these two."""
        lo, hi = (min(pa.x1, pb.x1), max(pa.x0, pb.x0)) if horizontal else \
                 (min(pa.y1, pb.y1), max(pa.y0, pb.y0))
        for _, pc in boxes:
            if pc is pa or pc is pb:
                continue
            if horizontal:
                if pc.x0 < hi - 1 and pc.x1 > lo + 1 and \
                   min(pa.y1, pb.y1) - max(pc.y0, max(pa.y0, pb.y0)) > 1:
                    return True
            else:
                if pc.y0 < hi - 1 and pc.y1 > lo + 1 and \
                   min(pa.x1, pb.x1) - max(pc.x0, max(pa.x0, pb.x0)) > 1:
                    return True
        return False

    for i, (a, pa) in enumerate(boxes):
        for b, pb in boxes[i + 1:]:
            v_over = min(pa.y1, pb.y1) - max(pa.y0, pb.y0) > 1
            h_over = min(pa.x1, pb.x1) - max(pa.x0, pb.x0) > 1
            gap = shared = None
            if v_over and not blocked(pa, pb, True):
                gap = (max(pa.x0, pb.x0) - min(pa.x1, pb.x1)) / px_per_mm
                shared = a.get_shared_y_axes().joined(a, b)
            elif h_over and not blocked(pa, pb, False):
                gap = (max(pa.y0, pb.y0) - min(pa.y1, pb.y1)) / px_per_mm
                shared = a.get_shared_x_axes().joined(a, b)
            if gap is not None and gap > max_gap_mm + 0.05:
                wide.append((gap, shared, a.get_title()[:26], b.get_title()[:26]))

    if verbose:
        tight = sum(1 for g, *_ in wide if False)
        print(f"    layout: {len(axes)} panels, {len(wide)} neighbour gaps above {max_gap_mm:.0f} mm")
        for g, sh, ta, tb in sorted(wide, reverse=True)[:6]:
            tag = "shared axis, SHOULD BE TIGHT" if sh else "own scale, expected"
            print(f"      {g:5.1f} mm  {tag:28s} {ta} | {tb}")
    # Anything rendering outside the canvas is clipped in the PDF and the compile
    # log never mentions it. Checked here because it is invisible until someone looks.
    fw, fh = fig.canvas.get_width_height()
    for ax in axes:
        for art in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if art is None or not art.get_text().strip():
                continue
            try:
                e = art.get_window_extent(r)
            except Exception:
                continue
            if e.x0 < -1 or e.y0 < -1 or e.x1 > fw + 1 or e.y1 > fh + 1:
                problems.append(f"text '{art.get_text()[:24]}' extends outside the canvas")

    for p in problems:
        print(f"    LAYOUT FAIL: {p}")
    assert not problems, problems
    bad = [w for w in wide if w[1]]
    assert not bad, f"{len(bad)} shared-axis neighbours are further apart than {max_gap_mm} mm"
    return wide


def assert_aspect(fig, max_ratio=1.05, row_limited=(), verbose=True):
    """Fail when a lettered panel is taller than it is wide.

    An ordinary data panel should be at most as tall as it is wide. A panel that
    is taller than that is almost never a considered choice: it is what happens
    when the canvas height is fixed first and the rows are stretched to fill it,
    which buys page area at the cost of every panel on the page. Annotation
    tracks and colour bars carry no letter and are exempt, because a one-cell
    strip is meant to be thin.

    `row_limited` names panels whose height is set by a category count rather
    than by the layout, such as a ranked list of every compound. Name one only
    after checking the millimetres per row: a list at three millimetres a row is
    row-limited, and the same panel at nine is stretched.
    """
    bad = []
    for g in panel_geometry(fig):
        if g["label"] == "?" or g["w"] <= 0 or g["label"] in row_limited:
            continue
        r = g["h"] / g["w"]
        if r > max_ratio:
            bad.append((g["label"], g["w"], g["h"], r))
    if verbose and bad:
        for l, w, h, r in sorted(bad, key=lambda t: -t[3]):
            print(f"    panel {l}: {w:.1f} x {h:.1f} mm, h/w {r:.2f}")
    assert not bad, (
        "%d lettered panel(s) taller than wide, worst h/w %.2f"
        % (len(bad), max(b[3] for b in bad)))
    return True


def assert_margins(fig, max_bottom_mm=6.0, max_top_mm=6.0, verbose=True):
    """Fail when the canvas carries dead space past the drawn content.

    Measured from the figure's own tight bounding box, so it counts tick labels
    and annotations, not just the axes rectangles. The bottom is the one that
    matters most in a manuscript: whitespace there pushes the caption away from
    the figure it belongs to and reads as a layout accident.
    """
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fw, fh = (v / MM for v in fig.get_size_inches())
    bottom = bb.y0 / MM
    top = fh - bb.y1 / MM
    if verbose:
        print(f"    margins: top {top:.1f} mm, bottom {bottom:.1f} mm")
    assert bottom <= max_bottom_mm, (
        "%.1f mm of dead space below the drawn content, limit %.1f"
        % (bottom, max_bottom_mm))
    assert top <= max_top_mm, (
        "%.1f mm of dead space above the drawn content, limit %.1f"
        % (top, max_top_mm))
    return True


def assert_no_overlap(fig, tol_mm=0.2, verbose=True):
    """Fail when two panel rectangles intersect.

    Occlusion between panels is the one layout fault a reader cannot work
    around, so it is checked on the axes rectangles rather than inferred.

    Full containment is exempt. An axes drawn entirely inside another is an
    inset or an overlay segment and is deliberate; the fault this looks for is
    a PARTIAL intersection, where two panels were placed independently and one
    has grown into the other.
    """
    gs = [g for g in panel_geometry(fig) if g["w"] > 0]
    bad = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a, b = gs[i], gs[j]
            ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
            if ox <= tol_mm or oy <= tol_mm:
                continue
            inside = lambda u, v: (u["x"] >= v["x"] - tol_mm
                                   and u["y"] >= v["y"] - tol_mm
                                   and u["x"] + u["w"] <= v["x"] + v["w"] + tol_mm
                                   and u["y"] + u["h"] <= v["y"] + v["h"] + tol_mm)
            if inside(a, b) or inside(b, a):
                continue
            bad.append((a["label"], b["label"], ox, oy))
    if verbose and bad:
        for la, lb, ox, oy in bad:
            print(f"    panels {la} and {lb} overlap by {ox:.1f} x {oy:.1f} mm")
    assert not bad, "%d overlapping panel pair(s)" % len(bad)
    return True


def assert_hero(ax, min_mm2=12432.0):
    """0.8 of an A6 page. A figure that declares a hero must actually have one.

    Accepts a list, because a hero is sometimes a composite: two landscapes on one
    colour scale read as a single display and are measured as one.
    """
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    fig = axes[0].figure
    W, H = figure_mm(fig)
    area = sum((a.get_position().width * W) * (a.get_position().height * H) for a in axes)
    b = axes[0].get_position()
    assert area >= min_mm2, (f"hero panel is {area:,.0f} mm2, below the "
                             f"{min_mm2:,.0f} mm2 floor (0.8 A6)")
    part = "" if len(axes) == 1 else f" over {len(axes)} parts"
    print(f"    hero{part}: {area:,.0f} mm2 ({area / 15540 * 100:.0f}% of A6)")
    return area


def panel_geometry(fig):
    """Every drawn axes as (label, x, y, width, height) in millimetres.

    A panel taller than it is wide is not illegal, but it is nearly always the
    result of stretching a grid to fill a canvas height that was chosen first,
    so the ratio is worth reporting rather than assuming.
    """
    fw, fh = (v / MM for v in fig.get_size_inches())
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        b = ax.get_position()
        out.append({
            "label": _panel_letter_of(ax),
            "x": b.x0 * fw, "y": b.y0 * fh,
            "w": b.width * fw, "h": b.height * fh,
        })
    return out


def _panel_letter_of(ax):
    """The panel letter this axes carries, if panel_label put one on it."""
    for t in ax.texts:
        s = t.get_text().strip()
        if len(s) <= 3 and s[:1].isalpha() and t.get_fontweight() == "bold":
            return s
    return "?"


def raise_text_floor(fig, min_pt=5.0):
    """Raise non-empty visible text below the final-size print floor."""
    repaired = 0
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if float(text.get_fontsize()) + 1e-9 < min_pt:
            text.set_fontsize(min_pt)
            repaired += 1
    return repaired


def assert_text_floor(fig, min_pt=5.0):
    """Fail when a visible label remains below the print-size floor."""
    failures = [
        (text.get_text(), float(text.get_fontsize()))
        for text in fig.findobj(match=Text)
        if text.get_visible()
        and text.get_text().strip()
        and float(text.get_fontsize()) + 1e-9 < min_pt
    ]
    assert not failures, f"text below {min_pt:g} pt: {failures[:8]}"


def apply_bar_hatches(fig):
    """Pair each bar series with a stable hatch and visible ink edge."""
    changed = 0
    for ax in fig.axes:
        containers = [item for item in ax.containers if isinstance(item, BarContainer)]
        for index, container in enumerate(containers):
            hatch = BAR_HATCH[index % len(BAR_HATCH)]
            for patch in container:
                patch.set_hatch(hatch)
                patch.set_edgecolor(INK)
                patch.set_linewidth(0.30)
                changed += 1
    return changed


def save(fig, path, width_mm=W_2COL, height_mm=H_MAIN, png=True, dpi=450):
    """Save at authored size and assert the rendered geometry."""
    raise_text_floor(fig, min_pt=5.0)
    apply_bar_hatches(fig)
    assert_text_floor(fig, min_pt=5.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(path, format="pdf")
        if png:
            fig.savefig(path[:-4] + ".png", dpi=dpi)
    w, h = (v / MM for v in fig.get_size_inches())
    assert abs(w - width_mm) <= TOL_MM, f"{path}: width {w:.3f} mm, expected {width_mm}"
    assert abs(h - height_mm) <= TOL_MM, f"{path}: height {h:.3f} mm, expected {height_mm}"
    assert height_mm <= H_SUPP_MAX, f"{path}: height {height_mm} exceeds the {H_SUPP_MAX} mm ceiling"
    print(f"  wrote {os.path.basename(path)}  {w:.3f} x {h:.3f} mm")
    dump = os.environ.get("NS_GEOMETRY_DIR")
    if dump:
        import json
        os.makedirs(dump, exist_ok=True)
        name = os.path.basename(path)[:-4]
        with open(os.path.join(dump, name + ".json"), "w") as fh:
            json.dump({"figure": name, "w_mm": w, "h_mm": h,
                       "panels": panel_geometry(fig)}, fh, indent=1)
    return path


def new_figure(width_mm=W_2COL, height_mm=H_MAIN):
    apply()
    assert_palette_disjoint()
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def dotplot(ax, labels, values, colors=None, err=None, vline=None, size=14,
            lo=None, hi=None):
    """A dot plot, used wherever a bar chart would need a non-zero axis origin.

    A bar encodes its value from zero, so an axis starting at 0.45 overstates
    every difference on it.  This lab has reshipped that defect four times, so
    ranked comparisons here are dots rather than bars.
    """
    y = np.arange(len(labels))[::-1]
    base = vline if vline is not None else 0.0
    ax.hlines(y, base, values, color=NEUTRAL, lw=0.7, zorder=2)
    ax.scatter(values, y, s=size, color=(colors or INK), zorder=3, linewidths=0)
    # Explicit bounds when the interval is asymmetric, which a bootstrap interval
    # normally is. Passing a half-width instead draws an interval the analysis
    # never produced: a stratum whose true interval was [-0.154, 0.037] was being
    # drawn as [-0.138, 0.052] because the half-width was recentred on the point.
    if lo is not None and hi is not None:
        for yi, a, b in zip(y, lo, hi):
            ax.plot([a, b], [yi, yi], color=INK, lw=0.8, zorder=4)
    elif err is not None:
        for yi, v, e in zip(y, values, err):
            ax.plot([v - e, v + e], [yi, yi], color=INK, lw=0.8, zorder=4)
    if vline is not None:
        ax.axvline(vline, color=INK, lw=0.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    return y


def _segment_hits_box(p0, p1, lb):
    """Exact segment against axis-aligned rectangle, in display coordinates.

    Sampling a segment at a fixed number of points is not good enough here: a
    full-width rule sampled twelve times steps over a nine-pixel letter box, so
    the letter was reported clear while sitting on the rule. Liang-Barsky gives
    the answer without a resolution to get wrong.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(p1[0]) - x0, float(p1[1]) - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - lb.x0), (dx, lb.x1 - x0),
                 (-dy, y0 - lb.y0), (dy, lb.y1 - y0)):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def _points_in_bbox(ax, lb):
    """How many drawn points fall inside a display-coordinate box.

    Used for two things that are the same problem: a legend sitting on the data,
    and a panel letter sitting on the data.

    An image counts as data everywhere it is drawn. Without this a panel letter
    placed inside a heatmap sat on real cells and the check reported nothing,
    because a heatmap has no scatter offsets and no line vertices.
    """
    n = 0
    for im in ax.images:
        try:
            ib = im.get_window_extent(ax.figure.canvas.get_renderer())
        except Exception:
            continue
        if lb.x0 < ib.x1 and ib.x0 < lb.x1 and lb.y0 < ib.y1 and ib.y0 < lb.y1:
            n += 1
    # A marker is tested at its centre but drawn with a radius, so the box is
    # widened by that radius before the test. Without this a letter placed two
    # points clear of a marker centre still lands on the glyph, which is what a
    # reader sees. Found on a dumbbell panel whose top row sat under its letter.
    dpp = ax.figure.dpi / 72.0
    for coll in ax.collections:
        try:
            pts = coll.get_offsets()
        except Exception:
            continue
        if pts is None or len(pts) == 0:
            continue
        try:
            sizes = coll.get_sizes()
            rad = (float(max(sizes)) ** 0.5) / 2.0 if len(sizes) else 3.0
        except Exception:
            rad = 3.0
        pad = max(rad, 1.5) * dpp
        for x, y in ax.transData.transform(pts):
            n += (lb.x0 - pad <= x <= lb.x1 + pad
                  and lb.y0 - pad <= y <= lb.y1 + pad)
    # hlines and vlines produce a LineCollection, whose get_offsets is not its
    # vertices, so the segments were invisible here and a panel letter sat on a
    # dumbbell's top rule. Sample each segment rather than only its endpoints.
    for coll in ax.collections:
        try:
            segs = coll.get_segments()
        except Exception:
            continue
        for seg in segs or ():
            if len(seg) < 2:
                continue
            pts = ax.transData.transform(np.asarray(seg, float))
            for k in range(len(pts) - 1):
                if _segment_hits_box(pts[k], pts[k + 1], lb):
                    n += 1
    for ln in ax.lines:
        xy = ln.get_xydata()
        if xy is None or len(xy) == 0 or len(xy) > 400:
            continue
        pad = max(ln.get_markersize() / 2.0, 1.5) * dpp
        for x, y in ax.transData.transform(xy):
            n += (lb.x0 - pad <= x <= lb.x1 + pad
                  and lb.y0 - pad <= y <= lb.y1 + pad)
    # A step histogram is a Polygon patch, not a Line2D. Legends are frameless in this
    # style, so an outline crossing the legend is drawn straight through the text.
    # A Rectangle's path lives in unit space and is placed by its own transform, so
    # transforming its vertices through transData gives nonsense. Use the rendered
    # extent instead, which is right for every patch type.
    try:
        r = ax.figure.canvas.get_renderer()
    except Exception:
        r = None
    for pa in ax.patches:
        if r is None:
            break
        try:
            pb = pa.get_window_extent(r)
        except Exception:
            continue
        if pb.x0 < lb.x1 and lb.x0 < pb.x1 and pb.y0 < lb.y1 and lb.y0 < pb.y1:
            n += 1
    return int(n)


def _points_under(ax, leg):
    ax.figure.canvas.draw()
    return _points_in_bbox(ax, leg.get_window_extent(ax.figure.canvas.get_renderer()))


def legend_clear(ax, max_growth=1.6, **kw):
    """Place a legend that covers no drawn point.

    A legend sitting on the data is not cosmetic. In one panel it covered the
    worst-predicted cell line, the single point that panel existed to show. This
    places the legend, measures what it hides, and opens headroom until it hides
    nothing, rather than trusting a corner to be empty.
    """
    kw.setdefault("loc", "best")
    leg = ax.legend(**kw)
    if _points_under(ax, leg) == 0:
        return leg
    # Try opening the value axis. Vertical plots gain headroom; horizontal ones,
    # where y is categorical and growing it would just add blank rows, gain width.
    for setter, getter in ((ax.set_ylim, ax.get_ylim), (ax.set_xlim, ax.get_xlim)):
        lo, hi = getter()
        span = hi - lo
        for step in (1.10, 1.20, 1.32, 1.45, max_growth):
            setter(lo, lo + span * step)
            leg.remove()
            leg = ax.legend(**kw)
            if _points_under(ax, leg) == 0:
                return leg
        setter(lo, hi)          # restore before trying the other axis
        leg.remove()
        leg = ax.legend(**kw)
    return leg


def _lch_to_hex(L, C, h_deg):
    """CIELCh -> sRGB hex, clipped to gamut."""
    import math
    h = math.radians(h_deg)
    a, b = C * math.cos(h), C * math.sin(h)
    fy = (L + 16) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116) / 7.787

    X, Y, Z = finv(fx) * 0.95047, finv(fy) * 1.0, finv(fz) * 1.08883
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gam(c):
        c = max(0.0, min(1.0, c))
        return 1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c

    return "#%02X%02X%02X" % tuple(int(round(gam(c) * 255)) for c in (r, g, bl))


def _lab(hex_):
    """CIELAB triple, for the perceptual-separation assertion below."""
    import matplotlib.colors as _mc
    r, g, b = _mc.to_rgb(hex_)
    f = lambda u: u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    X = r * .4124 + g * .3576 + b * .1805
    Y = r * .2126 + g * .7152 + b * .0722
    Z = r * .0193 + g * .1192 + b * .9505
    q = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = q(X / .95047), q(Y / 1.0), q(Z / 1.08883)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def min_delta_e(colours):
    """Smallest CIE76 distance over all pairs. Below about 11 two small print
    swatches stop being separable, which is the failure a narrow arc makes
    possible: the hues stay legal and the reader still cannot tell them apart."""
    labs = [_lab(c) for c in colours]
    return min(sum((a - b) ** 2 for a, b in zip(labs[i], labs[j])) ** 0.5
               for i in range(len(labs)) for j in range(i + 1, len(labs)))


def _checked(colours, n):
    """Refuse a set whose closest pair is no longer separable in print.

    Narrowing the arc to blue-purple-pink keeps every hue legal while quietly
    pushing categories together, so the count that fits is smaller than it was.
    This turns that into a failure instead of a figure nobody can read.
    """
    if len(colours) > 1:
        d = min_delta_e(colours)
        assert d >= 11.0, (
            "nominal_n(%d) puts two categories %.1f deltaE apart. Below 11 they "
            "stop being separable as small print swatches, so this many levels "
            "does not fit the requested arc. Name fewer levels and group the "
            "rest." % (n, d))
    return colours


def nominal_n(n):
    """Return the approved five- or six-level nominal set."""
    if not 1 <= n <= len(NOMINAL_EXTENDED):
        raise ValueError(
            f"the figure palette supports 1--{len(NOMINAL_EXTENDED)} levels, got {n}"
        )
    return NOMINAL_EXTENDED[:n]


def assert_nominal_n(n):
    cols = nominal_n(n)
    lum = [_luminance(c) for c in cols]
    spread = max(lum) - min(lum)
    assert spread <= 0.22, f"generated {n}-colour set has luminance spread {spread:.3f}"
    assert len(set(cols)) == n, "generated set contains duplicates"
    return spread


def assert_density(fig, min_per_mm2=0.010, min_area_mm2=1200.0, verbose=True):
    """Flag panels given far more area than they draw anything in.

    A heatmap of six rows stretched over 108 mm reads as dense to a pixel counter
    and carries the same information as the same strip at 16 mm. Measured here as
    drawn elements per square millimetre, which separates the two.
    """
    fig.canvas.draw()
    W, H = figure_mm(fig)
    thin = []
    for ax in fig.axes:
        b = ax.get_position()
        area = (b.width * W) * (b.height * H)
        if area < min_area_mm2:
            continue
        n = 0
        for c in ax.collections:
            try:
                n += len(c.get_offsets())
            except Exception:
                pass
        for l in ax.lines:
            n += len(l.get_xydata())
        for im in ax.images:
            n += int(im.get_array().size)
        n += len(ax.patches)
        if n / area < min_per_mm2:
            thin.append((ax.get_title()[:30] or "(untitled)", area, n, n / area))
    if verbose and thin:
        print(f"    density: {len(thin)} panel(s) below {min_per_mm2} elements/mm2")
        for t, a, n, d in sorted(thin, key=lambda r: r[3]):
            print(f"      {d:6.3f}/mm2  {a:6.0f} mm2  {n:6d} elements  {t}")
    return thin
