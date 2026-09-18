#!/usr/bin/env python3
"""Render the entropy-collapse figure set from the parsed series.

Run INSIDE the verl container (the login node has no matplotlib):

    SB=/n/netscratch/barak_lab/Lab/sqin/docker_images/verl-sgl059-latest
    singularity exec -B /n/netscratch -B /n/home05 --pwd $PWD "$SB" \
        python3 colbench/collapse/figures.py

Figures (see RESULTS.md for what each one argues):
  f1_two_phases_<run>       benign vs position-divergent entropy rise
  f2_mean_h_lies_<run>      mean H oscillates; turn0/turn3+ and the ratio do not
  f3_drift_sign_<run>       Cov over time, overall and per turn
  f4_time_to_detonation     CROSS-RUN: approach to detonation is reproducible
  f6_length_epiphenomenal   CROSS-RUN: length/turn-count sign-flip across runs

REMOVED, do not re-add without new evidence:
  f3 (old)  dH scattered against Cov -- no power; dH's batch-draw noise is ~10x
            the mean dH it was meant to explain. Replaced by f3_drift_sign.
  f5 (old)  a "cascade" heatmap claiming the divergence propagates late->early.
            The clean monotone wavefront (turn3+ 981, turn2 999, turn1 1007,
            turn0 1026) holds in ent003_replay ONLY. In simsft_s2 turn 0 never
            doubles at all -- it FALLS -- and turn 1 crosses only after
            detonation; in simsft_s2_ent003 turn 1 and turn 2 cross in the wrong
            order (422 vs 553). It was a single-run pattern dressed as a
            mechanism, the same failure mode as the per-turn-length "finding".

COLOR (per the dataviz method, validated with its script):
  Turn index is ORDERED, so buckets take a 4-step ORDINAL blue ramp
  (#86b6ef,#3987e5,#1c5cab,#0d366b -- all checks pass, light end 2.06:1), NOT
  four categorical hues. Runs are categorical identity and take the first three
  slots (blue/orange/aqua), which are the ones that validate on the all-pairs
  list; aqua is below 3:1 on the light surface so every run line is
  direct-labeled (the relief rule). Mean H is an aggregate reference, so it is
  neutral gray rather than a fifth hue competing with the ramp.
  NO DUAL-AXIS ANYWHERE: two measures of different scale get stacked panels on
  a shared x, never two y-scales.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES, FIGS = os.path.join(HERE, "series"), os.path.join(HERE, "figures")

# ── palette ───────────────────────────────────────────────────────────────────
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BUCKET = {  # ordinal blue: later turn = darker
    "turn0": "#86b6ef", "turn1": "#3987e5",
    "turn2": "#1c5cab", "turn3plus": "#0d366b",
}
MEANH = "#8a8880"          # aggregate reference, deliberately hue-less
RUNC = {"spec_simsft_s2": "#2a78d6",
        "spec_simsft_s2_ent003": "#eb6834",
        "spec_ent003_replay900_diag2": "#1baf7a"}
PHASE_B = "#f4f3f0"        # faint neutral band; never reads as data
DIVERGING = LinearSegmentedColormap.from_list(
    "bluegrayred",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a52020"],
)

RUNS = ("spec_simsft_s2", "spec_simsft_s2_ent003", "spec_ent003_replay900_diag2")
LABEL = {"spec_simsft_s2": "simsft_s2  (SFT sim)",
         "spec_simsft_s2_ent003": "simsft_s2_ent003  (SFT sim + entropy penalty)",
         "spec_ent003_replay900_diag2": "ent003_replay  (entropy penalty, from step 900)"}
SHORT = {"spec_simsft_s2": "simsft_s2",
         "spec_simsft_s2_ent003": "simsft_s2_ent003",
         "spec_ent003_replay900_diag2": "ent003_replay"}
BK = [("turn0", "turn 0"), ("turn1", "turn 1"),
      ("turn2", "turn 2"), ("turn3plus", "turn 3+")]
VK = "val-core/colbench_spec_local/reward/mean@1"


# ── data helpers ──────────────────────────────────────────────────────────────
def load(run: str) -> list[dict]:
    rows = json.load(open(os.path.join(SERIES, f"{run}.json")))
    for r in rows:
        r["step"] = int(r["step"])
    return sorted(rows, key=lambda r: r["step"])


def series(rows, key):
    """(steps, values) for one metric, missing steps dropped -- never interpolated."""
    xs = [r["step"] for r in rows if key in r]
    return xs, [r[key] for r in rows if key in r]


def ratio(rows):
    out = [(r["step"], r["actor/entropy_turn3plus"] / r["actor/entropy_turn0"])
           for r in rows
           if "actor/entropy_turn0" in r and r["actor/entropy_turn0"] > 0]
    return [s for s, _ in out], [v for _, v in out]


def smooth(xs, ys, half=10):
    """Centered mean over +-half STEPS (not +-half samples) -- gap-safe."""
    out = []
    for x in xs:
        win = [y for x2, y in zip(xs, ys) if abs(x2 - x) <= half]
        out.append(sum(win) / len(win))
    return out


def detonation(rows, thr=2.0, hold=3):
    """First step where mean H exceeds thr and stays above for `hold` samples."""
    xs, ys = series(rows, "actor/entropy")
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y > thr and all(v > thr for v in ys[i:i + hold]):
            return x
    return None


def inversion(rows, hold=20):
    """First step the SMOOTHED ratio crosses 1.0 upward and stays above."""
    xs, ys = ratio(rows)
    if not xs:
        return None
    sm = smooth(xs, ys)
    for i, (x, y) in enumerate(zip(xs, sm)):
        if y > 1.0 and all(v > 1.0 for v in sm[i:i + hold]):
            return x
    return None


def forward_diff(rows):
    """(cov[t], H[t+1]-H[t], step) -- the pairing the replicator formula needs."""
    H = {r["step"]: r["actor/entropy"] for r in rows if "actor/entropy" in r}
    C = {r["step"]: r["actor/adv_logp_cov"] for r in rows
         if "actor/adv_logp_cov" in r}
    return [(C[s], H[s + 1] - H[s], s) for s in sorted(C)
            if s in H and s + 1 in H]


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def monotonicity(vals):
    """|sum of diffs| / sum of |diffs|: 1.0 = perfectly monotone, ~0 = noise.

    Stride-free on purpose. Counting "reversals" needs a sampling stride, and
    the count then depends on the stride you picked -- which is a knob nobody
    should be able to turn after seeing the answer.
    """
    d = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    tot = sum(abs(v) for v in d)
    return abs(sum(d)) / tot if tot else float("nan")


def place_labels(ax, items, dy_pts=11.5):
    """Right-edge direct labels, pushed apart so they never overplot.

    items: [(y_data, text, color)]. Returns nothing; annotates in axes coords so
    the spreading is in POINTS and survives a log scale.
    """
    if not items:
        return
    tr = ax.transData
    inv = ax.transAxes.inverted()
    rows = []
    for y, text, color in items:
        _, py = inv.transform(tr.transform((ax.get_xlim()[1], y)))
        rows.append([py, text, color])
    rows.sort(key=lambda r: r[0])
    hgt = ax.get_window_extent().height or 1
    gap = dy_pts * (ax.figure.dpi / 72.0) / hgt
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] < gap:
            rows[i][0] = rows[i - 1][0] + gap
    for py, text, color in rows:
        ax.annotate(text, xy=(1.0, min(py, 1.0)), xycoords="axes fraction",
                    xytext=(6, 0), textcoords="offset points", color=color,
                    fontsize=8, va="center", fontweight="bold",
                    annotation_clip=False)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx * sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


# ── chrome ────────────────────────────────────────────────────────────────────
def style(ax, xlabel=None, ylabel=None, logy=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, ls="-", zorder=0)   # solid hairline, never dashed
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def nat_ticks(ax):
    """Plain decimal ticks on a log axis -- 0.1/0.2/0.5/1/2/5 reads far better
    than 10^-1 for entropies in nats."""
    lo, hi = ax.get_ylim()
    keep = [v for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)
            if lo <= v <= hi]
    ax.set_yticks(keep)
    ax.set_yticklabels(["%g" % v for v in keep])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def figure(w, h, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, 1, figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, (axes if nrows > 1 else [axes])


def title(fig, main, sub):
    fig.text(0.012, 0.975, main, color=INK, fontsize=12.5, fontweight="bold",
             va="top")
    fig.text(0.012, 0.925, sub, color=INK2, fontsize=9, va="top")


def marker(ax, x, label, color=INK2, top=0.97):
    """A vertical event rule. Solid hairline -- dashes would read as a threshold."""
    ax.axvline(x, color=color, lw=0.9, zorder=1, alpha=0.75)
    ax.annotate(label, xy=(x, top), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points",
                color=color, fontsize=7.5, va="top", ha="left")


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ── F1: the two phases ────────────────────────────────────────────────────────
def f1_two_phases(run):
    rows = load(run)
    det, inv = detonation(rows), inversion(run and rows)
    fig, (ax,) = figure(9.2, 5.0)
    style(ax, "training step", "token-mean entropy (nats, log scale)", logy=True)

    xs, ys = series(rows, "actor/entropy")
    if det:
        ax.axvspan(inv or xs[0], xs[-1], color=PHASE_B, zorder=0)
    elif inv:
        ax.axvspan(inv, xs[-1], color=PHASE_B, zorder=0)
    sy = smooth(xs, ys)
    ax.plot(xs, ys, color=MEANH, lw=0.7, zorder=2, alpha=0.35)
    ax.plot(xs, sy, color=MEANH, lw=2.6, zorder=3, solid_capstyle="round")
    tags = [(sy[-1], "mean H", MEANH)]
    for key, lab in BK:
        bx, by = series(rows, f"actor/entropy_{key}")
        if not bx:
            continue
        sb = smooth(bx, by)
        ax.plot(bx, by, color=BUCKET[key], lw=0.6, zorder=3, alpha=0.30)
        ax.plot(bx, sb, color=BUCKET[key], lw=1.9, zorder=4,
                solid_capstyle="round")
        tags.append((sb[-1], lab, BUCKET[key]))
    if inv:
        marker(ax, inv, f"ordering inverts\n(turn3+ > turn0), step {inv}")
    if det:
        marker(ax, det, f"detonation\nstep {det}", top=0.62)

    # NB do NOT wash out this region -- mean H is valid there; only the
    # per-turn split is missing. Say so in words instead of dimming real data.
    bstart = min((r["step"] for r in rows if "actor/entropy_turn0" in r),
                 default=None)
    if bstart and bstart > xs[0] + 5:
        ax.annotate("mean H only here \u2014 the trainer was not yet\n"
                    "emitting the per-turn split",
                    xy=((xs[0] + bstart) / 2, 0.80),
                    xycoords=("data", "axes fraction"), color=INK2,
                    fontsize=7.5, ha="center", va="top")
    ax.set_xlim(xs[0], xs[-1] + (xs[-1] - xs[0]) * 0.055)
    nat_ticks(ax)
    place_labels(ax, tags)
    phase = ("Phase A (unshaded): turn 0 is the MOST uncertain position — "
             "the policy sharpens as its own\ncontext accumulates. Phase B "
             "(shaded): turn 0 becomes the LEAST uncertain while the late "
             "turns climb.\nMean H rises in both phases and cannot tell them "
             "apart."
             if inv else "The ordering has not inverted.")
    title(fig, f"Two phases of entropy rise — {SHORT[run]}",
          f"{LABEL[run]}.  Thin = per step, thick = smoothed ±10 steps."
          f"\n{phase}")
    fig.subplots_adjust(top=0.765)
    save(fig, f"f1_two_phases_{SHORT[run]}.png")


# ── F2: mean H lies ───────────────────────────────────────────────────────────
def f2_mean_h_lies(run):
    rows = load(run)
    det = detonation(rows)
    fig, axes = figure(9.2, 7.2, nrows=3, sharex=True,
                       gridspec_kw={"height_ratios": [1, 1, 1.1], "hspace": 0.13})
    a, b, c = axes
    style(a, None, "mean H (nats)", logy=True)
    style(b, None, "bucket H (nats)", logy=True)
    style(c, "training step", "H(turn 3+) / H(turn 0)", logy=True)


    xs, ys = series(rows, "actor/entropy")
    a.plot(xs, ys, color=MEANH, lw=1.6, solid_capstyle="round")
    a.plot(xs, smooth(xs, ys), color=INK, lw=2.0, solid_capstyle="round")
    a.annotate("thin = per step,  thick = smoothed ±10 steps",
               xy=(0.04, 0.055), xycoords="axes fraction", color=INK2,
               fontsize=7.5, ha="left")

    btags = []
    for key, lab in (("turn0", "turn 0"), ("turn3plus", "turn 3+")):
        bx, by = series(rows, f"actor/entropy_{key}")
        if not bx:
            continue
        b.plot(bx, smooth(bx, by), color=BUCKET[key], lw=2.0,
               solid_capstyle="round")
        btags.append((smooth(bx, by)[-1], lab, BUCKET[key]))

    place_labels(b, btags)
    rx, rv = ratio(rows)
    rs = smooth(rx, rv)
    c.plot(rx, rv, color="#b9d3f4", lw=1.1, solid_capstyle="round")
    c.plot(rx, rs, color=BUCKET["turn3plus"], lw=2.0, solid_capstyle="round")
    c.axhline(1.0, color=INK2, lw=0.9)
    c.annotate("ratio = 1", xy=(0.012, 1.0),
               xycoords=("axes fraction", "data"), xytext=(0, 6),
               textcoords="offset points", color=INK2, fontsize=7.5)

    # Compare monotonicity on the SAME steps: the bucketed, pre-detonation window.
    hi = det if det else xs[-1] + 1
    hsub = [(x, y) for x, y in zip(xs, ys) if rx[0] <= x < hi]
    rsub = [(x, y) for x, y in zip(rx, rs) if x < hi]
    sm_h_all = smooth(xs, ys)
    mh = monotonicity([v for x, v in zip(xs, sm_h_all) if rx[0] <= x < hi])
    mr = monotonicity([t[1] for t in rsub])
    a.annotate(f"monotonicity {mh:.2f}", xy=(0.012, 0.88),
               xycoords="axes fraction", color=INK, fontsize=9,
               fontweight="bold")
    c.annotate(f"monotonicity {mr:.2f}", xy=(0.012, 0.88),
               xycoords="axes fraction", color=BUCKET["turn3plus"], fontsize=9,
               fontweight="bold")

    # The punchline, on SMOOTHED values: mean H falls back to within a few
    # percent of its early-training level while the ratio has moved several
    # fold. NB an earlier RAW single-step version of this claim ("H at 440 is
    # below H at 220") was an artefact of comparing two noisy points -- on the
    # smoothed series the dip lands slightly ABOVE the early level. The claim
    # is "returns to near-baseline", never "below baseline".
    if det:
        early = [v for x, v in zip(xs, sm_h_all) if rx[0] <= x <= rx[0] + 20]
        early = sum(early) / len(early)
        dv, dx = min((v, x) for x, v in zip(xs, sm_h_all) if det - 60 <= x < det)
        dr = dict(zip(rx, rs)).get(dx)
        r0 = [v for x, v in zip(rx, rs) if rx[0] <= x <= rx[0] + 20]
        r0 = sum(r0) / len(r0)
        for ax_ in axes:
            marker(ax_, dx, "", color="#e34948")
        a.annotate(f"step {dx}: mean H {dv:.2f}, vs {early:.2f} at step "
                   f"{int(rx[0])}\n\u2014 within "
                   f"{abs(dv / early - 1) * 100:.0f}% of baseline, yet only\n"
                   f"{det - dx} steps from detonation",
                   xy=(0.04, 0.78), xycoords="axes fraction",
                   color="#e34948", fontsize=8, ha="left", va="bottom",
                   fontweight="bold")
        if dr:
            c.annotate(f"\u2026while the ratio went {r0:.1f}\u00d7 "
                       f"\u2192 {dr:.1f}\u00d7 over that same span",
                       xy=(0.04, 0.78), xycoords="axes fraction",
                       color="#e34948", fontsize=8, ha="left", va="bottom",
                       fontweight="bold")
        for ax_ in axes:
            marker(ax_, det, "")
        c.annotate(f"detonation {det}", xy=(det, 0.055),
                   xycoords=("data", "axes fraction"), xytext=(-4, 0),
                   textcoords="offset points", color=INK2, fontsize=7.5,
                   ha="right", va="bottom")
    else:
        last = rx[-1]
        for ax_ in axes:
            marker(ax_, last, "")
        c.annotate(f"still running\n(step {last}, ratio {rs[-1]:.2f}×)",
                   xy=(last, rs[-1]), xytext=(-8, 6), textcoords="offset points",
                   color=INK2, fontsize=8, ha="right", fontweight="bold")
    c.set_xlim(rx[0] - 4, rx[-1] + (rx[-1] - rx[0]) * 0.055)
    for ax_ in axes:
        nat_ticks(ax_)
    title(fig, f"Mean H is not the state variable — {SHORT[run]}",
          f"{LABEL[run]}\n" + ("Mean H oscillates and comes back to near its "
           "early-training level; the turn0/turn3+ split and their ratio do "
           "not." if det else "Mean H has stayed flat for the whole run, "
           "while turn 0 and turn 3+ separate steadily underneath it."))
    fig.subplots_adjust(top=0.875)
    save(fig, f"f2_mean_h_lies_{SHORT[run]}.png")


# ── F3: drift is dead (the H2 refutation) ─────────────────────────────────────
def f3_drift_sign(run):
    """Cov over training time, overall and per solver turn.

    This replaces an earlier version that scattered dH against Cov. That plot
    had no power: dH[t] = H[t+1] - H[t] is measured on a FRESH draw of 120
    tasks x 4 samples, so its step-to-step noise (~+-0.05) is an order of
    magnitude above the mean dH it was meant to explain (~+0.006). The only
    thing the data supports is the SIGN of Cov, which needs no dH axis at all.

    Plots PRE-DETONATION steps only. After detonation the advantage collapses
    with the reward (zero_frac -> 1.00), so Cov decays to ~0 -- but it first
    SPIKES (0.15-0.23) at the detonation step itself, and that spike is 3-5x
    the whole pre-collapse range, so including it flattens the signal this
    figure is about. The numbers are in the caption instead.
    """
    rows = load(run)
    det = detonation(rows)
    keep = [r for r in rows if det is None or r["step"] < det]
    fig, (a, b) = figure(9.6, 6.9, nrows=2, sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.12})
    style(a, None, "Cov(A, log π)\nall solver tokens")
    style(b, "training step", "Cov(A, log π)\nby solver turn")
    for ax in (a, b):
        ax.axhline(0.0, color=INK, lw=1.0, zorder=5)

    def stat(key):
        xs = [r["step"] for r in keep if key in r]
        ys = [r[key] for r in keep if key in r]
        if not xs:
            return None
        return xs, ys, sum(1 for y in ys if y > 0) / len(ys), sum(ys) / len(ys)

    xs, ys, fpos, mean_cov = stat("actor/adv_logp_cov")
    a.plot(xs, ys, color=BUCKET["turn1"], lw=0.7, alpha=0.35, zorder=3)
    a.plot(xs, smooth(xs, ys), color=BUCKET["turn1"], lw=2.3, zorder=4,
           solid_capstyle="round")
    a.annotate(f"Cov > 0 on {fpos:.0%} of the {len(xs)} pre-detonation steps "
               f"(mean {mean_cov:+.4f})\n→ the policy gradient is pushing "
               f"entropy DOWN essentially throughout",
               xy=(0.015, 0.94), xycoords="axes fraction", color=INK,
               fontsize=9, va="top", fontweight="bold")

    tags, summary = [], []
    for key, lab in BK:
        got = stat(f"actor/adv_logp_cov_{key}")
        if not got:
            continue
        bx, by, fp, mc = got
        sb = smooth(bx, by)
        b.plot(bx, by, color=BUCKET[key], lw=0.5, alpha=0.20, zorder=3)
        b.plot(bx, sb, color=BUCKET[key], lw=2.0, zorder=4,
               solid_capstyle="round", label=lab)
        tags.append((sb[-1], f"{lab}   {fp:.0%} > 0", BUCKET[key]))
        summary.append((lab, fp, mc))
    place_labels(b, tags)
    leg = b.legend(frameon=False, fontsize=8, loc="upper left", ncol=4)
    for t in leg.get_texts():
        t.set_color(INK2)
    b.annotate("\n".join(f"{lab:>7}:  {fp:3.0%} of steps > 0,  mean {mc:+.5f}"
                         for lab, fp, mc in summary),
               xy=(0.015, 0.80), xycoords="axes fraction", color=INK,
               fontsize=8, va="top", family="monospace")

    if det:
        for ax in (a, b):
            ax.axvline(det, color=INK2, lw=0.9, alpha=0.75)
        b.annotate(f"detonation {det}", xy=(det, 0.02),
                   xycoords=("data", "axes fraction"), xytext=(-5, 0),
                   textcoords="offset points", color=INK2, fontsize=7.5,
                   ha="right", va="bottom")
    b.set_xlim(xs[0], xs[-1] + (xs[-1] - xs[0]) * 0.11)
    tail = ("At the detonation step itself Cov SPIKES, then decays toward 0 as "
            "the reward\nvariance vanishes; both are excluded here (see the "
            "docstring)." if det else
            "This run has not detonated, so every step is shown.")
    title(fig, f"Drift sharpens where it has signal — and it has none at turn 0 "
               f"— {SHORT[run]}",
          f"{LABEL[run]}.  Pre-detonation steps only. Thin = per step, thick = "
          f"smoothed ±10 steps.\nUnder Ḣ = −Cov(A, log π), "
          f"positive Cov means the gradient is SHARPENING. Cov is strongest "
          f"at turn 3+ \u2014 the bucket whose\nentropy explodes \u2014 and "
          f"decays to a coin flip at turn 0, the bucket whose entropy stays "
          f"flat. So drift is not driving\nthe rise: it opposes it, and does "
          f"so most strongly exactly where the rise is worst.\n{tail}")
    fig.subplots_adjust(top=0.715)
    save(fig, f"f3_drift_sign_{SHORT[run]}.png")


# ── F4: CROSS-RUN, approach to detonation ─────────────────────────────────────
def f4_time_to_detonation():
    fig, (a, b) = figure(9.6, 6.6, nrows=2, sharex=True,
                         gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.12})
    style(a, None, "H(turn 3+) / H(turn 0)", logy=True)
    style(b, "steps relative to each run's own detonation", "val reward (mean@1)")
    a.axhline(1.0, color=INK2, lw=0.9)
    a.annotate("ratio = 1", xy=(0.012, 1.0), xycoords=("axes fraction", "data"),
               xytext=(0, 6), textcoords="offset points", color=INK2,
               fontsize=7.5)
    for ax in (a, b):
        ax.axvline(0, color=INK, lw=1.1)

    live, atags, peaks = [], [], []
    for run in RUNS:
        rows = load(run)
        det = detonation(rows)
        rx, rv = ratio(rows)
        sm = smooth(rx, rv)
        if det is None:
            live.append((run, rx[-1], sm[-1]))
            continue
        keep = [(x - det, y) for x, y in zip(rx, sm) if x - det <= 6]
        a.plot([t[0] for t in keep], [t[1] for t in keep], color=RUNC[run],
               lw=2.0, solid_capstyle="round", label=SHORT[run])
        atags.append((keep[-1][1], f"{SHORT[run]}  {keep[-1][1]:.0f}x", RUNC[run]))
        vx, vy = series(rows, VK)
        vk = [(x - det, y) for x, y in zip(vx, vy) if x - det <= 6]
        b.plot([t[0] for t in vk], [t[1] for t in vk], color=RUNC[run], lw=2.0,
               marker="o", ms=4.2, mec=SURFACE, mew=1.0, solid_capstyle="round")
        pk = max(vk, key=lambda t: t[1])
        b.plot([pk[0]], [pk[1]], marker="o", ms=9, color=SURFACE,
               mec=RUNC[run], mew=1.8, zorder=6)
        peaks.append((pk, run))

    place_labels(a, atags)
    a.annotate("detonation", xy=(0, 0.985), xycoords=("data", "axes fraction"),
               xytext=(-5, 0), textcoords="offset points", color=INK,
               fontsize=8, ha="right", va="top", fontweight="bold")
    # Three peaks sit within ~50 steps of each other, so labelling AT the
    # markers collides however it is offset. The rings mark them; the values go
    # in a stacked block, one line per run in its own colour.
    lo, hi = b.get_ylim()
    b.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.06)
    for i, (pk, run) in enumerate(sorted(peaks, key=lambda t: -t[0][1])):
        b.annotate(f"{SHORT[run]}:  val peak {pk[1]:.3f} at {pk[0]:.0f} steps",
                   xy=(0.015, 0.30 - 0.085 * i), xycoords="axes fraction",
                   color=RUNC[run], fontsize=8.5, fontweight="bold")

    # A still-running run has no detonation step, so it is drawn as a LEVEL.
    # Placing it anywhere on this x-axis would assert a position we do not know.
    for run, last, now in live:
        a.axhline(now, color=RUNC[run], lw=1.5)
        a.annotate(f"{SHORT[run]}: {now:.1f}x at step {last}, still running",
                   xy=(0.012, now), xycoords=("axes fraction", "data"),
                   xytext=(0, -14), textcoords="offset points",
                   color=RUNC[run], fontsize=8, fontweight="bold")
        a.plot([], [], color=RUNC[run], lw=2.0,
               label=f"{SHORT[run]} (running, level only)")

    hands, labs_ = a.get_legend_handles_labels()
    leg = fig.legend(hands, labs_, frameon=False, fontsize=8.5, ncol=3,
                     loc="upper left", bbox_to_anchor=(0.075, 0.845))
    for t in leg.get_texts():
        t.set_color(INK2)
    nat_ticks(a)
    for ax in (a, b):
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 30)
    title(fig, "The approach to detonation reproduces across runs",
          "Aligned on each run's own detonation step (mean H > 2 sustained); "
          "ratio smoothed ±10 steps.\nAll three land at 20–26x although "
          "they detonate at steps 464, 747 and 1015; val reward peaks first, "
          "60–120 steps out.")
    fig.subplots_adjust(top=0.755, right=0.845)
    save(fig, "f4_time_to_detonation.png")


# ── F6: CROSS-RUN, length is epiphenomenal ────────────────────────────────────
def f6_length_epiphenomenal():
    metrics = [("actor/solver_turn_len/mean", "tokens per solver turn"),
               ("actor/entropy_turns/mean", "solver turns per episode")]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.0), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.33, "wspace": 0.18})
    for r, (key, ylab) in enumerate(metrics):
        pool = []
        for run in RUNS:
            rows = load(run)
            det = detonation(rows)
            xs, ys = series(rows, key)
            pool += [y for x, y in zip(xs, ys) if det is None or x < det]
        ymin, ymax = min(pool), max(pool)
        pad = (ymax - ymin) * 0.12
        for c, run in enumerate(RUNS):
            ax = axes[r][c]
            rows = load(run)
            det = detonation(rows)
            xs, ys = series(rows, key)
            keep = [(x, y) for x, y in zip(xs, ys) if det is None or x < det]
            style(ax, "training step" if r == 1 else None,
                  ylab if c == 0 else None)
            ax.plot([t[0] for t in keep], [t[1] for t in keep],
                    color="#cfd4d8", lw=1.0)
            sx = [t[0] for t in keep]
            ax.plot(sx, smooth(sx, [t[1] for t in keep]), color=RUNC[run],
                    lw=2.0, solid_capstyle="round")
            ax.set_ylim(ymin - pad * 2.6, ymax + pad)
            if c:
                ax.tick_params(labelleft=False)
            hx, hy = series(rows, "actor/entropy")
            hmap = dict(zip(hx, hy))
            both = [(y, hmap[x]) for x, y in keep if x in hmap]
            rho = pearson([t[0] for t in both], [t[1] for t in both])
            b, _ = ols([t[0] for t in keep], [t[1] for t in keep])
            up = b > 0
            # Sign is the whole point, so it takes the diverging poles:
            # cool = negative, warm = positive.
            pole = "#a52020" if rho > 0 else "#1c5cab"
            ax.annotate(f"trend {'↑' if up else '↓'} "
                        f"{b * 100:+.2f} / 100 steps",
                        xy=(0.5, 0.115), xycoords="axes fraction", color=INK2,
                        fontsize=8.5, ha="center", va="bottom")
            ax.annotate(f"corr with mean H = {rho:+.3f}",
                        xy=(0.5, 0.032), xycoords="axes fraction", color=pole,
                        fontsize=9.5, ha="center", va="bottom",
                        fontweight="bold")
            if r == 0:
                ax.set_title(SHORT[run], color=RUNC[run], fontsize=10,
                             fontweight="bold", pad=8)
    title(fig, "Length and turn count are epiphenomenal — they flip sign between runs",
          "Pre-detonation steps only; thick line smoothed ±10 steps, shared "
          "y-scale per row. Correlation is coloured by SIGN.\nAll three runs "
          "collapse the same way, so neither variable can be the mechanism.")
    fig.subplots_adjust(top=0.80)
    save(fig, "f6_length_epiphenomenal.png")


def main():
    print("per-run figures:")
    for run in RUNS:
        f1_two_phases(run)
        f2_mean_h_lies(run)
        f3_drift_sign(run)
    print("cross-run figures:")
    f4_time_to_detonation()
    f6_length_epiphenomenal()


if __name__ == "__main__":
    main()
