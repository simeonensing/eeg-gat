#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-ready, hard-coded plotting script (single-column figures).

Inputs (fixed): results/
Models (fixed): gwt_gat (graph-based) and classical
Outputs (PDF only): results/figures/*.pdf

Design goals (journal single-column ~85 mm wide):
- Figure width ~3.35 in (85 mm). Titles concise; second line shows model in brackets.
- Font sizes tuned for print: title 9 pt, labels 8 pt, ticks 7 pt, legend 7 pt.
- ROC/PR: show per-fold curves with compact legend (Fold n) + an average curve.
- Calibration: combined plot with two traces; ECE/Brier moved below x-axis label.
- Pred. probability histogram: classes by name (Sham, Active).
- Confusion matrices: aggregate + per-fold grids; concise titles.
- Topomaps: real scalp maps via MNE; narrow colorbar but readable ticks/label; no clipping.

Notes:
- Uses only ASCII hyphens to avoid missing-glyph warnings in some serif fonts.
- Saves vector PDFs (editable text) without tight bbox. Margins are tuned to avoid cropping.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re, textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec

import mne
from mne.viz import plot_topomap
from mne import create_info




# ---------------- Global style (single-column friendly) ----------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Base sizes (pts)
    "font.size": 8,                # default
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

SAVE_DIR = Path("../results")
FIG_DIR = SAVE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PREFIXES = ["gwt_gat", "classical"]
MODEL_LABEL = {"gwt_gat": "Graph-based", "classical": "Classical"}
MONTAGE_NAME = "standard_1020"

# Single-column sizes (inches)
COL_W = 3.35   # ~85 mm
H_ROC_PR = 2.40
H_CALIB = 2.40
H_HIST = 2.20
H_CM = 2.40
TOPOMAP_SIZE = (COL_W, 2.90)

# PDF saving (vector). dpi used only if any raster elements are present.
PDF_DPI = 600

def save_pdf(out_base: Path):
    path = out_base.with_suffix(".pdf")
    plt.savefig(path, dpi=PDF_DPI, bbox_inches=None, metadata={"Creator": "Matplotlib"})
    print("[Saved]", path)

# ---------------- Utility helpers ----------------

def _short_title(main: str, prefix: str) -> str:
    model = MODEL_LABEL.get(prefix, prefix)
    return f"{main}\n({model})"

# ---- ROC ----

def plot_roc(prefix: str):
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc as sk_auc  # use sklearn's auc like the original

    paths = sorted(SAVE_DIR.glob(f"roc_{prefix}_fold*.csv"))
    if not paths:
        print(f"[ROC] No CSVs for {prefix}")
        return

    plt.figure(figsize=(COL_W, H_ROC_PR))

    fpr_grid = np.linspace(0.0, 1.0, 200)
    tprs_on_grid = []
    aucs = []

    # per-fold curves
    for i, p in enumerate(paths, 1):
        df = pd.read_csv(p)
        if {"fpr", "tpr"} <= set(df.columns):
            fpr = np.asarray(df["fpr"], float)
            tpr = np.asarray(df["tpr"], float)

            # label "Fold n"
            nums = re.findall(r"\d+", p.stem)
            fold_lab = f"Fold {nums[-1]}" if nums else f"Fold {i}"

            # plot raw fold
            plt.plot(fpr, tpr, lw=0.9, alpha=0.9, label=fold_lab)

            # interpolate this fold onto a common FPR grid (original behavior)
            tpr_i = np.interp(fpr_grid, fpr, tpr)
            tpr_i[0] = 0.0
            tpr_i[-1] = 1.0
            tprs_on_grid.append(tpr_i)

            # fold AUC (like original)
            try:
                aucs.append(float(sk_auc(fpr, tpr)))
            except Exception:
                aucs.append(float(np.trapz(tpr, fpr)))

    # mean ROC across folds (original averaging on a dense grid)
    if tprs_on_grid:
        mean_tpr = np.mean(np.vstack(tprs_on_grid), axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        sd_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        plt.plot(
            fpr_grid, mean_tpr, lw=2.2, zorder=5,
            label=f"Mean ROC (AUC={mean_auc:.3f}±{sd_auc:.3f})"
        )

    # chance line and cosmetics
    plt.plot([0, 1], [0, 1], "--", lw=0.8, alpha=0.6, color="tab:pink")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(_short_title("ROC — Trade-off between TPR and FPR", prefix), pad=6)
    plt.legend(loc="lower right", frameon=False, ncol=1, title=None)
    plt.tight_layout()
    save_pdf(FIG_DIR / f"roc_{prefix}_outer_folds")
    plt.close()



# ---- PR ----

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pr(prefix: str, show_grid: bool = False):
    """
    Plot per-fold PR curves for `prefix`, add the macro curve, and read
    Macro PR + Mean AP from results/results_summary.csv for the legend.

    - If `prefix` refers to the graph pipeline (e.g., 'gwt_gat'), we read Graph_* columns.
    - If it refers to the classical pipeline, we read Classical_* columns.
    - Legend is rendered in a dedicated panel below the plot (max 2 columns).
    """
    def step_ap_from_curve(rec, pre, pos_rate=None):
        # (kept for fallback if results_summary.csv is missing)
        rec = np.asarray(rec, float)
        pre = np.asarray(pre, float)
        if rec.size == 0:
            return float("nan")
        if rec[0] > 0.0:  # prepend (0,1)
            rec = np.r_[0.0, rec]
            pre = np.r_[1.0, pre]
        if pos_rate is not None and rec[-1] < 1.0:  # append (1, pos_rate)
            rec = np.r_[rec, 1.0]
            pre = np.r_[pre, float(pos_rate)]
        pre = np.maximum.accumulate(pre[::-1])[::-1]  # precision envelope
        drec = np.diff(rec)
        return float(np.sum(pre[1:] * drec))

    # Files
    paths = sorted(SAVE_DIR.glob(f"pr_{prefix}_fold*.csv"))
    macro_path = SAVE_DIR / f"pr_macro_{prefix}.csv"
    summary_csv = SAVE_DIR / "results_summary.csv"

    if not paths and not macro_path.exists():
        print(f"[PR] No CSVs for {prefix}")
        return

    # Decide which column set to read from results_summary.csv
    # Heuristic: classical if 'class' in prefix, else graph.
    use_group = "Classical" if re.search(r"class", prefix, flags=re.I) else "Graph"
    col_mean = f"{use_group}_Mean"

    # Try to read Macro PR and Mean AP from results_summary.csv
    macro_from_summary = np.nan
    mean_ap_from_summary = np.nan
    if summary_csv.exists():
        try:
            df_sum = pd.read_csv(summary_csv)
            # 'Macro PR' row
            mpr = df_sum.loc[df_sum["Metric"].str.strip().str.lower() == "macro pr"]
            if not mpr.empty and col_mean in mpr.columns:
                macro_from_summary = float(mpr.iloc[0][col_mean])
            # 'AP' row
            ap = df_sum.loc[df_sum["Metric"].str.strip().str.lower() == "ap"]
            if not ap.empty and col_mean in ap.columns:
                mean_ap_from_summary = float(ap.iloc[0][col_mean])
        except Exception as e:
            print(f"[PR] Could not read results_summary.csv: {e}")

    # First pass: gather per-fold curves
    per_fold_curves = []
    # Fallback AP computation (if summary missing); we can ignore if you never want it
    ap_vals_fallback = []

    for p in paths:
        df = pd.read_csv(p)
        if {"recall", "precision"} <= set(df.columns):
            nums = re.findall(r"\d+", p.stem)
            fold_lab = f"Fold {nums[-1]}" if nums else "Fold"
            rec = df["recall"].to_numpy(float)
            pre = df["precision"].to_numpy(float)
            per_fold_curves.append((rec, pre, fold_lab))

            # Fallback AP (not used if summary found)
            pos_rate = float(df["pos_rate"].iloc[0]) if "pos_rate" in df.columns else None
            ap_i = step_ap_from_curve(rec, pre, pos_rate)
            if np.isfinite(ap_i):
                ap_vals_fallback.append(ap_i)

    # Macro curve for this prefix
    macro_curve = None
    if macro_path.exists():
        dm = pd.read_csv(macro_path)
        if {"recall", "precision"} <= set(dm.columns):
            rec_m = dm["recall"].to_numpy(float)
            pre_m = dm["precision"].to_numpy(float)
            macro_curve = (rec_m, pre_m)

    # Build the Average label using summary if available, else fallback
    macro_area = macro_from_summary
    mean_ap = mean_ap_from_summary
    if not np.isfinite(macro_area) and macro_curve is not None:
        # Fallback area for the macro curve if summary missing
        macro_area = float(np.trapezoid(macro_curve[1], macro_curve[0])) if macro_curve[0].size else float("nan")
    if not np.isfinite(mean_ap):
        mean_ap = float(np.nanmean(ap_vals_fallback)) if ap_vals_fallback else float("nan")

    label_avg = None
    if macro_curve is not None:
        if np.isfinite(macro_area) and np.isfinite(mean_ap):
            label_avg = f"Average (Macro PR={macro_area:.3f}; Mean AP={mean_ap:.3f})"
        elif np.isfinite(macro_area):
            label_avg = f"Average (Macro PR={macro_area:.3f})"
        elif np.isfinite(mean_ap):
            label_avg = f"Average (Mean AP={mean_ap:.3f})"
        else:
            label_avg = "Average"

    # --- Dynamic figure height based on legend rows, with max 2 columns ---
    legend_labels = [lab for *_, lab in per_fold_curves]
    if label_avg:
        legend_labels.append(label_avg)

    n_items = len(legend_labels)
    ncol = 1 if n_items <= 1 else 2
    rows = int(np.ceil(n_items / max(1, ncol))) if n_items else 1

    base_h = H_ROC_PR
    per_row_h = 0.35
    fig_h = base_h + per_row_h * rows

    fig = plt.figure(figsize=(COL_W, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, (per_row_h * rows) / fig_h])
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")

    # Plot per-fold curves
    for rec, pre, lab in per_fold_curves:
        ax.step(rec, pre, where="post", lw=0.9, alpha=0.9, label=lab)

    # Macro curve (brown)
    if macro_curve is not None and label_avg is not None:
        ax.plot(macro_curve[0], macro_curve[1], lw=1.6, color="tab:brown", label=label_avg, zorder=5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(_short_title("Precision–Recall tradeoff — outer folds", prefix), pad=6)

    if show_grid:
        ax.grid(True, which="major", alpha=0.35)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15)

    # Legend below plot
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
                     handlelength=2.2, handletextpad=0.6, columnspacing=1.0)

    fig.tight_layout()
    save_pdf(FIG_DIR / f"pr_{prefix}_outer_folds")
    plt.close(fig)




def plot_calibration_combined():
    """
    Combined calibration plot with one ECE/Brier line per model placed
    strictly *below* the x-label. We don't move the x-label. Instead we
    add figure height if needed, then render the lines in figure coords.
    """

    fig, ax = plt.subplots(figsize=(COL_W, H_CALIB))
    # ax.minorticks_on()  # optional: minor ticks too
    # ax.grid(True, which='major', alpha=0.35)  # major grid
    # ax.grid(True, which='minor', alpha=0.15)  # lighter minor grid
    # ax.set_axisbelow(True)  # keep grid behind dataset

    # Plot curves + collect per-model lines
    lines_below = []
    plotted_any = False
    for prefix in PREFIXES:
        curve = SAVE_DIR / f"calibration_curve_{prefix}.csv"
        stats = SAVE_DIR / f"calibration_stats_{prefix}.csv"
        label = MODEL_LABEL.get(prefix, prefix)

        if curve.exists():
            dfc = pd.read_csv(curve)
            if {"bin_mean_pred", "frac_positives"} <= set(dfc.columns):
                ax.plot(
                    dfc["bin_mean_pred"], dfc["frac_positives"],
                    "-o", ms=2.8, lw=1.0, label=label
                )
                plotted_any = True

        if stats.exists():
            dst = pd.read_csv(stats)
            if not dst.empty and {"ece", "brier"} <= set(dst.columns):
                ece   = float(dst["ece"].iloc[0])
                brier = float(dst["brier"].iloc[0])
                lines_below.append(f"{label}: ECE={ece:.3f} · Brier={brier:.3f}")

    # Axes cosmetics
    ax.plot([0, 1], [0, 1], "--", lw=0.8, color="gray", label="Perfect")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")  # do NOT move this
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability", pad=6)
    ax.legend(loc="upper left", frameon=False, ncol=1, title=None)

    # If nothing plotted, still save a scaffold
    if not plotted_any:
        out = FIG_DIR / "calibration_combined.pdf"
        fig.savefig(out)
        plt.close(fig)
        return

    extra_artists = []
    if lines_below:
        # Render once to measure positions in pixels
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Pixel geometry
        dpi = fig.dpi
        fig_w_in, fig_h_in = fig.get_size_inches()
        fig_h_px = fig_h_in * dpi

        xl = ax.xaxis.get_label()
        xl_bb = xl.get_window_extent(renderer=renderer)  # in pixels
        ax_bb = ax.get_window_extent(renderer=renderer)  # in pixels
        left_px = ax_bb.x0  # left align to axes left

        # Estimate text line height for chosen fontsize
        fs = 7
        probe = fig.text(0, 0, "Hg", fontsize=fs)
        fig.canvas.draw()
        line_h_px = probe.get_window_extent(renderer=renderer).height
        probe.remove()

        # Desired space below label
        pad_between = 4   # px between x-label and first stats line
        pad_bottom  = 4   # px at very bottom
        needed_px   = pad_between + len(lines_below) * line_h_px + pad_bottom

        # Current room (bottom of label to figure bottom)
        current_room_px = max(xl_bb.y0, 0.0)

        # If not enough, increase figure height (adds space below without moving the label)
        shortfall_px = max(0.0, needed_px - current_room_px)
        if shortfall_px > 0:
            fig.set_size_inches(fig_w_in, fig_h_in + shortfall_px / dpi, forward=True)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            # refresh metrics
            fig_w_in, fig_h_in = fig.get_size_inches()
            fig_h_px = fig_h_in * dpi
            xl_bb = xl.get_window_extent(renderer=renderer)
            ax_bb = ax.get_window_extent(renderer=renderer)
            left_px = ax_bb.x0

        # Now place lines strictly below the x-label (y decreases towards bottom)
        start_y_px = xl_bb.y0 - pad_between - line_h_px  # first line baseline
        for i, text in enumerate(lines_below):
            y_px  = start_y_px - i * line_h_px
            y_fig = y_px / fig_h_px
            x_fig = left_px / (fig_w_in * dpi)
            t = fig.text(x_fig, y_fig, text, fontsize=fs, ha="left", va="bottom")
            extra_artists.append(t)

    # Save ensuring extra artists are included in the bbox (prevents cropping)
    out = FIG_DIR / "calibration_combined.pdf"
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=extra_artists, pad_inches=0.01)
    plt.close(fig)


# ---- Predicted probability histogram ----

def plot_pred_prob_hist(prefix: str):
    probs = SAVE_DIR / f"probabilities_{prefix}.csv"
    if not probs.exists():
        print(f"[ProbHist] No probabilities CSV for {prefix}")
        return
    dfp = pd.read_csv(probs)
    if not {"y_true","prob"} <= set(dfp.columns):
        print(f"[ProbHist] Missing columns in {probs.name}")
        return
    fig, ax = plt.subplots(figsize=(COL_W, H_HIST))
    # Map to class names
    sham = dfp.loc[dfp["y_true"]==0, "prob"].values
    active = dfp.loc[dfp["y_true"]==1, "prob"].values
    ax.hist(sham, bins=20, alpha=0.55, label="Sham", density=True)
    ax.hist(active, bins=20, alpha=0.55, label="Active", density=True)
    ax.set_xlabel("Predicted probability of Active")
    ax.set_ylabel("Density")
    ax.set_title(_short_title("Predicted probability distribution", prefix), pad=6)
    ax.legend(loc="upper center", frameon=False, ncol=2)
    plt.tight_layout()
    save_pdf(FIG_DIR / f"pred_prob_hist_{prefix}")
    plt.close(fig)

# ---- Confusion matrices ----

def plot_confusion(prefix: str):
    pth = SAVE_DIR / f"predictions_{prefix}.csv"
    if not pth.exists():
        print(f"[Confusion] predictions_{prefix}.csv not found")
        return
    df = pd.read_csv(pth)
    if not {"y_true","prob","fold"} <= set(df.columns):
        print(f"[Confusion] Missing columns in {pth.name}")
        return
    df["y_pred"] = (df["prob"] >= 0.5).astype(int)

    # ---------- Aggregate ----------
    y = df["y_true"].values
    yhat = df["y_pred"].values
    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(y, yhat):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(COL_W, H_CM))
    im = ax.imshow(cm, cmap="Blues", vmin=0, aspect="equal")
    ax.set_title(_short_title("Confusion matrix — aggregation of outer folds", prefix), pad=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Sham","Active"]); ax.set_yticklabels(["Sham","Active"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                    color="white" if cm[r,c] > cm.max()/2 else "black", fontsize=8)
    fig.tight_layout()
    save_pdf(FIG_DIR / f"confusion_matrix_{prefix}")
    plt.close(fig)

    # ---------- Per-fold grid ----------
    folds = sorted(df["fold"].unique())
    if len(folds) == 0:
        return

    cms, vmax = [], 0
    for f in folds:
        dff = df[df["fold"] == f]
        cmf = np.zeros((2,2), dtype=int)
        for t, p in zip(dff["y_true"].values, dff["y_pred"].values):
            cmf[t, p] += 1
        cms.append((f, cmf))
        vmax = max(vmax, cmf.max())

    n = len(folds)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    # Wider figure as columns increase; keep squares
    w_per, h_per = 2.2, 2.2
    fig_w = max(COL_W, w_per * ncols)
    fig_h = max(H_CM, h_per * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    for i, (f, cmf) in enumerate(cms):
        ax = axes[i]
        ax.imshow(cmf, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

        ax.set_title(f"Fold {int(f)}", pad=4)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])

        # Outer-edge labels to reduce clutter
        if i // ncols == nrows - 1:
            ax.set_xticklabels(["Sham","Active"])
            ax.set_xlabel("Predicted")
        else:
            ax.set_xticklabels([])
        if i % ncols == 0:
            ax.set_yticklabels(["Sham","Active"])
            ax.set_ylabel("True")
        else:
            ax.set_yticklabels([])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cmf[r, c]), ha="center", va="center",
                        color="white" if cmf[r, c] > vmax/2 else "black", fontsize=7)

        # Thin gridlines
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Clear gap for title: reserve top 7% of the figure for the suptitle
    fig.suptitle(_short_title("Confusion matrices by fold", prefix), y=0.99, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])  # <-- this creates the visible gap

    save_pdf(FIG_DIR / f"confusion_matrices_{prefix}")
    plt.close(fig)



# ---------------- Topomap helpers ----------------
SUPTITLE_FONTSIZE = 9
SUPTITLE_Y = 0.95
TITLE_WRAP_CHARS = 44
CBAR_LABEL_FONTSIZE = 8
CBAR_TICK_FONTSIZE = 7

DEFAULT_CMAP_SEQ_POS = "plasma"
DEFAULT_CMAP_SEQ_NEG = "plasma_r"
DEFAULT_CMAP_DIV     = "RdBu_r"

# Layout tuned to avoid cbar/tick clipping while keeping cbar narrow.
SUBPLOT_ADJUST = dict(left=0.06, right=0.84, top=0.86, bottom=0.12, wspace=0.10)
GSPEC_WIDTH_RATIOS = [30, 2.0]   # topo : colorbar (narrow)
RES = 512
CONTOURS = 0


def sentence_case(text: str) -> str:
    if not text:
        return text
    s = text.strip().lower()
    s = s[0:1].upper() + s[1:]
    for ac in ["EEG","ROC","PR","AUC","GAT","GWT"]:
        s = re.sub(rf"\\b{ac.lower()}\\b", ac, s)
    return s


def wrap_multiline_title(s: str, width: int = TITLE_WRAP_CHARS) -> str:
    # keep concise and wrapped
    lines = s.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(wrapped)


def choose_cmap_and_limits(vals: np.ndarray):
    vals = np.asarray(vals, float)
    vmin = np.nanmin(vals); vmax = np.nanmax(vals)
    if (vmin < 0) and (vmax > 0):
        vmax_abs = float(np.nanpercentile(np.abs(vals), 99))
        if not np.isfinite(vmax_abs) or vmax_abs <= 0:
            vmax_abs = max(abs(vmin), abs(vmax), 1.0)
        return DEFAULT_CMAP_DIV, -vmax_abs, vmax_abs
    if vmin >= 0:
        high = float(np.nanpercentile(vals, 99))
        if not np.isfinite(high) or high <= 0:
            high = max(vmax, 1.0)
        return DEFAULT_CMAP_SEQ_POS, 0.0, high
    low = float(np.nanpercentile(vals, 1))
    if not np.isfinite(low):
        low = vmin
    return DEFAULT_CMAP_SEQ_NEG, low, 0.0


def _compute_sphere(info: mne.Info, ch_names: list[str]) -> tuple[float,float,float,float]:
    mont = info.get_montage()
    posd = mont.get_positions()["ch_pos"] if mont is not None else {}
    pos3d = np.array([posd[ch] for ch in ch_names if ch in posd and posd[ch] is not None])
    if pos3d.size == 0:
        r = 0.095
    else:
        r = float(np.max(np.linalg.norm(pos3d[:, :2], axis=1)))
        if not np.isfinite(r) or r <= 0:
            r = 0.095
    r *= 1.002
    return (0.0, 0.0, 0.0, r)


def _build_info_from_channels(ch_names, montage_name: str) -> mne.Info:
    alias = {"AF3": "Fp1", "AF4": "Fp2", "Fp1": "AF3", "Fp2": "AF4"}
    std = mne.channels.make_standard_montage(montage_name)
    pos = std.get_positions()["ch_pos"]
    ch_pos, missing = {}, []
    for ch in ch_names:
        if ch in pos and pos[ch] is not None:
            ch_pos[ch] = np.asarray(pos[ch][:3], float)
        elif ch in alias and alias[ch] in pos and pos[alias[ch]] is not None:
            ch_pos[ch] = np.asarray(pos[alias[ch]][:3], float)
        else:
            missing.append(ch)
    if missing:
        print(f"[Topomap] WARNING: No 3D position for channels: {missing}")
    dig = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = create_info(ch_names=list(ch_pos.keys()), sfreq=100.0, ch_types=["eeg"] * len(ch_pos))
    info.set_montage(dig, on_missing="ignore")
    return info


def _plot_topomap(values_vec, info, title, outbase, cbar_label, vmin, vmax, cmap):
    fig = plt.figure(figsize=TOPOMAP_SIZE, dpi=300)
    gs = GridSpec(1, 2, width_ratios=GSPEC_WIDTH_RATIOS, figure=fig)
    ax_topo, ax_cbar = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    # Two-line concise title already wrapped
    fig.suptitle(title, y=SUPTITLE_Y, fontsize=SUPTITLE_FONTSIZE, ha="center")

    sphere = _compute_sphere(info, info["ch_names"])
    try:
        im, _ = plot_topomap(values_vec, info, axes=ax_topo, outlines='head', cmap=cmap,
                             vmin=vmin, vmax=vmax, extrapolate='head', image_interp='cubic',
                             res=RES, contours=CONTOURS, sphere=sphere, show=False)
    except TypeError:
        im, _ = plot_topomap(values_vec, info, axes=ax_topo, outlines='head', cmap=cmap,
                             vlim=(vmin, vmax), extrapolate='head', image_interp='cubic',
                             res=RES, contours=CONTOURS, sphere=sphere, show=False)
    # Slim colorbar with readable ticks/label (no clipping)
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=CBAR_LABEL_FONTSIZE, labelpad=6)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE, pad=2)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # Save
    save_pdf(outbase)
    plt.close(fig)


def make_topomap_title(stem: str, method_name: str | None, prefix: str) -> str:
    # Concise informative title + second line model label in brackets
    if method_name is None or 'consensus' in (stem or '').lower():
        main = "Overall channel importance"
    elif 'gradient' in str(method_name).lower() or 'grad' in str(method_name).lower():
        main = "Directional influence by channel"
    elif 'occlusion' in str(method_name).lower():
        main = "Occlusion test channel influence"
    else:
        main = f"Channel influence: {method_name}"
    model_line = f"({MODEL_LABEL.get(prefix, prefix)})"
    return wrap_multiline_title(f"{main}\n{model_line}")


def choose_and_plot_topomaps(prefix: str):
    vals_csv = SAVE_DIR / f"{prefix}_topomap_values.csv"
    cons_csv = SAVE_DIR / f"{prefix}_consensus_topomap.csv"

    if cons_csv.exists():
        d = pd.read_csv(cons_csv)
        if "Channel" in d.columns:
            ch = d["Channel"].astype(str).tolist()
            value_col = next((c for c in ["ConsensusImportance","Importance","Value"] if c in d.columns), None)
            if value_col is None:
                print(f"[Topomap] No value column in {cons_csv.name}")
            else:
                vals = d[value_col].to_numpy(float)
                info = _build_info_from_channels(ch, MONTAGE_NAME)
                order = info["ch_names"]
                vec = np.array([dict(zip(ch, vals)).get(c, np.nan) for c in order], float)
                vec = np.nan_to_num(vec, nan=(np.nanmean(vec) if np.isfinite(np.nanmean(vec)) else 0.0))
                cmap, vmin, vmax = choose_cmap_and_limits(vec)
                title = make_topomap_title(cons_csv.stem, None, prefix)
                _plot_topomap(vec, info, title, FIG_DIR / f"topomap_consensus_{prefix}",
                              "Consensus", vmin, vmax, cmap)

    if vals_csv.exists():
        d = pd.read_csv(vals_csv)
        if not {"Method","Channel","Importance"} <= set(d.columns):
            print(f"[Topomap] Unexpected columns in {vals_csv.name}")
        else:
            methods = sorted(d["Method"].astype(str).unique())
            all_ch = sorted(set(d["Channel"].astype(str)))
            info = _build_info_from_channels(all_ch, MONTAGE_NAME)
            order = info["ch_names"]
            for mth in methods:
                dm = d[d["Method"].astype(str) == mth]
                m = dm.set_index("Channel")["Importance"].to_dict()
                vec = np.array([m.get(c, np.nan) for c in order], float)
                vec = np.nan_to_num(vec, nan=(np.nanmean(vec) if np.isfinite(np.nanmean(vec)) else 0.0))
                cmap, vmin, vmax = choose_cmap_and_limits(vec)
                title = make_topomap_title(vals_csv.stem, mth, prefix)
                label = ("Gradient x Input" if ("grad" in mth.lower()) else
                         ("Occlusion" if "occlusion" in mth.lower() else "Importance"))
                safe = mth.lower().replace(" ", "_")
                _plot_topomap(vec, info, title, FIG_DIR / f"topomap_{prefix}_{safe}", label, vmin, vmax, cmap)

# ---------------- Run all (hard-coded) ----------------

def plot_all():
    for prefix in PREFIXES:
        print(f"\n=== Processing {prefix} ===")
        plot_roc(prefix)
        plot_pr(prefix)
        plot_pred_prob_hist(prefix)
        plot_confusion(prefix)
        choose_and_plot_topomaps(prefix)
    plot_calibration_combined()

if __name__ == "__main__":
    plot_all()
