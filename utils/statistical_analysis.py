from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, brier_score_loss

from main import SAVE_DIR
from utils.classical_features import expected_calibration_error



def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, random_state: int = 42):
    """Compute bootstrap confidence interval for a metric."""
    if len(values) < 2:
        return float('nan'), float('nan')

    rng = np.random.default_rng(random_state)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper


def paired_ttest_metrics(metrics_a: List[dict], metrics_b: List[dict], metric_name: str):
    """Perform paired t-test between two pipelines for a given metric."""
    vals_a = [m[metric_name] for m in metrics_a if np.isfinite(m[metric_name])]
    vals_b = [m[metric_name] for m in metrics_b if np.isfinite(m[metric_name])]

    min_len = min(len(vals_a), len(vals_b))
    if min_len < 2:
        return float('nan'), float('nan')

    vals_a = vals_a[:min_len]
    vals_b = vals_b[:min_len]

    t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
    return t_stat, p_value


def plot_confusion_matrices(per_fold_preds: List[dict], prefix: str):
    """Plot confusion matrices across folds."""
    n_folds = len(per_fold_preds)
    if n_folds == 0:
        return

    all_ytrue = np.concatenate([pf['ytrue'] for pf in per_fold_preds])
    all_probs = np.concatenate([pf['probs'] for pf in per_fold_preds])
    all_ypred = (all_probs >= 0.5).astype(int)

    cm_agg = confusion_matrix(all_ytrue, all_ypred)

    if n_folds > 1:
        n_cols = min(3, n_folds)
        n_rows = int(np.ceil((n_folds + 1) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = axes.flatten() if n_folds > 1 else [axes]

        for i, pf in enumerate(per_fold_preds):
            ytrue, probs = pf['ytrue'], pf['probs']
            ypred = (probs >= 0.5).astype(int)
            cm = confusion_matrix(ytrue, ypred)

            im = axes[i].imshow(cm, cmap='Blues', aspect='auto')
            axes[i].set_title(f'Fold {i + 1}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['Sham', 'Active'])
            axes[i].set_yticklabels(['Sham', 'Active'])

            for r in range(2):
                for c in range(2):
                    axes[i].text(c, r, str(cm[r, c]), ha='center', va='center',
                                 color='white' if cm[r, c] > cm.max() / 2 else 'black')

        if n_folds < len(axes):
            ax_agg = axes[n_folds]
            im = ax_agg.imshow(cm_agg, cmap='Blues', aspect='auto')
            ax_agg.set_title('Aggregated (All Folds)')
            ax_agg.set_xlabel('Predicted')
            ax_agg.set_ylabel('True')
            ax_agg.set_xticks([0, 1])
            ax_agg.set_yticks([0, 1])
            ax_agg.set_xticklabels(['Sham', 'Active'])
            ax_agg.set_yticklabels(['Sham', 'Active'])

            for r in range(2):
                for c in range(2):
                    ax_agg.text(c, r, str(cm_agg[r, c]), ha='center', va='center',
                                color='white' if cm_agg[r, c] > cm_agg.max() / 2 else 'black')

        for i in range(n_folds + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        save_path = SAVE_DIR / f"confusion_matrices_{prefix}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {save_path}")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(cm_agg, cmap='Blues', aspect='auto')
        ax.set_title(f'Confusion Matrix ({prefix})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Sham', 'Active'])
        ax.set_yticklabels(['Sham', 'Active'])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm_agg[r, c]), ha='center', va='center',
                        color='white' if cm_agg[r, c] > cm_agg.max() / 2 else 'black')

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        save_path = SAVE_DIR / f"confusion_matrix_{prefix}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {save_path}")


def plot_calibration_curve(per_fold_preds: List[dict], prefix: str, n_bins: int = 10):
    """Reliability diagram + per-class prob hist + ECE & Brier saved to CSV and printed."""
    if not per_fold_preds:
        return

    all_ytrue = np.concatenate([pf['ytrue'] for pf in per_fold_preds])
    all_probs = np.concatenate([pf['probs'] for pf in per_fold_preds])

    # ---- compute numbers ----
    ece = expected_calibration_error(all_ytrue, all_probs, n_bins=max(10, n_bins))
    brier = brier_score_loss(all_ytrue, all_probs) if all_probs.size else float('nan')

    # reliability curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    try:
        prob_true, prob_pred = calibration_curve(all_ytrue, all_probs, n_bins=n_bins, strategy='uniform')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax1.plot(prob_pred, prob_true, 's-', label=f'{prefix}', markersize=8)
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Calibration (ECE={ece:.3f}, Brier={brier:.3f})')
        ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1])
    except Exception as e:
        ax1.text(0.5, 0.5, f'Calibration curve error:\n{str(e)}',
                 ha='center', va='center', transform=ax1.transAxes)

    # probability histograms
    ax2.hist(all_probs[all_ytrue == 0], bins=20, alpha=0.5, label='Class 0', density=True)
    ax2.hist(all_probs[all_ytrue == 1], bins=20, alpha=0.5, label='Class 1', density=True)
    ax2.set_xlabel('Predicted Probability'); ax2.set_ylabel('Density')
    ax2.set_title('Predicted Probability Distribution'); ax2.legend(loc='upper center')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_img = SAVE_DIR / f"calibration_{prefix}.png"
    plt.savefig(save_img, dpi=200); plt.close()
    print(f"[Saved] {save_img}")

    # save numbers
    df_stats = pd.DataFrame([{
        "prefix": prefix,
        "ece": ece,
        "brier": brier,
        "n_samples": int(all_probs.size),
        "pos_rate": float(all_ytrue.mean()) if all_ytrue.size else float('nan')
    }])
    save_csv = SAVE_DIR / f"calibration_stats_{prefix}.csv"
    df_stats.to_csv(save_csv, index=False)
    print(f"[Saved] {save_csv}")


def export_statistical_results(sum_graph, sum_class, metrics_graph, metrics_class,
                               macro_graph, macro_class):
    """Export statistical test results including bootstrap CIs and paired t-tests."""
    rows = []

    for k, label in [("acc", "Accuracy"), ("bacc", "Balanced Acc"), ("f1", "F1"),
                     ("prec", "Precision"), ("rec", "Recall"), ("auc", "ROC AUC"), ("ap", "AP")]:
        row = {'Metric': label}

        if k in sum_graph:
            vals_g = [m[k] for m in metrics_graph if np.isfinite(m[k])]
            if vals_g:
                mean_g, std_g, n_g = sum_graph[k]
                ci_lower_g, ci_upper_g = bootstrap_ci(np.array(vals_g))
                row['Graph_Mean'] = mean_g
                row['Graph_Std'] = std_g
                row['Graph_CI_Lower'] = ci_lower_g
                row['Graph_CI_Upper'] = ci_upper_g
                row['Graph_N'] = n_g
            else:
                row['Graph_Mean'] = np.nan
                row['Graph_Std'] = np.nan
                row['Graph_CI_Lower'] = np.nan
                row['Graph_CI_Upper'] = np.nan
                row['Graph_N'] = 0

        if k in sum_class:
            vals_c = [m[k] for m in metrics_class if np.isfinite(m[k])]
            if vals_c:
                mean_c, std_c, n_c = sum_class[k]
                ci_lower_c, ci_upper_c = bootstrap_ci(np.array(vals_c))
                row['Classical_Mean'] = mean_c
                row['Classical_Std'] = std_c
                row['Classical_CI_Lower'] = ci_lower_c
                row['Classical_CI_Upper'] = ci_upper_c
                row['Classical_N'] = n_c
            else:
                row['Classical_Mean'] = np.nan
                row['Classical_Std'] = np.nan
                row['Classical_CI_Lower'] = np.nan
                row['Classical_CI_Upper'] = np.nan
                row['Classical_N'] = 0

        if k in sum_graph and k in sum_class:
            t_stat, p_value = paired_ttest_metrics(metrics_graph, metrics_class, k)
            row['T_Statistic'] = t_stat
            row['P_Value'] = p_value
            row['Significant'] = 'Yes' if p_value < 0.05 else 'No'
        else:
            row['T_Statistic'] = np.nan
            row['P_Value'] = np.nan
            row['Significant'] = 'N/A'

        rows.append(row)

    macro_row = {
        'Metric': 'Macro PR',
        'Graph_Mean': macro_graph,
        'Graph_Std': np.nan,
        'Graph_CI_Lower': np.nan,
        'Graph_CI_Upper': np.nan,
        'Graph_N': np.nan,
        'Classical_Mean': macro_class,
        'Classical_Std': np.nan,
        'Classical_CI_Lower': np.nan,
        'Classical_CI_Upper': np.nan,
        'Classical_N': np.nan,
        'T_Statistic': np.nan,
        'P_Value': np.nan,
        'Significant': 'N/A'
    }
    rows.append(macro_row)

    df = pd.DataFrame(rows)
    save_path = SAVE_DIR / "statistical_results.csv"
    df.to_csv(save_path, index=False)
    print(f"[Saved] {save_path}")

    return df


def print_statistical_summary(df_stats: pd.DataFrame):
    """Print formatted statistical summary to console."""
    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS WITH CONFIDENCE INTERVALS")
    print("=" * 100)
    print(f"{'Metric':<15} {'Graph (95% CI)':<35} {'Classical (95% CI)':<35} {'p-value':<10}")
    print("-" * 100)

    for _, row in df_stats.iterrows():
        metric = row['Metric']

        if np.isfinite(row['Graph_Mean']):
            graph_str = f"{row['Graph_Mean']:.3f} [{row['Graph_CI_Lower']:.3f}, {row['Graph_CI_Upper']:.3f}]"
        else:
            graph_str = "N/A"

        if np.isfinite(row['Classical_Mean']):
            class_str = f"{row['Classical_Mean']:.3f} [{row['Classical_CI_Lower']:.3f}, {row['Classical_CI_Upper']:.3f}]"
        else:
            class_str = "N/A"

        if np.isfinite(row['P_Value']):
            p_str = f"{row['P_Value']:.4f}"
            if row['P_Value'] < 0.001:
                p_str = "<0.001***"
            elif row['P_Value'] < 0.01:
                p_str = f"{row['P_Value']:.4f}**"
            elif row['P_Value'] < 0.05:
                p_str = f"{row['P_Value']:.4f}*"
        else:
            p_str = "N/A"

        print(f"{metric:<15} {graph_str:<35} {class_str:<35} {p_str:<10}")

    print("=" * 100)
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print("=" * 100)
