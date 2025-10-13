from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from config import CFG
SAVE_DIR = Path(CFG.data.save_dir)



def fmt_num(x: float, width: int = 0) -> str:
    s = "nan" if (x is None or not np.isfinite(x)) else f"{x:.3f}"
    return f"{s:<{width}}" if width else s


def fmt_triplet(t) -> str:
    if not t:
        return "—"
    mean, sd, n = t
    return f"{fmt_num(mean)} ± {fmt_num(sd)} (n={n})"


def export_results_to_csv(sum_graph, sum_class, macro_graph, macro_class, best_params_graph, best_params_class):
    """Export results to CSV for LaTeX table."""
    rows = []

    for k, label in [("acc", "Accuracy"), ("bacc", "Balanced Acc"), ("f1", "F1"),
                     ("prec", "Precision"), ("rec", "Recall"), ("auc", "ROC AUC"), ("ap", "AP")]:
        row = {'Metric': label}
        if k in sum_graph:
            mean, sd, n = sum_graph[k]
            row['Graph_Mean'] = mean
            row['Graph_Std'] = sd
            row['Graph_N'] = n
        else:
            row['Graph_Mean'] = np.nan
            row['Graph_Std'] = np.nan
            row['Graph_N'] = 0

        if k in sum_class:
            mean, sd, n = sum_class[k]
            row['Classical_Mean'] = mean
            row['Classical_Std'] = sd
            row['Classical_N'] = n
        else:
            row['Classical_Mean'] = np.nan
            row['Classical_Std'] = np.nan
            row['Classical_N'] = 0
        rows.append(row)

    rows.append({
        'Metric': 'Macro PR',
        'Graph_Mean': macro_graph,
        'Graph_Std': np.nan,
        'Graph_N': np.nan,
        'Classical_Mean': macro_class,
        'Classical_Std': np.nan,
        'Classical_N': np.nan
    })

    df_metrics = pd.DataFrame(rows)
    metrics_path = SAVE_DIR / "results_summary.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[Saved] {metrics_path}")

    params_rows = []
    if best_params_graph:
        for param, value in best_params_graph.items():
            params_rows.append({'Pipeline': 'Graph', 'Parameter': param, 'Value': value})
    if best_params_class:
        for param, value in best_params_class.items():
            params_rows.append({'Pipeline': 'Classical', 'Parameter': param, 'Value': value})

    if params_rows:
        df_params = pd.DataFrame(params_rows)
        params_path = SAVE_DIR / "best_parameters.csv"
        df_params.to_csv(params_path, index=False)
        print(f"[Saved] {params_path}")
