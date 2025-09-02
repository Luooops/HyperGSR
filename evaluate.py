import os
import json
import numpy as np
import hydra
import pandas as pd

from src.eval_metrics import evaluate

def _load_npy_matrix_array(path: str) -> np.ndarray:
    """Load npy that may contain a list of (n_t, n_t) arrays or a stacked (N, n_t, n_t) array."""
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # convert list-like object array to proper 3D numeric array
        arr = np.stack(list(arr), axis=0)
    return arr

def _save_metrics_csv(per_fold, avg, out_dir, filename="metrics.csv"):
    """
    per_fold: List[Dict[str, float]]
    avg: Dict[str, float]
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i, d in enumerate(per_fold, 1):
        row = {"fold": f"fold_{i}"}
        row.update({k: float(v) for k, v in d.items()})
        rows.append(row)
    avg_row = {"fold": "average"}
    avg_row.update({k: float(v) for k, v in avg.items()})
    rows.append(avg_row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, filename), index=False)
    print(f"Saved to {os.path.join(out_dir, filename)}")

@hydra.main(version_base="1.3.2", config_path="configs", config_name="experiment")
def main(config) -> None:
    base_dir = config.experiment.base_dir
    model_name = config.model.name
    dataset_type = config.dataset.name
    run_name = config.experiment.run_name
    if config.model.name == 'hyper_gsr':
        run_dir = f'{base_dir}/{model_name}/{dataset_type}/{config.model.hyper_dual_learner.mode}/{run_name}/'
    else:
        run_dir = f'{base_dir}/{model_name}/{dataset_type}/{run_name}/'

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    fold_dirs = sorted(
        [os.path.join(run_dir, d) for d in os.listdir(run_dir) if d.startswith("fold_")]
    )
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories under: {run_dir}")

    print(f"[Eval] run_dir = {run_dir}")
    per_fold = []

    for fd in fold_dirs:
        print(f"Evaluating fold {fd}")
        pred_path = os.path.join(fd, "eval_output.npy")
        gt_path   = os.path.join(fd, "target.npy")

        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            raise FileNotFoundError(f"Missing npy in {fd}. Need eval_output.npy and target.npy")

        pred = _load_npy_matrix_array(pred_path)  # shape: (N, n_t, n_t)
        gt   = _load_npy_matrix_array(gt_path)    # shape: (N, n_t, n_t)

        metrics = evaluate(pred, gt, show_progress=False)
        per_fold.append(metrics)

        pretty = {k: round(float(v), 6) for k, v in metrics.items()}
        print(f"  - {os.path.basename(fd)}: {pretty}")

    keys = per_fold[0].keys()
    avg = {k: float(np.mean([m[k] for m in per_fold])) for k in keys}
    pretty_avg = {k: round(v, 6) for k, v in avg.items()}

    print("\n[Average across folds]")
    for k, v in pretty_avg.items():
        print(f"  {k}: {v}")

    _save_metrics_csv(per_fold, avg, run_dir)

if __name__ == "__main__":
    main()
