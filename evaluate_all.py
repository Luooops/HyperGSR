"""Evaluate the 18 previously specified training runs; never launch training.

Run in the hypergsr environment:
    python evaluate_all.py

Each successful evaluation writes metrics.csv into its existing run directory
(replacing metrics.csv if already present). Paths are relative to this script.
"""
from pathlib import Path
import subprocess
import sys


# Exact dataset presets and output roots from the original 18 training CLIs.
EXPERIMENTS = [
    ("csv", "results/ablation_v3"),
    ("morph35_func160", "results/cross_modal_35_160"),
    ("morph35_func268", "results/cross_modal_35_268"),
]
MODELS = [
    ("stp_gsr", "stp_gsr"),
    ("direct_sr", "direct_sr"),
    ("hyper_gsr", "hypergsr_baseline"),
    ("hyper_gsr", "hypergsr_geo"),
    ("hyper_gsr", "hypergsr_shrink001"),
    ("hyper_gsr", "hypergsr_geo_shrink001"),
]


def main():
    root = Path(__file__).resolve().parent
    failed = []
    total = len(EXPERIMENTS) * len(MODELS)
    completed = 0
    for dataset, base_dir in EXPERIMENTS:
        for model, run_name in MODELS:
            completed += 1
            label = f"{base_dir} / {run_name}"
            print(f"\n[{completed}/{total}] {label}", flush=True)
            command = [
                sys.executable, "-u", str(root / "evaluate.py"),
                f"dataset={dataset}", f"model={model}",
                f"experiment.base_dir={base_dir}",
                f"experiment.run_name={run_name}",
                "hydra.job.chdir=false",
            ]
            if model == "hyper_gsr":
                command.append("model.hyper_dual_learner.mode=trans")
            # Evaluation reads saved predictions, so training-only overrides
            # (geometry, shrinkage, epochs, etc.) are not needed here.
            result = subprocess.run(command, cwd=root, check=False)
            if result.returncode:
                failed.append((label, result.returncode))
                print(f"FAILED (exit {result.returncode}): {label}", flush=True)

    print(f"\nFinished: {total - len(failed)}/{total} evaluations succeeded.")
    for label, code in failed:
        print(f"  FAILED (exit {code}): {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.", file=sys.stderr)
        sys.exit(130)
