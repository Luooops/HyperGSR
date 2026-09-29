"""Evaluate the 18 training runs, or one directory containing fold_* folders.

    python evaluate_all.py
    python evaluate_all.py --rerun
    python evaluate_all.py --path results/my_run

Existing metrics.csv files are skipped unless --rerun is supplied.
Default batch paths are relative to this script; --path is relative to your
current working directory (absolute paths are also accepted).
"""
import argparse
from pathlib import Path
import sys


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path', type=Path, help='One model directory containing fold_* folders')
    parser.add_argument('--rerun', action='store_true', help='Recompute and replace existing metrics.csv (default: false)')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if args.path is not None:
        paths = [args.path.expanduser().resolve()]
    else:
        # All three dataset presets have dataset.name=csv in the saved layout.
        paths = [root / base / model / 'csv' / suffix / run
                 for _, base in EXPERIMENTS for model, run in MODELS
                 for suffix in ['trans' if model == 'hyper_gsr' else '']]
    failed = []
    succeeded = skipped = 0
    for index, path in enumerate(paths, 1):
        print(f'\n[{index}/{len(paths)}] {path}', flush=True)
        if not args.rerun and (path / 'metrics.csv').is_file():
            skipped += 1
            print('SKIPPED: metrics.csv already exists (use --rerun to replace).', flush=True)
            continue
        try:
            if not path.is_dir():
                raise FileNotFoundError(f'Run directory not found: {path}')
            # Load scientific dependencies only when an evaluation is needed.
            from evaluate import evaluate_run
            evaluate_run(path)
            succeeded += 1
        except Exception as exc:
            failed.append((path, str(exc)))
            print(f'FAILED: {exc}', flush=True)
    print(f'\nFinished: {succeeded} succeeded, {skipped} skipped, {len(failed)} failed.')
    for path, error in failed:
        print(f'  FAILED: {path}: {error}')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nEvaluation interrupted.', file=sys.stderr)
        sys.exit(130)
