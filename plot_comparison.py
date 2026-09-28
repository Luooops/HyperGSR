"""Publication-style comparisons of any number of evaluated models.

Example:
    python plot_comparison.py --results_paths run_a run_b --names Full Ablation --no_show
Paths may point to metrics.csv or its parent directory. All eight metrics are errors
(lower is better). Error bars are sample SD across folds, not confidence intervals.
"""
import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_STP_PATH = 'results/stp_gsr/csv/run_stp_gsr/metrics.csv'
DEFAULT_EDGEATTR_PATH = 'results/hyper_gsr/csv/trans/run_hyper_emb_edgeattr_shrink001/metrics.csv'
DEFAULT_COORD_PATH = 'results/hyper_gsr/csv/trans/run_hyper_emb_coord_shrink001/metrics.csv'
METRICS = ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz',
           'clustering_diff', 'laplacian_frobenius_distance']
LABELS = ['MAE', 'Degree MAE', 'Betweenness MAE', 'Eigenvector MAE',
          'PageRank MAE', 'Katz MAE', 'Clustering difference', 'Laplacian distance']
STYLE = {'font.family': 'sans-serif', 'font.size': 9, 'axes.titlesize': 10,
         'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.linewidth': .6, 'axes.edgecolor': '#777777', 'text.color': '#252525',
         'axes.labelcolor': '#252525', 'xtick.color': '#454545', 'ytick.color': '#454545',
         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}


def load_data(stp_gsr_path=DEFAULT_STP_PATH, hyper_edgeattr_path=DEFAULT_EDGEATTR_PATH,
              hyper_coord_path=DEFAULT_COORD_PATH, *, results_paths=None, names=None):
    """Load any number of runs; retain the original three-path Python API."""
    if results_paths is None:
        results_paths = [stp_gsr_path, hyper_edgeattr_path, hyper_coord_path]
        if names is None:
            names = ['STP-GSR', 'Hyper-GSR (EdgeAttr)', 'Hyper-GSR (Coord)']
    paths = [Path(p).expanduser() for p in results_paths]
    paths = [p / 'metrics.csv' if p.is_dir() else p for p in paths]
    if not paths:
        raise ValueError('Provide at least one results path.')
    if names is None:
        names = [p.parent.name if p.name == 'metrics.csv' else p.stem for p in paths]
    if len(names) != len(paths) or any(not str(n).strip() for n in names):
        raise ValueError('Provide exactly one non-empty name per results path.')
    if len(set(names)) != len(names):
        raise ValueError('Model names must be unique; use --names for ambiguous directory names.')
    frames = []
    for path, name in zip(paths, names):
        frame = pd.read_csv(path)
        frame['Method'] = name
        frames.append(frame)
    _summarize(frames)  # Validate before writing any outputs.
    return tuple(frames)


def _summarize(data):
    means, spreads, folds, names = [], [], [], []
    for frame in data:
        missing = set(['fold', 'Method'] + METRICS) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        if frame.empty or frame['Method'].nunique() != 1:
            raise ValueError('Each input must contain one non-empty model.')
        name = str(frame['Method'].iloc[0])
        tags = frame['fold'].astype(str).str.strip().str.lower()
        if tags.duplicated().any():
            raise ValueError(f'{name}: duplicate fold/average rows.')
        if not (tags.eq('average') | tags.str.fullmatch(r'fold_\d+')).all():
            raise ValueError(f'{name}: expected fold_N or average in the fold column.')
        values = frame[METRICS].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
            raise ValueError(f'{name}: metrics must be finite, non-negative numbers.')
        raw = values.loc[tags != 'average']
        avg = values.loc[tags == 'average']
        means.append(avg.iloc[0] if len(avg) else raw.mean())
        spreads.append(raw.std(ddof=1))
        folds.append(raw)
        names.append(name)
    if not names or len(set(names)) != len(names):
        raise ValueError('Provide one or more models with unique names.')
    return (pd.DataFrame(means, index=names), pd.DataFrame(spreads, index=names), folds)


def _inputs(data):
    return _summarize(load_data() if data is None else data)


def _palette(n):
    # Stable, color-vision-friendly first ten colors; extend for larger comparisons.
    base = list(sns.color_palette('colorblind', 10))
    return base[:n] if n <= 10 else base + list(sns.color_palette('husl', n - 10))


def _wrap(names, width=23):
    return [textwrap.fill(str(name), width=width) for name in names]


def _save(fig, stem, output_dir, show, formats, dpi):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output / f'{stem}.{fmt}', dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def improvement_percent(means, baseline=None):
    """Zero baseline has undefined relative improvement and is exported as NaN."""
    baseline = means.index[0] if baseline is None else baseline
    if baseline not in means.index:
        raise ValueError(f'Unknown baseline {baseline!r}; choose one of {list(means.index)}')
    ref = means.loc[baseline].replace(0, np.nan)
    return (means.loc[baseline] - means.drop(index=baseline)).div(ref).mul(100)


def create_bar_plots(data=None, *, output_dir='.', show=True, formats=('png', 'pdf'), dpi=600):
    means, sd, folds = _inputs(data)
    n = len(means)
    colors = _palette(n)
    # Horizontal bars keep long ablation names readable, with no fixed model limit.
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(17, max(6.4, 2.4 + .66 * n)),
                                 layout='constrained')
        fig.suptitle('Model comparison · lower is better\nBars: reported average (or fold mean); whiskers: fold SD; dots: folds', fontsize=11)
        for k, (ax, metric, label) in enumerate(zip(axes.flat, METRICS, LABELS)):
            y = np.arange(n)
            vals = means[metric].to_numpy()
            ax.barh(y, vals, color=colors, height=.62, zorder=2)
            for j, raw in enumerate(folds):
                if len(raw) >= 2:
                    ax.errorbar(vals[j], j, xerr=sd.iloc[j][metric], fmt='none',
                                ecolor='#333333', capsize=2, elinewidth=.8, zorder=3)
                if len(raw):
                    ax.scatter(raw[metric], j + np.linspace(-.13, .13, len(raw)),
                               s=9, facecolor='white', edgecolor='#444444', linewidth=.5, zorder=4)
            extent = np.maximum(vals + sd[metric].fillna(0).to_numpy(),
                                [raw[metric].max() if len(raw) else 0 for raw in folds])
            limit = max(float(extent.max()), 1e-12)
            for j, v in enumerate(vals):
                ax.text(extent[j] + .025 * limit, j, f'{v:.3g}', va='center', fontsize=8)
            ax.set_xlim(0, limit * 1.3)
            ax.set_yticks(y, _wrap(means.index))
            ax.invert_yaxis()
            ax.set_title(f'{chr(97 + k)}   {label}', loc='left', pad=9)
            ax.grid(axis='x', color='#e6e6e6', linewidth=.6, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis='y', length=0)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 4), useMathText=True)
            ax.xaxis.get_major_locator().set_params(nbins=4)
        _save(fig, 'metrics_comparison', output_dir, show, formats, dpi)


def create_improvement_heatmap(data=None, *, output_dir='.', show=True,
                               baseline=None, formats=('png', 'pdf'), dpi=600):
    means, _, _ = _inputs(data)
    baseline = means.index[0] if baseline is None else baseline
    improvement = improvement_percent(means, baseline)
    if improvement.empty:
        return improvement
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, max(2.6, 1.6 + .48 * len(improvement))), layout='constrained')
        bound = max(1., float(improvement.abs().max().max())) if improvement.notna().any().any() else 1.
        sns.heatmap(improvement, ax=ax, annot=True, fmt='.1f', cmap='BrBG',
                    vmin=-bound, vmax=bound, center=0, linewidths=.6, linecolor='white',
                    cbar_kws={'label': 'Relative improvement (%)'},
                    xticklabels=_wrap(LABELS, 14), yticklabels=_wrap(improvement.index))
        for row, col in np.argwhere(improvement.isna().to_numpy()):
            ax.text(col + .5, row + .5, 'N/A', ha='center', va='center', fontsize=8)
        ax.tick_params(axis='both', rotation=0, length=0)
        ax.set_title(f'Relative to {baseline} · positive is better\nN/A: baseline is zero', pad=12)
        _save(fig, 'improvement_heatmap', output_dir, show, formats, dpi)
    return improvement


def create_radar_chart(data=None, *, output_dir='.', show=True, formats=('png', 'pdf'), dpi=600):
    means, _, _ = _inputs(data)
    span = means.max() - means.min()
    scores = (1 - (means - means.min()).div(span.replace(0, np.nan))).fillna(.5)
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    angles = np.r_[angles, angles[0]]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, max(6, .28 * len(means))), subplot_kw={'projection': 'polar'}, layout='constrained')
        for i, (name, row) in enumerate(scores.iterrows()):
            ax.plot(angles, np.r_[row.to_numpy(), row.iloc[0]], color=_palette(len(means))[i],
                    linewidth=1.4, marker='o', markersize=3, linestyle=['-', '--', ':', '-.'][i % 4], label=name)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1], _wrap(LABELS, 16))
        ax.tick_params(axis='x', pad=18)
        for angle, tick in zip(angles[:-1], ax.get_xticklabels()):
            side = np.sin(angle)
            tick.set_horizontalalignment('left' if side > .1 else 'right' if side < -.1 else 'center')
        ax.set_ylim(0, 1.05)
        ax.set_yticks([.25, .5, .75, 1])
        ax.grid(color='#dddddd', linewidth=.6)
        ax.spines['polar'].set_color('#cccccc')
        ax.set_title('Within-comparison min–max scores · higher is better\nTied metrics = 0.5; scores depend on included models', pad=26)
        ax.legend(loc='center left', bbox_to_anchor=(1.22, .5), frameon=False)
        _save(fig, 'radar_chart', output_dir, show, formats, dpi)


def create_summary_table(data=None, *, output_dir='.', show=True, formats=('png', 'pdf'), dpi=600):
    means, sd, _ = _inputs(data)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(15, max(2.6, 1.2 + .5 * len(means))), layout='constrained')
        ax.axis('off')
        cells = [[f'{v:.4g}' + (f' ± {s:.2g}' if pd.notna(s) else '')
                  for v, s in zip(means.loc[name], sd.loc[name])] for name in means.index]
        table = ax.table(cellText=cells, rowLabels=_wrap(means.index), colLabels=_wrap(LABELS, 14),
                         cellLoc='center', bbox=[.2, 0, .8, .88])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#e8edf1' if r == 0 else ('#f4f6f8' if r % 2 else 'white'))
            if r == 0:
                cell.set_text_props(weight='bold')
            elif c >= 0 and means.iloc[r - 1, c] == means.iloc[:, c].min():
                cell.set_text_props(weight='bold')
        ax.set_title('Metric summary · lower is better\nAverage ± fold SD (when ≥2 folds); bold indicates column minimum', pad=10)
        _save(fig, 'summary_table', output_dir, show, formats, dpi)
    return means.T.rename(index=dict(zip(METRICS, LABELS)))


def main(data=None, *, output_dir='.', show=True, baseline=None, formats=('png', 'pdf'), dpi=600):
    data = load_data() if data is None else tuple(data)
    means, sd, folds = _summarize(data)
    improvement = improvement_percent(means, baseline)
    options = dict(output_dir=output_dir, show=show, formats=formats, dpi=dpi)
    create_bar_plots(data, **options)
    create_improvement_heatmap(data, baseline=baseline, **options)
    create_radar_chart(data, **options)
    create_summary_table(data, **options)
    means.to_csv(Path(output_dir) / 'summary_metrics.csv', index_label='model')
    sd.to_csv(Path(output_dir) / 'fold_sd.csv', index_label='model')
    improvement.to_csv(Path(output_dir) / 'improvement_percent.csv', index_label='model', na_rep='N/A')
    print(f'Saved comparisons for {len(means)} models to {Path(output_dir).resolve()}')
    print('Fold counts: ' + ', '.join(f'{name}={len(raw)}' for name, raw in zip(means.index, folds)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_paths', '--results-paths', nargs='+', help='Any number of metrics.csv files or run directories')
    parser.add_argument('--names', nargs='+', help='Model names, in the same order as paths')
    parser.add_argument('--baseline', help='Baseline model name (default: first input)')
    parser.add_argument('--stp_path', default=DEFAULT_STP_PATH, help='Legacy three-model input')
    parser.add_argument('--edgeattr_path', default=DEFAULT_EDGEATTR_PATH)
    parser.add_argument('--coord_path', default=DEFAULT_COORD_PATH)
    parser.add_argument('--output_dir', '--output-dir', default='.')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--no_show', '--no-show', action='store_true')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    try:
        data = load_data(args.stp_path, args.edgeattr_path, args.coord_path,
                         results_paths=args.results_paths, names=args.names)
        main(data, output_dir=args.output_dir, show=not args.no_show,
             baseline=args.baseline, formats=args.formats, dpi=args.dpi)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
