"""New publication figures for code recovery and resource-limited inheritance.

Run after the original generator:
    python -m analysis.revision_figures --root .
All plotted outcomes come from completed, recorded analysis outputs. This module
reuses analysis.figures styling and atomic PDF/SVG/PNG export, then merges the
existing figure_manifest.json with input and generator hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .figures import (
    BLUE, ORANGE, GREEN, PURPLE, INK, GRAY, PALE,
    plt, np, pd, style, panel, grid, save, rounded, arrow,
)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import t as student_t


def _cycle_label(permutation):
    seen, cycles = set(), []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current, cycle = start, []
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = int(permutation[current])
        if len(cycle) > 1:
            cycles.append('(' + ' '.join(map(str, cycle)) + ')')
    return ''.join(cycles) if cycles else 'identity'


def fig_permutation(root, out, registry):
    folder = 'results/permutation_dynamics/'
    summary = pd.read_csv(root / (folder + 'target_summary.csv'))
    periods = pd.read_csv(root / (folder + 'period_distribution.csv'))
    tracking = pd.read_csv(root / 'results/cycle_population/metric_tracking.csv')
    execution = json.loads((root / (folder + 'execution.json')).read_text())
    with np.load(root / (folder + 'actors_q03.npz')) as data:
        actors = data['actors'].copy()
    with np.load(root / (folder + 'q03_type_1-2.npz')) as data:
        successor = data['next_index'].copy()
        native_basin = data['native_basin'].copy()
        target = data['target'].copy()
    assert len(actors) == 6 and native_basin.sum() == 2
    assert execution['target_types'] == len(summary)
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.7), layout='constrained')

    ax = axes[0, 0]
    panel(ax, 'A', 'Complete three-symbol reader graph')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.50, .965, r'Fixed target $\pi=(1\ 2)$; $f\mapsto f\pi f^{-1}$',
            ha='center', va='top', fontsize=10)
    positions = {0: (.14, .73), 1: (.14, .30), 2: (.55, .30),
                 5: (.85, .30), 3: (.85, .73), 4: (.55, .73)}
    for index, next_index in enumerate(successor):
        start, finish = positions[index], positions[int(next_index)]
        color = BLUE if native_basin[index] else ORANGE
        if index == next_index:
            ax.add_patch(FancyArrowPatch((start[0]+.105, start[1]+.035),
                                        (start[0]+.105, start[1]-.035),
                                        connectionstyle='arc3,rad=-1.6',
                                        arrowstyle='-|>', mutation_scale=12,
                                        color=color, linewidth=1.5))
        else:
            radius = .24 if {index, int(next_index)} == {2, 5} else 0
            shrink = 25 if radius else 13
            ax.add_patch(FancyArrowPatch(start, finish,
                                        connectionstyle=f'arc3,rad={radius}',
                                        shrinkA=shrink, shrinkB=shrink, arrowstyle='-|>',
                                        mutation_scale=12, color=color, linewidth=1.5))
    for index, (x, y) in positions.items():
        color = BLUE if native_basin[index] else ORANGE
        rounded(ax, (x-.102, y-.065), .204, .13,
                '#EDF5FA' if native_basin[index] else '#FCF0E9', color)
        ax.text(x, y, _cycle_label(actors[index]), ha='center', va='center',
                fontsize=10, color=color)
    ax.text(.14, .12, 'Native basin\n2 of 6', ha='center', va='top', fontsize=9, color=BLUE)
    ax.text(.70, .12, 'Nonnative 2-cycle\n4 of 6', ha='center', va='top', fontsize=9, color=ORANGE)
    ax.text(.5, .855, r'Every state: success $=1$, information $=\log_2 3$',
            ha='center', va='center', fontsize=8.5, color=GRAY,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})

    ax = axes[0, 1]
    panel(ax, 'B', 'All target cycle types through eight symbols')
    for q, frame in summary.groupby('q', sort=True):
        ordered = frame.sort_values(['never_native_fraction', 'cycle_type'])
        offsets = np.linspace(-.17, .17, len(ordered)) if len(ordered) > 1 else [0]
        ax.scatter(q+np.asarray(offsets), ordered.never_native_fraction,
                   s=23, facecolor='#9BB6C9', edgecolor='white', linewidth=.45, zorder=3)
    transpositions = summary[summary.transposition_prediction_denominator.notna()].sort_values('q')
    ax.plot(transpositions.q, transpositions.never_native_fraction, color=ORANGE,
            marker='D', ms=4, lw=1.5, label='Transposition target (exact)', zorder=4)
    ax.set(xlim=(.65, 8.4), ylim=(-.035, 1.045), xticks=range(1, 9),
           xlabel=r'Alphabet size $q$', ylabel='Never-native fraction\n(uniform initial permutations)')
    grid(ax, 'y')
    handles = [Line2D([], [], marker='o', ls='', color='#9BB6C9', ms=5,
                      label='One target per cycle type'),
               Line2D([], [], marker='D', color=ORANGE, ms=4,
                      label='Transposition target (exact)')]
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, .13), fontsize=8)
    ax.text(.02, .965, f"{int(execution['target_types'])} types; {int(execution['actor_target_pairs']):,} actor–target pairs",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .92, 'pad': 2})

    ax = axes[1, 0]
    panel(ax, 'C', r'An eight-symbol target with a 192-cycle')
    selected = periods[(periods.q == 8) & (periods.cycle_type == '1-3-4')].sort_values('period')
    target_row = summary[(summary.q == 8) & (summary.cycle_type == '1-3-4')].iloc[0]
    assert int(selected.actor_count.sum()) == int(target_row.total_actors)
    positions = np.arange(len(selected))
    colors = [BLUE if value == 1 else ORANGE if value == 192 else '#AEC2CE'
              for value in selected.period]
    ax.barh(positions, selected.actor_fraction, color=colors, height=.65, zorder=3)
    for y, row in zip(positions, selected.itertuples()):
        label = f'{row.actor_count:,}' + (' native' if row.period == 1 else '')
        ax.text(row.actor_fraction+.015, y, label, va='center', fontsize=8.5)
    ax.set_yticks(positions, [str(int(period)) for period in selected.period])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.02), xlabel='Fraction of all 40,320 initial readers',
           ylabel='Eventual cycle period')
    ax.set_xticks([0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1.00'])
    grid(ax, 'x')
    ax.text(.98, .96, r'Target cycle type $(1,3,4)$', transform=ax.transAxes,
            ha='right', va='top', fontsize=8.7)

    ax = axes[1, 1]
    panel(ax, 'D', 'Noiseless paths at finite decoding width')
    for q, horizon, color, marker in [(3, 32, BLUE, 'o'), (8, 192, ORANGE, 's')]:
        for metric, linestyle in [('linear','-'),('equidistant','--')]:
            data=tracking[(tracking.q==q)&(tracking.horizon==horizon)&(tracking.metric==metric)].sort_values('sigma_D')
            ax.plot(data.sigma_D,data.path_probability,color=color,marker=marker,ms=3.5,ls=linestyle,
                    label=f'$q={q}$: '+('linear' if metric=='linear' else 'equal distances'))
    ax.set(xlim=(.143,.257),ylim=(0,1.06),xticks=[.15,.2,.25],
           xlabel=r'Conditional decoding width $\sigma_D$',
           ylabel='Specified noiseless-path probability')
    grid(ax);ax.legend(loc='lower left',fontsize=7.8)
    ax.text(.03,.57,'Horizons: 32 births ($q=3$)\n192 births ($q=8$)',transform=ax.transAxes,fontsize=8.2)
    sources = [folder + name for name in ('target_summary.csv', 'period_distribution.csv',
               'finite_width_shadowing.csv', 'execution.json', 'actors_q03.npz', 'q03_type_1-2.npz')]
    sources.append('results/cycle_population/metric_tracking.csv')
    save(fig, out, 'fig6_permutation', sources, registry)


def fig_resource_limit(root, out, registry):
    folder = 'results/resource_population/'
    errors = pd.read_csv(root / (folder + 'fluid_errors.csv'))
    slopes = pd.read_csv(root / (folder + 'fluid_scaling_descriptive.csv'))
    means = pd.read_csv(root / (folder + 'fluid_mean_trajectories.csv'))
    primary = errors[errors.sweep == 100]
    assert len(primary) == 10 and set(primary.independent_replicates) == {200}
    fig = plt.figure(figsize=(8.7, 5.7), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.03, 1.2])
    ax = fig.add_subplot(gs[:, 0])
    panel(ax, 'A', 'Four-class density error at sweep 100')
    for sigma, color, marker in [(.5, BLUE, 'o'), (.8, ORANGE, 's')]:
        data = primary[np.isclose(primary.sigma, sigma)].sort_values('scale')
        yerr = np.vstack([data.mse-data.mse_ci95_low, data.mse_ci95_high-data.mse])
        ax.errorbar(data.scale, data.mse, yerr=yerr, color=color, marker=marker,
                    ms=5, capsize=3, lw=1.7, label=f'$\\sigma={sigma:g}$', zorder=4)
    guide_x = np.array([1., 16.])
    ax.plot(guide_x, 200/guide_x, color=GRAY, ls=':', lw=1.5,
            label=r'$1/\Omega$ reference', zorder=1)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set(xlim=(.83, 19), ylim=(5, 650), xlabel=r'Population and resource scale $\Omega$',
           ylabel=r'Mean squared density error $\|n/\Omega-x\|_2^2$')
    ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
    ax.set_yticks([10, 30, 100, 300], ['10', '30', '100', '300'])
    grid(ax)
    ax.legend(loc='upper right', fontsize=9)
    slope05 = slopes[(np.isclose(slopes.sigma, .5)) & (slopes.sweep == 100)].iloc[0]
    slope08 = slopes[(np.isclose(slopes.sigma, .8)) & (slopes.sweep == 100)].iloc[0]
    ax.text(.04, .06, '200 independent paths per condition\nApprox. t intervals over squared errors\n\n'
            'Descriptive log–log slopes:\n'
            f'$\\sigma=0.5$: {slope05.log_mse_log_scale_slope:.3f}\n'
            f'$\\sigma=0.8$: {slope08.log_mse_log_scale_slope:.3f}',
            transform=ax.transAxes, va='bottom', fontsize=8.7,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .94, 'pad': 3})
    sources = [folder + name for name in ('fluid_errors.csv', 'fluid_scaling_descriptive.csv',
                                         'fluid_mean_trajectories.csv', 'plan_executed.json')]
    for row, sigma in enumerate((.5, .8)):
        ax = fig.add_subplot(gs[row, 1])
        panel(ax, 'BC'[row], f'Two-value reader density, $\\sigma={sigma:g}$')
        predicted = means[(np.isclose(means.sigma, sigma)) & (means.scale == 1)].sort_values('sweep')
        ax.plot(predicted.sweep, predicted.fluid_A+predicted.fluid_P, color=INK,
                lw=2, label='Fluid recursion', zorder=5)
        for scale, color, linestyle, marker in [(1, ORANGE, '--', None), (16, BLUE, ':', 'o')]:
            filename = folder + f'fluid/sigma{sigma:g}_scale{scale}.npz'
            sources.append(filename)
            with np.load(root / filename) as data:
                assert tuple(data['class_names']) == ('Z', 'C', 'A', 'P')
                counts = data['counts'].copy()
                sweeps = data['sweep'].copy()
            values = (counts[:, :, 2]+counts[:, :, 3]) / scale
            mean = values.mean(axis=0)
            half = student_t.ppf(.975, len(values)-1)*values.std(axis=0, ddof=1)/math.sqrt(len(values))
            saved = means[(np.isclose(means.sigma, sigma)) & (means.scale == scale)].sort_values('sweep')
            assert np.allclose(mean, saved.mean_A+saved.mean_P)
            ax.fill_between(sweeps, mean-half, mean+half, color=color, alpha=.13, linewidth=0)
            ax.plot(sweeps, mean, color=color, ls=linestyle, lw=1.5,
                    marker=marker, markevery=20, ms=3.5, label=f'Mean, $\\Omega={scale}$', zorder=4)
        ax.set(xlim=(0, 200), xlabel='Sweep', ylabel=r'$(A+P)/\Omega$')
        grid(ax)
        if row == 0:
            ax.set_ylim(50, 115)
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_ylim(0, 85)
            ax.text(.98, .95, 'Shading: approx. pointwise 95% CI\nfor the mean across 200 paths',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8.2)
    save(fig, out, 'fig7_resource_limit', sources, registry)


def _reader_word(ax, word, x, y, width=.051, height=.12):
    """Keys stay neutral; values are colored by their recorded binary symbol."""
    for position, symbol in enumerate(word):
        left = x + position*width
        is_key = position % 2 == 0
        face = PALE if is_key else BLUE if symbol == 0 else ORANGE
        edge = '#A2AEB8' if is_key else 'white'
        ax.add_patch(Rectangle((left, y), width-.004, height,
                               facecolor=face, edgecolor=edge, linewidth=.8))
        ax.text(left+(width-.004)/2, y+height/2, str(symbol),
                ha='center', va='center', fontsize=11,
                color=INK if is_key else 'white', fontweight='bold')


def fig_rewiring(root, out, registry):
    folder = 'results/resource_population/'
    trajectories = pd.read_csv(root / (folder + 'rewiring_mean_trajectories.csv'))
    groups = pd.read_csv(root / (folder + 'rewiring_groups.csv'))
    contrast = pd.read_csv(root / (folder + 'rewiring_contrasts.csv'))
    checkpoints = json.loads((root / (folder + 'checkpoints.json')).read_text())
    primary = contrast[contrast.metric == 'survival500'].iloc[0]
    identifiers = sorted(groups.checkpoint_id.unique())
    assert len(identifiers) == int(primary.fixed_checkpoints)
    assert set(groups.independent_continuations) == {200}
    assert all(all(point['checks'].values()) for point in checkpoints)
    for point in checkpoints:
        delta = np.array(point['after_counts'])-np.array(point['before_counts'])
        expected = np.zeros(16, dtype=int)
        expected[[0, 15]] = -10
        expected[[1, 14]] = 10
        assert np.array_equal(delta, expected)
    fig = plt.figure(figsize=(8.7, 6.4), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1., 1.6])
    ax = fig.add_subplot(gs[0, :])
    panel(ax, 'A', 'A one-time exchange of existing value slots')
    ax.texts[-1].set_x(-.06)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    before_x, after_x = .14, .66
    ax.text(before_x+.10, .96, 'Homogeneous pair', ha='center', fontsize=10)
    ax.text(after_x+.10, .96, 'Two-value pair', ha='center', fontsize=10)
    for x in (before_x, after_x):
        for i, label in enumerate((r'$k_1$', r'$v_1$', r'$k_2$', r'$v_2$')):
            ax.text(x+i*.051+.024, .81, label, ha='center', fontsize=8.5, color=GRAY)
    _reader_word(ax, (0, 0, 0, 0), before_x, .61)
    _reader_word(ax, (1, 1, 1, 1), before_x, .39)
    _reader_word(ax, (0, 0, 0, 1), after_x, .61)
    _reader_word(ax, (1, 1, 1, 0), after_x, .39)
    arrow(ax, (.41, .56), (.60, .56), PURPLE)
    ax.text(.505, .70, r'Exchange $v_2$', ha='center', fontsize=10, color=PURPLE)
    ax.text(.505, .43, '10 disjoint pairs', ha='center', fontsize=9, color=GRAY)
    ax.text(.5, .23, 'Same population, resource, key slots, per-position symbol totals, and one-step birth probabilities.',
            ha='center', fontsize=8.9)
    ax.text(.5, .08, 'Initially: no native or P readers; zero decoding information in both arms.',
            ha='center', fontsize=8.9, color=GRAY)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, 'B', 'Continuation after repertoire loss')
    for arm, color, label in [('unchanged', GRAY, 'Unchanged'), ('rewired', ORANGE, 'Rewired')]:
        data = trajectories[trajectories.arm == arm]
        assert data.groupby('sweep').size().eq(len(identifiers)).all()
        mean = data.groupby('sweep').survival_fraction.mean()
        ax.step(mean.index, mean.values, where='post', color=color, lw=2, label=label)
        endpoint = groups[groups.arm == arm]
        count = int(endpoint.survival500_count.sum())
        n = int(endpoint.independent_continuations.sum())
        assert np.isclose(mean.iloc[-1], count/n)
        ax.scatter([500], [count/n], color=color, s=25, zorder=4)
        offset = .11 if arm == 'rewired' else .045
        ax.text(490, count/n+offset, f'{count:,}/{n:,}', ha='right', fontsize=9, color=color)
    ax.set(xlim=(-5, 515), ylim=(-.025, 1.045), xlabel='Sweeps after intervention',
           ylabel='Fraction of continuations still nonempty')
    grid(ax)
    ax.legend(loc='upper right', fontsize=9)
    ax.text(.97, .43, f"Survival gain: {100*primary.difference:.1f} pp\n"
            f"95% Hoeffding: [{100*primary.ci95_low:.1f}, {100*primary.ci95_high:.1f}] pp\n"
            'Conditional on 10 fixed checkpoints',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.7,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .94, 'pad': 3})

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, 'C', 'Survival at sweep 500 by checkpoint')
    y = np.arange(len(identifiers))
    for arm, color, shift, marker in [('unchanged', GRAY, -.10, 'o'),
                                     ('rewired', ORANGE, .10, 's')]:
        data = groups[groups.arm == arm].set_index('checkpoint_id').loc[identifiers]
        x = 100*data.survival500_fraction.to_numpy()
        lower = 100*data.survival500_wilson_low.to_numpy()
        upper = 100*data.survival500_wilson_high.to_numpy()
        ax.errorbar(x, y+shift,
                    xerr=np.vstack([np.maximum(0, x-lower), np.maximum(0, upper-x)]),
                    fmt=marker, color=color, ms=4, capsize=2.5, elinewidth=1.1, zorder=4)
    ax.set_yticks(y, [str(i) for i in identifiers])
    ax.invert_yaxis()
    ax.set(xlim=(-4, 85), xlabel='Nonempty at sweep 500 (%)',
           ylabel='Checkpoint: original repeat ID')
    ax.set_xticks([0, 20, 40, 60, 80])
    grid(ax, 'x')
    ax.text(.35, .97, '200 paths per arm\nWilson 95% intervals',
            transform=ax.transAxes, ha='center', va='top', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .94, 'pad': 2})
    sources = [folder + name for name in ('rewiring_mean_trajectories.csv', 'rewiring_groups.csv',
               'rewiring_contrasts.csv', 'checkpoints.json', 'plan_executed.json')]
    save(fig, out, 'fig8_rewiring', sources, registry)


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=path.parent,
                                     suffix='.json', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(root=None, only=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    out = root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    style()
    registry = {}
    renderers = {6: fig_permutation, 7: fig_resource_limit, 8: fig_rewiring}
    for number in (sorted(renderers) if only is None else only):
        renderers[number](root, out, registry)
    for item in registry.values():
        item['sources'] = list(dict.fromkeys(item['sources'] + ['analysis/figures.py']))
        item['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in item['sources']}
        item['generator'] = 'analysis/revision_figures.py'
        item['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = out / 'figure_manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    previous.update(registry)
    _atomic_json(manifest, previous)
    print(json.dumps({'figures': list(registry), 'formats': ['pdf', 'svg', 'png'],
                      'manifest': str(manifest)}, indent=2))
    return registry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--only', type=int, nargs='+', choices=[6, 7, 8])
    arguments = parser.parse_args()
    run(arguments.root, arguments.only)
