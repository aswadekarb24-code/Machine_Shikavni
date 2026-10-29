"""gen_tables.py — Reproducible data pipeline for conference_101719.tex

Run from the project root:
    python -m papers.gen_tables

Produces:
  papers/figures/fig_pipeline_comparison.pdf
  papers/figures/fig_ensemble_vs_single.pdf
  papers/figures/fig_per_corruption_ensemble.pdf
  papers/figures/fig_specialist_heatmap.pdf
  papers/latex_snippets.txt   (all table LaTeX for copy-paste)
"""

import glob
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
PALETTE = {
    'trialcnn':       '#2563EB',   # blue
    'resnet18':       '#16A34A',   # green
    'mobilenet_v2':   '#DC2626',   # red
    'efficientnet_b0':'#D97706',   # amber
    'ensemble_32':    '#7C3AED',   # purple
}
ARCH_LABEL = {
    'trialcnn':       'TrialCNN',
    'resnet18':       'ResNet-18',
    'mobilenet_v2':   'MobileNetV2',
    'efficientnet_b0':'EfficientNet-B0',
    'ensemble_32':    'Ensemble-32',
}
MEMBER_LABEL = {
    'noise': 'Noise', 'light': 'Light', 'weather': 'Weather', 'fog': 'Fog',
    'defocus': 'Defocus', 'heavyblur': 'HeavyBlur', 'motion': 'Motion',
    'generalist': 'Generalist',
}
DATASETS = ['BTSD', 'CTSD', 'GTSRB']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_old_noise(transform: str) -> pd.DataFrame:
    """Load train or robust old_evalnoise CSV."""
    pattern = os.path.join(RESULTS, f'noise_results_all_models_{transform}_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No old-noise CSV matching {pattern}')
    return pd.read_csv(files[-1])   # latest timestamp


def load_ensemble_noise() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, 'noise_results_all_ensembles.csv'))


def load_history_files(pattern: str) -> pd.DataFrame:
    """Load history CSVs and extract best-test-f1 row per file."""
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        best = df.loc[df['test f1'].idxmax()].copy()
        # add filename metadata
        fname = os.path.basename(f)
        best['_file'] = fname
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Pipeline Progression (3-phase comparison)
# ═══════════════════════════════════════════════════════════════════════════

def fig_pipeline_comparison():
    """Bar chart comparing clean & corrupted accuracy across 3 preprocessing phases."""

    df_train = load_old_noise('train')
    df_robust = load_old_noise('robust')
    df_ens = load_ensemble_noise()

    archs = ['trialcnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0']
    phases = [
        ('Phase 1\nTrain-Tform', df_train),
        ('Phase 2\nRobust Shield', df_robust),
    ]

    # Compute mean corrupted accuracy per model per phase (averaged over datasets)
    records = {}
    for label, df in phases:
        corr = df[df['Corruption'] != 'none']
        records[label] = corr.groupby('Model')['accuracy'].mean()

    # Phase 3: ensemble_32 and per-arch generalist (same preprocessing as robust)
    ens_corr = df_ens[df_ens['Corruption'] != 'none']
    ens_avg = ens_corr.groupby('Model')['accuracy'].mean()
    # map generalist member (same preprocessing as robust) per arch to phase3-specialist
    generalist = {a: ens_avg.get(f'{a}_generalist', np.nan) for a in archs}
    ensemble32_avg = ens_avg.get('ensemble_32', np.nan)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    n = len(archs)
    x = np.arange(n)
    width = 0.22
    offsets = [-1.1, 0, 1.1]
    phase_colors = ['#94A3B8', '#60A5FA', '#7C3AED']
    phase_labels = ['Phase 1 (train_tform)', 'Phase 2 (robust shield)', 'Phase 3 (specialist, best member)']

    for i, (label, df) in enumerate(phases):
        vals = [records[label].get(a, np.nan) * 100 for a in archs]
        bars = ax.bar(x + offsets[i] * width, vals, width, color=phase_colors[i],
                      label=phase_labels[i], edgecolor='white', linewidth=0.5)

    # Phase 3: use generalist as "same-pipeline" baseline for each arch
    vals3 = [generalist[a] * 100 for a in archs]
    ax.bar(x + offsets[2] * width, vals3, width, color=phase_colors[2],
           label=phase_labels[2], edgecolor='white', linewidth=0.5)

    # Horizontal dotted line for ensemble_32
    ax.axhline(ensemble32_avg * 100, ls='--', color='#7C3AED', lw=1.2,
               label=f'Ensemble-32 (all members, all archs): {ensemble32_avg*100:.1f}%')

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs])
    ax.set_ylabel('Mean Corrupted Accuracy (%)')
    ax.set_ylim(45, 96)
    ax.set_title('Preprocessing Pipeline Progression: Mean Accuracy Under 14 Corruptions')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig_pipeline_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Ensemble vs. Single Models (top-N bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def fig_ensemble_vs_single():
    """Grouped bar chart: top single models vs ensemble_32 per dataset."""

    df = load_ensemble_noise()
    corr = df[df['Corruption'] != 'none']
    avg = corr.groupby(['Model', 'Dataset'])['accuracy'].mean().unstack()
    avg['_avg'] = avg.mean(axis=1)

    # Top 8 single members + ensemble_32
    singles = avg[avg.index != 'ensemble_32'].nlargest(8, '_avg')
    ens = avg.loc[['ensemble_32']]
    top = pd.concat([singles, ens]).sort_values('_avg')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(top))
    h = 0.26
    offsets = [-h, 0, h]
    ds_colors = ['#F87171', '#34D399', '#60A5FA']

    for i, ds in enumerate(DATASETS):
        vals = top[ds].values * 100
        bars = ax.barh(y + offsets[i], vals, h, color=ds_colors[i],
                       label=ds, edgecolor='white', linewidth=0.5)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}', va='center', fontsize=6.5)

    ax.set_yticks(y)
    labels = []
    for m in top.index:
        if m == 'ensemble_32':
            labels.append('[*] Ensemble-32')
        else:
            arch, member = m.rsplit('_', 1)
            labels.append(f'{ARCH_LABEL.get(arch, arch)} / {MEMBER_LABEL.get(member, member)}')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean Corrupted Accuracy (%)')
    ax.set_title('Top Single Specialist Models vs. Ensemble-32\n(averaged over all 14 corruptions & 5 severities)')
    ax.set_xlim(60, 100)
    ax.legend(title='Dataset', loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Vertical line for ensemble avg
    ens_avg = top.loc['ensemble_32', '_avg'] * 100
    ax.axvline(ens_avg, ls='--', color='#7C3AED', lw=1.1)

    fig.tight_layout()
    _save(fig, 'fig_ensemble_vs_single')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Per-Corruption Breakdown for ensemble_32
# ═══════════════════════════════════════════════════════════════════════════

def fig_per_corruption_ensemble():
    """Horizontal bar chart: 14 corruptions for ensemble_32, per dataset."""

    df = load_ensemble_noise()
    corr = df[(df['Corruption'] != 'none') & (df['Model'] == 'ensemble_32')]
    avg = corr.groupby(['Corruption', 'Dataset'])['accuracy'].mean().unstack()
    avg['_avg'] = avg.mean(axis=1)
    avg = avg.sort_values('_avg')

    CORR_PRETTY = {
        'brightness': 'Brightness', 'contrast': 'Contrast',
        'defocus_blur': 'Defocus Blur', 'elastic_transform': 'Elastic Transform',
        'fog': 'Fog', 'frost': 'Frost', 'gaussian_noise': 'Gaussian Noise',
        'impulse_noise': 'Impulse Noise', 'jpeg_compression': 'JPEG Compression',
        'motion_blur': 'Motion Blur', 'pixelate': 'Pixelate',
        'shot_noise': 'Shot Noise', 'snow': 'Snow', 'zoom_blur': 'Zoom Blur',
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(avg))
    h = 0.26
    offsets = [-h, 0, h]
    ds_colors = ['#F87171', '#34D399', '#60A5FA']

    for i, ds in enumerate(DATASETS):
        vals = avg[ds].values * 100
        bars = ax.barh(y + offsets[i], vals, h, color=ds_colors[i],
                       label=ds, edgecolor='white', linewidth=0.5)

    # Overall avg bars (thin line)
    ax.barh(y, avg['_avg'].values * 100, 0.02, color='#1E293B', label='Overall Avg')

    ax.set_yticks(y)
    ax.set_yticklabels([CORR_PRETTY.get(c, c) for c in avg.index])
    ax.set_xlabel('Mean Accuracy (%)')
    ax.set_title('Ensemble-32 Per-Corruption Accuracy\n(averaged over all 5 severity levels)')
    ax.set_xlim(30, 105)
    ax.legend(title='Dataset', loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(avg['_avg'].mean() * 100, ls='--', color='#475569', lw=1,
               label='Grand mean')
    fig.tight_layout()
    _save(fig, 'fig_per_corruption_ensemble')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Specialist Member Heatmap (best test F1 per arch × member)
# ═══════════════════════════════════════════════════════════════════════════

def fig_specialist_heatmap():
    """Heatmap of best-checkpoint test F1 per (architecture × member), for BTSD (hardest dataset)."""

    hist = load_history_files('history_*_ensemble_*.csv')
    archs = ['trialcnn', 'resnet18', 'efficientnet_b0', 'mobilenet_v2']
    members = ['noise', 'light', 'weather', 'fog', 'defocus', 'heavyblur', 'motion', 'generalist']

    # Build matrix: 4 archs x 8 members x 3 datasets
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        sub = hist[hist['dataset'].str.upper() == ds]
        mat = np.full((len(archs), len(members)), np.nan)
        for i, arch in enumerate(archs):
            for j, member in enumerate(members):
                row = sub[(sub['architecture'] == arch) & (sub['member'] == member)]
                if not row.empty:
                    mat[i, j] = row['test f1'].values[0]

        im = ax.imshow(mat, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(members)))
        ax.set_xticklabels([MEMBER_LABEL[m] for m in members], rotation=45, ha='right', fontsize=7)
        ax.set_title(ds, fontweight='bold')
        if ax is axes[0]:
            ax.set_yticks(range(len(archs)))
            ax.set_yticklabels([ARCH_LABEL[a] for a in archs])
        # Annotate
        for i in range(len(archs)):
            for j in range(len(members)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                            color='black' if 0.6 < v < 0.9 else 'white')

    plt.colorbar(im, ax=axes[-1], shrink=0.85, label='Best Test F1')
    fig.suptitle('Specialist Member Best-Checkpoint F1 Score\n(4 architectures × 8 preprocessing variants)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'fig_specialist_heatmap')


# ═══════════════════════════════════════════════════════════════════════════
# LATEX TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v, pct=True):
    if np.isnan(v):
        return '--'
    return f'{v*100:.2f}' if pct else f'{v:.4f}'


def gen_table_phase1_phase2():
    """Table 2: Phase 1 (train_tform) vs Phase 2 (robust_tform) clean accuracy and mean corrupted accuracy."""
    df_t = load_old_noise('train')
    df_r = load_old_noise('robust')
    archs = ['trialcnn', 'resnet18', 'efficientnet_b0', 'mobilenet_v2']
    arch_labels = ['TrialCNN', 'ResNet-18', 'EfficientNet-B0', 'MobileNetV2']

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Phase 1 vs.\ Phase 2 Preprocessing: Best Test F1 on Clean Data and Mean Accuracy Under All Corruptions (averaged over GTSRB, BTSD, CTSD)}')
    lines.append(r'\label{tab:phase1_phase2}')
    lines.append(r'\begin{center}')
    lines.append(r'\setlength\tabcolsep{4pt}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Model} & \multicolumn{2}{c}{\textbf{Phase 1 (train\_tform)}} & \multicolumn{2}{c}{\textbf{Phase 2 (robust shield)}} \\')
    lines.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    lines.append(r' & Clean F1 & Corr.\ Acc. & Clean F1 & Corr.\ Acc. \\')
    lines.append(r'\midrule')

    for arch, label in zip(archs, arch_labels):
        # Phase 1
        clean_t = df_t[(df_t['Model'] == arch) & (df_t['Corruption'] == 'none')]['f1 score'].mean()
        corr_t = df_t[(df_t['Model'] == arch) & (df_t['Corruption'] != 'none')]['accuracy'].mean()
        # Phase 2
        clean_r = df_r[(df_r['Model'] == arch) & (df_r['Corruption'] == 'none')]['f1 score'].mean()
        corr_r = df_r[(df_r['Model'] == arch) & (df_r['Corruption'] != 'none')]['accuracy'].mean()

        delta = (corr_r - corr_t) * 100
        delta_str = f'+{delta:.1f}\\%' if delta >= 0 else f'{delta:.1f}\\%'
        lines.append(
            f'{label} & {clean_t*100:.2f} & {corr_t*100:.2f} & {clean_r*100:.2f} & {corr_r*100:.2f} ({delta_str}) \\\\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{center}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_table_specialist_summary():
    """Table 3: Best-checkpoint test F1 per architecture x dataset (mean over 8 specialist members)."""
    hist = load_history_files('history_*_ensemble_*.csv')
    archs = ['trialcnn', 'resnet18', 'efficientnet_b0', 'mobilenet_v2']
    arch_labels = ['TrialCNN', 'ResNet-18', 'EfficientNet-B0', 'MobileNetV2']

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Specialist Member Best-Checkpoint Test F1 (mean over 8 preprocessing variants). Numbers in parentheses show the best single member.}')
    lines.append(r'\label{tab:specialist_summary}')
    lines.append(r'\begin{center}')
    lines.append(r'\setlength\tabcolsep{4pt}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Architecture} & \textbf{BTSD} & \textbf{CTSD} & \textbf{GTSRB} & \textbf{Avg.} \\')
    lines.append(r'\midrule')

    for arch, label in zip(archs, arch_labels):
        sub = hist[hist['architecture'] == arch]
        vals = {}
        for ds in DATASETS:
            dsub = sub[sub['dataset'].str.upper() == ds]
            mean_f1 = dsub['test f1'].mean()
            best_f1 = dsub['test f1'].max()
            best_member = dsub.loc[dsub['test f1'].idxmax(), 'member'] if not dsub.empty else '?'
            vals[ds] = (mean_f1, best_f1, best_member)
        avg = np.mean([vals[d][0] for d in DATASETS])
        row = f'{label}'
        for ds in DATASETS:
            mv, bv, bm = vals[ds]
            row += f' & {mv*100:.2f} ({bv*100:.2f}\\,\\textit{{{MEMBER_LABEL[bm]}}})'
        row += f' & {avg*100:.2f} \\\\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{center}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_table_ensemble_vs_single():
    """Table 4: Top single models vs ensemble_32 — mean corrupted accuracy."""
    df = load_ensemble_noise()
    corr = df[df['Corruption'] != 'none']
    avg = corr.groupby(['Model', 'Dataset'])['accuracy'].mean().unstack()
    avg['Avg'] = avg.mean(axis=1)

    # Top 10 single models
    singles = avg[avg.index != 'ensemble_32'].nlargest(10, 'Avg')
    ens = avg.loc[['ensemble_32']]
    top = pd.concat([ens, singles]).sort_values('Avg', ascending=False)

    def model_label(m):
        if m == 'ensemble_32':
            return r'\textbf{Ensemble-32} (32 members)'
        arch, member = m.rsplit('_', 1)
        return f'{ARCH_LABEL.get(arch, arch)} / {MEMBER_LABEL.get(member, member)}'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Mean Corrupted Accuracy (\%) of Ensemble-32 vs.\ Top Single Specialist Models. Averaged over all 14 corruption types and 5 severity levels.}')
    lines.append(r'\label{tab:ensemble_vs_single}')
    lines.append(r'\begin{center}')
    lines.append(r'\setlength\tabcolsep{4pt}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Model} & \textbf{BTSD} & \textbf{CTSD} & \textbf{GTSRB} & \textbf{Avg.} \\')
    lines.append(r'\midrule')

    for m in top.index:
        row_data = top.loc[m]
        lab = model_label(m)
        if m == 'ensemble_32':
            lines.append(r'\rowcolor{gray!10}')
        lines.append(
            f'{lab} & {row_data["BTSD"]*100:.2f} & {row_data["CTSD"]*100:.2f} & {row_data["GTSRB"]*100:.2f} & {row_data["Avg"]*100:.2f} \\\\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\multicolumn{5}{l}{\footnotesize Full results: \url{https://github.com/aswadekarb24/Machine_Shikavni}} \\')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{center}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_table_per_corruption():
    """Table 5: ensemble_32 per-corruption mean accuracy avg over datasets."""
    df = load_ensemble_noise()
    corr = df[(df['Corruption'] != 'none') & (df['Model'] == 'ensemble_32')]
    avg = corr.groupby('Corruption').agg({'accuracy': 'mean', 'f1 score': 'mean'}).round(4)
    avg = avg.sort_values('accuracy', ascending=False)

    CORR_PRETTY = {
        'brightness': 'Brightness', 'contrast': 'Contrast',
        'defocus_blur': 'Defocus Blur', 'elastic_transform': 'Elastic Transform',
        'fog': 'Fog', 'frost': 'Frost', 'gaussian_noise': 'Gaussian Noise',
        'impulse_noise': 'Impulse Noise', 'jpeg_compression': 'JPEG Compression',
        'motion_blur': 'Motion Blur', 'pixelate': 'Pixelate',
        'shot_noise': 'Shot Noise', 'snow': 'Snow', 'zoom_blur': 'Zoom Blur',
    }

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Ensemble-32 Mean Accuracy (\%) and Weighted F1 per Corruption Type, averaged over the three datasets and all 5 severity levels.}')
    lines.append(r'\label{tab:per_corruption}')
    lines.append(r'\begin{center}')
    lines.append(r'\setlength\tabcolsep{5pt}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{lcc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Corruption} & \textbf{Accuracy (\%)} & \textbf{Weighted F1} \\')
    lines.append(r'\midrule')
    for corr_name, row in avg.iterrows():
        lines.append(f'{CORR_PRETTY.get(corr_name, corr_name)} & {row["accuracy"]*100:.2f} & {row["f1 score"]:.4f} \\\\')
    lines.append(r'\midrule')
    lines.append(f'\\textbf{{Mean}} & {avg["accuracy"].mean()*100:.2f} & {avg["f1 score"].mean():.4f} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{center}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def gen_table_clean_ensemble():
    """Table: Ensemble-32 clean accuracy per dataset."""
    df = load_ensemble_noise()
    clean = df[(df['Corruption'] == 'none') & (df['Model'] == 'ensemble_32')]
    return clean[['Dataset', 'accuracy', 'f1 score']].to_string(index=False)


# ═══════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=200)
        print(f'Saved: {path}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=== Generating figures ===')
    fig_pipeline_comparison()
    fig_ensemble_vs_single()
    fig_per_corruption_ensemble()
    fig_specialist_heatmap()

    print('\n=== Generating LaTeX table snippets ===')
    snippet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latex_snippets.txt')
    snippets = {
        'TABLE: Phase 1 vs Phase 2': gen_table_phase1_phase2(),
        'TABLE: Specialist Summary': gen_table_specialist_summary(),
        'TABLE: Ensemble vs Single': gen_table_ensemble_vs_single(),
        'TABLE: Per Corruption': gen_table_per_corruption(),
        'INFO: Ensemble Clean Accuracy': gen_table_clean_ensemble(),
    }
    with open(snippet_path, 'w') as f:
        for title, content in snippets.items():
            f.write(f'\n\n% {"="*60}\n% {title}\n% {"="*60}\n\n')
            f.write(content)
            f.write('\n')
    print(f'Saved LaTeX snippets to: {snippet_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
