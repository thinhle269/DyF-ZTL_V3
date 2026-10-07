import os
import sys

ROOT = 'D:/Samia/Round_02'
sys.path.insert(0, ROOT)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import src.preprocessing as prep
import src.models as models

OUT = os.path.join(ROOT, 'Extension', 'E4_firing')
os.makedirs(OUT, exist_ok=True)
CLASSES = ['ddos', 'dos', 'injection', 'normal', 'password', 'scanning', 'xss']
PRETTY = {'ddos': 'DDoS', 'dos': 'DoS', 'injection': 'Injection', 'normal': 'Normal', 'password': 'Password',
          'scanning': 'Scanning', 'xss': 'XSS'}
REP_SEED = 4
DATA = os.path.join(ROOT, 'dataset', 'Train_Test_Windows_10.csv')


def gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    if v.sum() == 0:
        return float('nan')
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def savefig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)


df = pd.read_csv(os.path.join(ROOT, 'results_r02', 'analysis', 'runs_flat.csv'), low_memory=False)
runs = df[(df.exp == 'main') & (df.method == 'DyF-ZTL-V') & (df.seed < 100)].sort_values('seed')
assert len(runs) == 10, len(runs)

rows, per_node, neff_rows, classnode = [], [], [], {}
for _, r in runs.iterrows():
    s = int(r.seed)
    _, val, _, _, D, classes = prep.load_and_process_data(DATA, num_clients=20, partition_seed=s, split='random', verbose=False)
    assert [c.lower() for c in classes] == CLASSES, classes
    sd = torch.load(os.path.join(r['dir'], f'global_{r.run_id}.pt'), map_location='cpu')
    model = models.DynamicFuzzyNet(D, 7)
    model.load_state_dict(sd)
    model.eval()
    Xv, yv = val.tensors
    with torch.no_grad():
        lf = model.fuzzy.log_firing(Xv)
        share = torch.softmax(lf, 1).numpy()
        raw = torch.exp(lf).numpy()
    W = model.consequent.weight.detach().numpy()
    dom = W.argmax(0)
    y = yv.numpy()
    R = share.shape[1]
    top1 = share.argmax(1)
    usage = share.mean(0)
    top1_share = np.bincount(top1, minlength=R) / len(y)
    ent = -(share * np.log(share + 1e-12)).sum(1)
    neff = np.exp(ent)
    active_rel = (share > 0.05).sum(1)
    active_abs = (raw > 1e-3).sum(1)
    cn = np.stack([share[y == k].mean(0) if (y == k).any() else np.zeros(R) for k in range(7)])
    classnode[s] = cn
    pd.DataFrame(cn, index=[PRETTY[c] for c in CLASSES], columns=[f'node{j}' for j in range(R)]).to_csv(
        os.path.join(OUT, f'firing_class_by_node_seed{s}.csv'))
    purity = [float((y[top1 == j] == dom[j]).mean()) if (top1 == j).any() else np.nan for j in range(R)]
    per_node.append(pd.DataFrame(dict(seed=s, node=range(R), dominant_class=[CLASSES[k] for k in dom], usage=usage,
                                      top1_share=top1_share, top1_purity=purity, raw_mean=raw.mean(0),
                                      raw_median=np.median(raw, 0))))
    neff_rows.append(pd.DataFrame(dict(seed=s, true_class=[CLASSES[k] for k in y], n_eff=neff, max_share=share.max(1),
                                       active_share_gt_0p05=active_rel, active_raw_gt_1e3=active_abs)))
    cs = np.sort(usage)[::-1]
    rows.append(dict(seed=s, n_records=len(y), neff_mean=neff.mean(), neff_median=float(np.median(neff)),
                     neff_p10=float(np.percentile(neff, 10)), neff_p90=float(np.percentile(neff, 90)),
                     records_maxshare_gt_0p5_pct=float((share.max(1) > 0.5).mean() * 100),
                     active_share_gt_0p05_mean=float(active_rel.mean()), active_raw_gt_1e3_mean=float(active_abs.mean()),
                     nodes_usage_ge_1pct=int((usage >= 0.01).sum()), nodes_never_top1=int((top1_share == 0).sum()),
                     top5_mass_pct=float(cs[:5].sum() * 100), nodes_for_90pct=int(np.searchsorted(np.cumsum(cs), 0.9) + 1),
                     usage_gini=gini(usage), top1_purity_weighted=float(np.nansum(np.array(purity) * top1_share) / top1_share[~np.isnan(purity)].sum())))
    print(f'seed {s}: n_eff mean {neff.mean():.2f} median {np.median(neff):.2f} | nodes >=1% {int((usage >= 0.01).sum())} | never top-1 {int((top1_share == 0).sum())}')

summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, 'firing_summary.csv'), index=False)
pn = pd.concat(per_node, ignore_index=True)
pn.to_csv(os.path.join(OUT, 'firing_per_node.csv'), index=False)
nr = pd.concat(neff_rows, ignore_index=True)
nr.to_csv(os.path.join(OUT, 'firing_neff_records.csv'), index=False)


cov = pd.read_csv(os.path.join(ROOT, 'results_r02', 'fuzzy_diagnostics', 'rules_main_DyF-ZTL-V.csv'))
chk = pn.merge(cov, left_on=['seed', 'node'], right_on=['seed', 'rule'])
print('cross-check top-1 share vs stored coverage: max |diff| = %.3f points' % (chk.top1_share * 100 - chk.coverage).abs().max())


palette = plt.get_cmap('tab10')
rep = pn[pn.seed == REP_SEED].copy()
order = np.lexsort((rep.node.values, rep.dominant_class.map(CLASSES.index).values))
rep = rep.iloc[order].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 2.9))
x = np.arange(len(rep))
ax.bar(x, rep.usage * 100, color=[palette(CLASSES.index(c)) for c in rep.dominant_class], alpha=0.85, label='mean normalised firing share (usage)')
ax.plot(x, rep.top1_share * 100, 'k_', ms=9, mew=1.5, label='share of records won (top-1)')
ax.set(xticks=x, xticklabels=[str(j) for j in rep.node], ylabel='% of validation records / firing mass')
ax.set_xlabel('rule node (grouped by dominant consequent class)', fontsize=8)
ax.tick_params(axis='x', labelsize=6)
bounds = np.cumsum(rep.dominant_class.value_counts().reindex(CLASSES).fillna(0).astype(int).values)[:-1]
for b in bounds:
    ax.axvline(b - 0.5, color='k', lw=0.5)
for k, c in enumerate(CLASSES):
    g = np.where(rep.dominant_class == c)[0]
    if len(g):
        ax.text(g.mean(), ax.get_ylim()[1] * (1.02 if k % 2 == 0 else 1.14), PRETTY[c], ha='center', va='bottom', fontsize=6.5, color=palette(k))
ax.legend(frameon=False, fontsize=7, loc='upper right')
savefig(fig, 'fig_firing_usage_rep')

cn = classnode[REP_SEED][:, rep.node.values]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
im = ax.imshow(cn * 100, aspect='auto', cmap='viridis', vmin=0, vmax=max(1.0, cn.max() * 100))
ax.set(yticks=range(7), yticklabels=[PRETTY[c] for c in CLASSES], xticks=range(len(rep)), xticklabels=[str(j) for j in rep.node])
ax.set_xlabel('rule node (grouped by dominant consequent class)', fontsize=8)
ax.set_ylabel('true class of the record', fontsize=8)
ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=7.5)
for b in bounds:
    ax.axvline(b - 0.5, color='w', lw=0.6)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label('mean firing share (%)', fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)
savefig(fig, 'fig_firing_class_node_rep')

fig, ax = plt.subplots(figsize=(4.6, 2.8))
for s, g in nr.groupby('seed'):
    v = np.sort(g.n_eff.values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=0.9, color='C0' if s != REP_SEED else 'C3', alpha=0.5 if s != REP_SEED else 1.0,
            label='seed 4 (representative)' if s == REP_SEED else ('other seeds' if s == 0 else None))
ax.set(xlabel='effective number of active rule nodes per record, exp(H)', ylabel='cumulative share of records', xlim=(1, 32))
ax.legend(frameon=False, fontsize=7)
savefig(fig, 'fig_firing_neff')


m, sd = summ.mean(numeric_only=True), summ.std(numeric_only=True)
def ms(c, d=1):
    return f"{m[c]:.{d}f}$\\pm${sd[c]:.{d}f}"
lines = [r'\begin{tableorg}[!htbp]', r'\centering', r'\scriptsize',
         r'\caption{Firing-strength distribution of the final DyF-ZTL (v2.2) global models on the validation split (one row per seed; last row mean $\pm$ SD). $n_{\mathrm{eff}}$ = effective number of active rule nodes per record ($\exp$ of the entropy of the normalised firing shares); ``active'' = nodes with share $>0.05$ (relative) or $\exp(\ell_j)>10^{-3}$ (absolute); usage = mean share of a node; the last columns count nodes with usage $\ge1\%$, nodes that never win a record, the usage mass of the five most used nodes, the number of nodes needed for 90\% of the usage mass, and the Gini coefficient of the usage distribution.}',
         r'\label{tab:firing}', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{adjustbox}{max width=\textwidth}',
         r'\begin{tabular}{lcccccccccc}', r'\toprule',
         r'\textbf{Seed} & \makecell{\bfseries $n_{\mathrm{eff}}$\\mean} & \makecell{\bfseries $n_{\mathrm{eff}}$\\median} & \makecell{\bfseries Records with\\max share $>0.5$ (\%)} & \makecell{\bfseries Active nodes\\(share $>0.05$)} & \makecell{\bfseries Active nodes\\($\exp\ell>10^{-3}$)} & \makecell{\bfseries Nodes with\\usage $\ge1\%$} & \makecell{\bfseries Nodes never\\top-1} & \makecell{\bfseries Top-5 usage\\mass (\%)} & \makecell{\bfseries Nodes for\\90\% mass} & \makecell{\bfseries Gini of\\usage} \\',
         r'\midrule']
for _, r in summ.iterrows():
    lines.append(f"{int(r.seed)} & {r.neff_mean:.2f} & {r.neff_median:.2f} & {r.records_maxshare_gt_0p5_pct:.1f} & {r.active_share_gt_0p05_mean:.2f} & {r.active_raw_gt_1e3_mean:.1f} & {int(r.nodes_usage_ge_1pct)} & {int(r.nodes_never_top1)} & {r.top5_mass_pct:.1f} & {int(r.nodes_for_90pct)} & {r.usage_gini:.2f} \\\\")
lines += [r'\midrule', f"mean$\\pm$SD & {ms('neff_mean', 2)} & {ms('neff_median', 2)} & {ms('records_maxshare_gt_0p5_pct')} & {ms('active_share_gt_0p05_mean', 2)} & {ms('active_raw_gt_1e3_mean')} & {ms('nodes_usage_ge_1pct')} & {ms('nodes_never_top1')} & {ms('top5_mass_pct')} & {ms('nodes_for_90pct')} & {ms('usage_gini', 2)} \\\\",
          r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{tableorg}']
open(os.path.join(OUT, 'tab_firing.tex'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

rep_s = summ[summ.seed == REP_SEED].iloc[0]
top = rep.sort_values('usage', ascending=False).head(3)
para = (r"\paragraph{Firing-strength distribution.} Figures~\ref{fig:firing_usage} and \ref{fig:firing_class_node} and Table~\ref{tab:firing} report how the firing mass is distributed over the 32 rule nodes on the validation split. "
        f"Per record, the normalised firing shares concentrate on few nodes: the effective number of active nodes is {m['neff_mean']:.2f}$\\pm${sd['neff_mean']:.2f} on average over the ten seeds (median {m['neff_median']:.2f}), {m['records_maxshare_gt_0p5_pct']:.0f}\\% of the records give more than half of the mass to a single node, and on average {m['active_share_gt_0p05_mean']:.1f} nodes exceed a share of 0.05 while {m['active_raw_gt_1e3_mean']:.0f} nodes have an absolute firing above $10^{{-3}}$ (the product t-norm makes most nodes fire weakly on every record). "
        f"Across records the usage is also concentrated: {m['nodes_usage_ge_1pct']:.1f}$\\pm${sd['nodes_usage_ge_1pct']:.1f} nodes receive at least 1\\% of the mass, the five most used nodes carry {m['top5_mass_pct']:.0f}\\% of it, {m['nodes_for_90pct']:.1f} nodes cover 90\\%, and {m['nodes_never_top1']:.1f}$\\pm${sd['nodes_never_top1']:.1f} nodes never win a record (Gini {m['usage_gini']:.2f}). "
        f"In the representative model (seed 4) the three most used nodes are node {int(top.iloc[0].node)} ({PRETTY[top.iloc[0].dominant_class]}-dominated, {top.iloc[0].usage * 100:.0f}\\% of the mass), node {int(top.iloc[1].node)} ({PRETTY[top.iloc[1].dominant_class]}, {top.iloc[1].usage * 100:.0f}\\%) and node {int(top.iloc[2].node)} ({PRETTY[top.iloc[2].dominant_class]}, {top.iloc[2].usage * 100:.0f}\\%); the class$\\times$node matrix shows that the \\emph{{Normal}} records spread their mass over several \\emph{{Normal}}-dominated nodes, whereas each attack class loads mainly one or two nodes. "
        r"The winning node is not class-pure (usage-weighted purity " + f"{m['top1_purity_weighted'] * 100:.0f}\\%" + r"), consistent with the inhibitory consequent layer: the class is decided by the combination of the active nodes, not by the single strongest one. The rule nodes that never fire strongly are the ones whose memberships are broad on every counter (Figures~\ref{fig:memb_sysdriver}--\ref{fig:memb_tcpaps})." + "\n")
figs = (r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig/fig_firing_usage_rep.png}
\caption{Usage of the 32 rule nodes of the representative DyF-ZTL (v2.2) model (seed 4) on the validation split: mean normalised firing share per node (bars, coloured by the node's dominant consequent class) and share of records on which the node has the largest firing (markers). Nodes grouped by dominant class.}
\label{fig:firing_usage}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig/fig_firing_class_node_rep.png}
\caption{Mean normalised firing share of every rule node for the validation records of each true class (representative model, seed 4). Each attack class loads one or two nodes; \emph{Normal} records spread over several \emph{Normal}-dominated nodes.}
\label{fig:firing_class_node}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\textwidth]{fig/fig_firing_neff.png}
\caption{Cumulative distribution of the effective number of active rule nodes per validation record, $\exp(H)$ of the normalised firing shares, for the ten final models (seed 4 highlighted).}
\label{fig:firing_neff}
\end{figure}
""")
open(os.path.join(OUT, 'E4_firing_paragraph.tex'), 'w', encoding='utf-8').write(para + figs)
print(summ.round(2).to_string(index=False))
print('done ->', OUT)

