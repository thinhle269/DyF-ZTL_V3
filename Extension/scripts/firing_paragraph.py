import os
import sys

ROOT = 'D:/Samia/Round_02'
sys.path.insert(0, ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import torch

OUT = os.path.join(ROOT, 'Extension', 'E4_firing')
CLASSES = ['ddos', 'dos', 'injection', 'normal', 'password', 'scanning', 'xss']
PRETTY = {'ddos': 'DDoS', 'dos': 'DoS', 'injection': 'Injection', 'normal': 'Normal', 'password': 'Password',
          'scanning': 'Scanning', 'xss': 'XSS'}
REP = 4
summ = pd.read_csv(os.path.join(OUT, 'firing_summary.csv'))
pn = pd.read_csv(os.path.join(OUT, 'firing_per_node.csv'))
cn = pd.read_csv(os.path.join(OUT, f'firing_class_by_node_seed{REP}.csv'), index_col=0)
df = pd.read_csv(os.path.join(ROOT, 'results_r02', 'analysis', 'runs_flat.csv'), low_memory=False)
r = df[(df.exp == 'main') & (df.method == 'DyF-ZTL-V') & (df.seed == REP)].iloc[0]
sd = torch.load(os.path.join(r['dir'], f'global_{r.run_id}.pt'), map_location='cpu')
W = sd['consequent.weight'].numpy()

m, s = summ.mean(numeric_only=True), summ.std(numeric_only=True)
p4 = pn[pn.seed == REP].set_index('node').sort_values('usage', ascending=False)
top = p4.index[:3].tolist()
w_top = W[:, top[0]]
loads = {}
for c in cn.index:
    row = cn.loc[c].sort_values(ascending=False)
    loads[c] = [(int(n.replace('node', '')), float(v)) for n, v in row.items()]
two = {c: sum(v for _, v in loads[c][:2]) * 100 for c in cn.index}
focused = [c for c in cn.index if two[c] >= 55]
spread = [c for c in cn.index if two[c] < 55 and c != 'Normal']


def nodes_txt(c, k=2):
    return ' and '.join(f"node {n} ({v * 100:.0f}\\%)" for n, v in loads[c][:k])


txt = [
    r"\paragraph{Firing-strength distribution.} Figures~\ref{fig:firing_usage}, \ref{fig:firing_class_node} and \ref{fig:firing_neff} and Table~\ref{tab:firing} report how the firing mass is distributed over the 32 rule nodes on the validation split (normalised shares $s_j(x)=\mathrm{softmax}_j(\ell(x))$ of the log-firing values, which the model feeds to the consequent layer as $\exp\ell_j$).",
    f"The mass of a record is shared by several nodes rather than by one: the effective number of active nodes, $\\exp$ of the entropy of the shares, is {m['neff_mean']:.1f}~$\\pm$~{s['neff_mean']:.1f} per record on average over the ten seeds (medians {summ.neff_median.min():.1f}--{summ.neff_median.max():.1f}), only {m['records_maxshare_gt_0p5_pct']:.0f}\\% of the records (range {summ.records_maxshare_gt_0p5_pct.min():.0f}--{summ.records_maxshare_gt_0p5_pct.max():.0f}\\% across seeds) give more than half of their mass to a single node, {m['active_share_gt_0p05_mean']:.1f} nodes exceed a share of 0.05 and {m['active_raw_gt_1e3_mean']:.0f} of the 32 nodes have an absolute firing above $10^{{-3}}$.",
    f"Across records the usage is unequal but not degenerate: {m['nodes_usage_ge_1pct']:.1f}~$\\pm$~{s['nodes_usage_ge_1pct']:.1f} nodes receive at least 1\\% of the total mass, the five most used nodes carry {m['top5_mass_pct']:.0f}\\%, {m['nodes_for_90pct']:.0f} nodes are needed for 90\\% (Gini coefficient {m['usage_gini']:.2f}), and {m['nodes_never_top1']:.0f}~$\\pm$~{s['nodes_never_top1']:.0f} nodes never have the largest share on any record.",
    f"In the representative model (seed 4) the most used node is node {top[0]} ({PRETTY[p4.loc[top[0], 'dominant_class']]}-dominated): it receives {p4.loc[top[0], 'usage'] * 100:.0f}\\% of the mass and has the largest share on {p4.loc[top[0], 'top1_share'] * 100:.0f}\\% of the records, yet only {p4.loc[top[0], 'top1_purity'] * 100:.1f}\\% of the records it wins belong to its dominant class, because all seven of its consequent weights are negative ({w_top.min():.1f} to {w_top.max():.1f}): it is a broad background node whose memberships cover the bulk of every class (Figure~\\ref{{fig:firing_class_node}}, {min(loads[c][0][1] if loads[c][0][0] == top[0] else next((v for n, v in loads[c] if n == top[0]), 0) for c in cn.index) * 100:.0f}--{max(next((v for n, v in loads[c] if n == top[0]), 0) for c in cn.index) * 100:.0f}\\% of the mass of each class) and only subtracts from every logit.",
    f"The class-specific mass sits in narrower nodes: node {top[1]} ({PRETTY[p4.loc[top[1], 'dominant_class']]}, {p4.loc[top[1], 'usage'] * 100:.0f}\\% of the mass, {p4.loc[top[1], 'top1_purity'] * 100:.0f}\\% purity when it wins) carries the \\emph{{Normal}} records, "
    + '; '.join(f"\\emph{{{c}}} records load {nodes_txt(c)}" for c in focused if c != 'Normal') + '. '
    + (f"{', '.join('\\emph{' + c + '}' for c in spread)} spread their mass over more nodes (two nodes carry only {min(two[c] for c in spread):.0f}--{max(two[c] for c in spread):.0f}\\%), and \\emph{{Scanning}} shares its strongest nodes with other classes, which is the firing-level counterpart of its missing excitatory rule." if spread else ''),
    f"The winning node is therefore not a class label: its usage-weighted purity is {m['top1_purity_weighted'] * 100:.0f}\\% on average ({summ.top1_purity_weighted.min() * 100:.0f}--{summ.top1_purity_weighted.max() * 100:.0f}\\% across seeds). Consistent with the inhibitory consequent layer of Figure~\\ref{{fig:consequent}}, a class is decided by the combination of the {m['neff_mean']:.0f}--{m['neff_mean'] + s['neff_mean']:.0f} nodes that fire on a record, and the model is interpretable at the level of memberships and rule firing but not as a list of single winning rules.",
]
para = ' '.join(txt) + '\n'
figs = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig/fig_firing_usage_rep.png}
\caption{Usage of the 32 rule nodes of the representative DyF-ZTL (v2.2) model (seed 4) on the validation split: mean normalised firing share per node (bars, coloured by the node's dominant consequent class) and share of records on which the node has the largest firing (markers). Nodes grouped by dominant class.}
\label{fig:firing_usage}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig/fig_firing_class_node_rep.png}
\caption{Mean normalised firing share of every rule node for the validation records of each true class (representative model, seed 4). The broad background node receives a fifth to a quarter of the mass of every class; the class-specific mass sits in one or two narrower nodes per class.}
\label{fig:firing_class_node}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\textwidth]{fig/fig_firing_neff.png}
\caption{Cumulative distribution of the effective number of active rule nodes per validation record, $\exp(H)$ of the normalised firing shares, for the ten final models (seed 4 highlighted).}
\label{fig:firing_neff}
\end{figure}
"""
open(os.path.join(OUT, 'E4_firing_paragraph.tex'), 'w', encoding='utf-8').write(para + figs)
print(para)
print('node %d weights:' % top[0], dict(zip(CLASSES, W[:, top[0]].round(1))))

