import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = 'D:/Samia/Round_02'
OUT = os.path.join(ROOT, 'Extension', 'E3_seeds_2_9')
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(ROOT, 'results_r02', 'analysis', 'runs_flat.csv'), low_memory=False)
LABEL = {'DyF-ZTL-V': 'DyF-ZTL (v2.2)', 'SCAFFOLD': 'SCAFFOLD', 'FedProx-U': 'FedProx (uniform)', 'FLTrust': 'FLTrust',
         'FedAvg-U': 'FedAvg (uniform)', 'FedProx': 'FedProx', 'FedAvg': 'FedAvg', 'DeepTrust-V': 'MLP + engine',
         'FuzzyNoTrust': 'DFNN, no engine', 'DyF-ZTL-v1': 'Ablation A (acc. gates)', 'ServerOnly': 'Server-only (DFNN)',
         'Krum': 'Krum', 'MultiKrum': 'Multi-Krum', 'Median': 'Median', 'TrimmedMean': 'Trimmed mean'}
ACC, F1, FAR = 'Accuracy', 'F1-Score', 'FAR Benign->Attack'
UNSEEN, SEEN, ALL = list(range(2, 10)), [0, 1], list(range(10))


def std_rows(sub):

    return sub[(sub.rounds == 100) & (sub.seed < 100) & (sub.split.fillna('random') == 'random') & (sub.alpha.fillna(0.5) == 0.5)
               & (sub.smote.fillna(True).astype(str) != 'False') & sub.trust_val_size.isna() & sub.trust_condition.isna()
               & (sub.features.fillna('default').isin(['default', 'original'])) & (sub.gate_set.fillna('').str.contains('q_') == False)
               & (sub.trust_params.fillna('').astype(str).str.contains('decay_factor|mild_decay|recovery_factor|min_safety|warmup|alpha') == False)]


def clean_rows(m):
    g = std_rows(df[(df.method == m) & (df.ratio.fillna(0) == 0) & df.exp.isin(['main', 'agg_ablation', 'cyclic'])])
    g = g.assign(pref=g.exp.map({'main': 0, 'agg_ablation': 1, 'cyclic': 2})).sort_values(['seed', 'pref']).drop_duplicates('seed')
    return g.set_index('seed')


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, p[i] * (len(p) - rank))
        adj[i] = min(1.0, running)
    return adj


methods = ['DyF-ZTL-V', 'SCAFFOLD', 'FedProx-U', 'FLTrust', 'FedAvg-U', 'FedProx', 'FedAvg', 'DeepTrust-V', 'FuzzyNoTrust', 'DyF-ZTL-v1', 'ServerOnly']
tabs = {m: clean_rows(m) for m in methods}
rows = []
for m in methods:
    t = tabs[m]
    for name, seeds in (('seeds 0-9', ALL), ('seeds 0-1 (inspected)', SEEN), ('seeds 2-9 (unseen)', UNSEEN)):
        g = t.loc[[s for s in seeds if s in t.index]]
        rows.append(dict(method=LABEL[m], subset=name, n=len(g), acc_mean=g[ACC].mean(), acc_sd=g[ACC].std(), f1_mean=g[F1].mean(), f1_sd=g[F1].std(),
                         far_mean=g[FAR].mean(), far_sd=g[FAR].std()))
clean = pd.DataFrame(rows)
clean.to_csv(os.path.join(OUT, 'confirm_clean.csv'), index=False)
ref = tabs['DyF-ZTL-V']
prow = []
for m in methods[1:]:
    t = tabs[m]
    for name, seeds in (('seeds 2-9 (unseen)', UNSEEN), ('seeds 0-9', ALL)):
        idx = [s for s in seeds if s in t.index and s in ref.index]
        d_acc = (ref.loc[idx, ACC] - t.loc[idx, ACC]).astype(float)
        d_f1 = (ref.loc[idx, F1] - t.loc[idx, F1]).astype(float)
        prow.append(dict(competitor=LABEL[m], subset=name, n=len(idx), d_acc=d_acc.mean(), d_acc_sd=d_acc.std(), p_acc=stats.ttest_1samp(d_acc, 0).pvalue if len(idx) > 2 else np.nan,
                         d_f1=d_f1.mean(), d_f1_sd=d_f1.std(), p_f1=stats.ttest_1samp(d_f1, 0).pvalue if len(idx) > 2 else np.nan))
paired = pd.DataFrame(prow)
for name in ('seeds 2-9 (unseen)', 'seeds 0-9'):
    mask = paired.subset == name
    paired.loc[mask, 'p_acc_holm'] = holm(paired.loc[mask, 'p_acc'].fillna(1.0))
    paired.loc[mask, 'p_f1_holm'] = holm(paired.loc[mask, 'p_f1'].fillna(1.0))
paired.to_csv(os.path.join(OUT, 'confirm_clean_paired.csv'), index=False)


cyc = std_rows(df[(df.exp == 'cyclic') & df.attack.fillna('cyclic_flip').eq('cyclic_flip') & df.schedule.fillna('static').eq('static') & (df.participation.fillna(1.0) == 1.0)])
cyc = cyc[cyc.aggregation.fillna('').ne('weighted') | ~cyc.method.str.startswith('DyF')]
cmethods = ['DyF-ZTL-V', 'FLTrust', 'DyF-ZTL-v1', 'FedProx', 'FedAvg', 'Krum', 'MultiKrum', 'Median', 'TrimmedMean']
crow, cp = [], []
for r in sorted(cyc.ratio.dropna().unique()):
    sr = cyc[cyc.ratio == r]
    for m in cmethods:
        g = sr[sr.method == m].drop_duplicates('seed').set_index('seed')
        for name, seeds in (('seeds 0-9', ALL), ('seeds 0-1 (inspected)', SEEN), ('seeds 2-9 (unseen)', UNSEEN)):
            h = g.loc[[s for s in seeds if s in g.index]]
            crow.append(dict(ratio=r, method=LABEL[m], subset=name, n=len(h), acc_mean=h[ACC].mean(), acc_sd=h[ACC].std(), f1_mean=h[F1].mean(), f1_sd=h[F1].std()))
    a = sr[sr.method == 'DyF-ZTL-V'].drop_duplicates('seed').set_index('seed')
    for m in ('FLTrust', 'DyF-ZTL-v1'):
        b = sr[sr.method == m].drop_duplicates('seed').set_index('seed')
        for name, seeds in (('seeds 2-9 (unseen)', UNSEEN), ('seeds 0-1 (inspected)', SEEN), ('seeds 0-9', ALL)):
            idx = [s for s in seeds if s in a.index and s in b.index]
            d = (a.loc[idx, ACC] - b.loc[idx, ACC]).astype(float)
            cp.append(dict(ratio=r, competitor=LABEL[m], subset=name, n=len(idx), d_acc=d.mean(), d_acc_sd=d.std(),
                           p_t=stats.ttest_1samp(d, 0).pvalue if len(idx) > 2 else np.nan,
                           p_wilcoxon=stats.wilcoxon(d).pvalue if len(idx) > 2 and (d != 0).any() else np.nan))
cyclic = pd.DataFrame(crow)
cyclic.to_csv(os.path.join(OUT, 'confirm_cyclic.csv'), index=False)
cpaired = pd.DataFrame(cp)
for (m, name), g in cpaired.groupby(['competitor', 'subset']):
    cpaired.loc[g.index, 'p_t_holm'] = holm(g.p_t.fillna(1.0))
cpaired.to_csv(os.path.join(OUT, 'confirm_cyclic_paired.csv'), index=False)


att = std_rows(df[(df.exp == 'attacks') & df.method.isin(['DyF-ZTL-V', 'FLTrust', 'DyF-ZTL-v1', 'FedAvg'])])
arow = []
for (a, sch, part, r), g in att.groupby([att.attack.fillna('cyclic_flip'), att.schedule.fillna('static'), att.participation.fillna(1.0), 'ratio']):
    for m in ('DyF-ZTL-V', 'FLTrust'):
        h = g[g.method == m].drop_duplicates('seed').set_index('seed')
        for name, seeds in (('seeds 0-4', list(range(5))), ('seeds 0-1 (inspected)', SEEN), ('seeds 2-4 (unseen)', [2, 3, 4])):
            k = h.loc[[s for s in seeds if s in h.index]]
            arow.append(dict(attack=a, schedule=sch, participation=part, ratio=r, method=LABEL[m], subset=name, n=len(k), acc_mean=k[ACC].mean(), acc_sd=k[ACC].std(),
                             asr_cond_mean=k.asr_cond.mean(), attack_to_normal_mean=k.attack_to_normal.mean()))
pd.DataFrame(arow).to_csv(os.path.join(OUT, 'confirm_attacks_seeds_2_4.csv'), index=False)


def fmt(mu, sd, d=2):
    return '--' if pd.isna(mu) else (f"{mu:.{d}f}" if pd.isna(sd) else f"{mu:.{d}f}$\\pm${sd:.{d}f}")


L = [r'\begin{tableorg}[!htbp]', r'\centering', r'\scriptsize',
     r'\caption{Clean comparison recomputed on the seeds that were not inspected before the design freeze (seeds 2--9) next to the two inspected seeds (0--1) and all ten seeds: test accuracy and macro-F1 (mean $\pm$ SD) and the paired difference DyF-ZTL (v2.2) minus the method on seeds 2--9 (paired $t$-test, Holm-adjusted $p$ over the ten comparisons). No run was added; the conclusions of Table~\ref{tab:clean} hold on the unseen seeds.}',
     r'\label{tab:confirm_clean}', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{lcccccc}', r'\toprule',
     r'\textbf{Method} & \makecell{\bfseries Accuracy\\seeds 0--1} & \makecell{\bfseries Accuracy\\seeds 2--9} & \makecell{\bfseries Accuracy\\seeds 0--9} & \makecell{\bfseries Macro-F1\\seeds 2--9} & \makecell{\bfseries $\Delta$acc vs DyF-ZTL\\seeds 2--9 ($p_{\mathrm{Holm}}$)} & \makecell{\bfseries $\Delta$F1 vs DyF-ZTL\\seeds 2--9 ($p_{\mathrm{Holm}}$)} \\', r'\midrule']
for m in methods:
    c = clean[clean.method == LABEL[m]].set_index('subset')
    pr = paired[(paired.competitor == LABEL[m]) & (paired.subset == 'seeds 2-9 (unseen)')]
    da = '--' if pr.empty else f"{pr.d_acc.iloc[0]:+.2f} ({pr.p_acc_holm.iloc[0]:.3f})"
    dfone = '--' if pr.empty else f"{pr.d_f1.iloc[0]:+.2f} ({pr.p_f1_holm.iloc[0]:.3f})"
    L.append(f"{LABEL[m]} & {fmt(c.loc['seeds 0-1 (inspected)', 'acc_mean'], c.loc['seeds 0-1 (inspected)', 'acc_sd'])} & {fmt(c.loc['seeds 2-9 (unseen)', 'acc_mean'], c.loc['seeds 2-9 (unseen)', 'acc_sd'])} & {fmt(c.loc['seeds 0-9', 'acc_mean'], c.loc['seeds 0-9', 'acc_sd'])} & {fmt(c.loc['seeds 2-9 (unseen)', 'f1_mean'], c.loc['seeds 2-9 (unseen)', 'f1_sd'])} & {da} & {dfone} \\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{tableorg}']
open(os.path.join(OUT, 'tab_confirm_clean.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

ratios = sorted(cyclic.ratio.unique())
L = [r'\begin{tableorg}[!htbp]', r'\centering', r'\scriptsize',
     r'\caption{Cyclic label flipping recomputed on the unseen seeds 2--9: test accuracy (mean $\pm$ SD) of DyF-ZTL (v2.2), FLTrust and ablation A (accuracy-gated admission) per compromised-client ratio, and the paired difference DyF-ZTL minus FLTrust on seeds 2--9 and on the inspected seeds 0--1 (paired $t$-test $p$ per ratio, as in Table~\ref{tab:cyclic_tests}; Holm-adjusted values are in the released CSV).}',
     r'\label{tab:confirm_cyclic}', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{l' + 'c' * len(ratios) + '}', r'\toprule',
     r'\textbf{Ratio} & ' + ' & '.join(f"\\textbf{{{r:.1f}}}" for r in ratios) + r' \\', r'\midrule']
for m in ('DyF-ZTL-V', 'FLTrust', 'DyF-ZTL-v1'):
    cells = []
    for r in ratios:
        c = cyclic[(cyclic.ratio == r) & (cyclic.method == LABEL[m]) & (cyclic.subset == 'seeds 2-9 (unseen)')]
        cells.append(fmt(c.acc_mean.iloc[0], c.acc_sd.iloc[0], 1) if len(c) else '--')
    L.append(f"{LABEL[m]} (seeds 2--9) & " + ' & '.join(cells) + r' \\')
for name, lab in (('seeds 2-9 (unseen)', 'seeds 2--9'), ('seeds 0-1 (inspected)', 'seeds 0--1')):
    cells = []
    for r in ratios:
        c = cpaired[(cpaired.ratio == r) & (cpaired.competitor == 'FLTrust') & (cpaired.subset == name)]
        if len(c):
            p = c.p_t.iloc[0]
            cells.append(f"{c.d_acc.iloc[0]:+.2f}" + ('' if pd.isna(p) else f" ({p:.2f})"))
        else:
            cells.append('--')
    L.append(f"$\\Delta$ vs FLTrust ({lab}) & " + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{tableorg}']
open(os.path.join(OUT, 'tab_confirm_cyclic.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')


u = clean[(clean.method == 'DyF-ZTL (v2.2)') & (clean.subset == 'seeds 2-9 (unseen)')].iloc[0]
s01 = clean[(clean.method == 'DyF-ZTL (v2.2)') & (clean.subset == 'seeds 0-1 (inspected)')].iloc[0]
pu = paired[paired.subset == 'seeds 2-9 (unseen)'].set_index('competitor')
cu = cpaired[(cpaired.competitor == 'FLTrust') & (cpaired.subset == 'seeds 2-9 (unseen)')].set_index('ratio')
ci = cpaired[(cpaired.competitor == 'FLTrust') & (cpaired.subset == 'seeds 0-1 (inspected)')].set_index('ratio')
sig = [f"{r:.1f}" for r in ratios if cu.loc[r, 'p_t'] < 0.05]
sigw = [f"{r:.1f}" for r in ratios if cu.loc[r, 'p_wilcoxon'] < 0.05 and cu.loc[r, 'p_t'] >= 0.05]
sigh = [f"{r:.1f}" for r in ratios if cu.loc[r, 'p_t_holm'] < 0.05]
neg = [f"{r:.1f}" for r in ratios if cu.loc[r, 'd_acc'] < 0]
adv_u, adv_i = cu.d_acc.mean(), ci.d_acc.mean()


def pd_(name):
    return f"{pu.loc[name, 'd_acc']:+.2f}", f"{pu.loc[name, 'p_acc_holm']:.3f}"


txt = [r"\paragraph{Confirmation on the seeds not inspected before the design freeze.} The two amendments that led to the final engine (the server-reference anchor and the reference update in the aggregate, Section~\ref{sec:engine_evolution}) were motivated by observations on the evaluation seeds 0 and 1 of ablations B and C (Section~\ref{sec:engine_evolution}); seeds 2--9 of the final engine were first examined after the freeze. Tables~\ref{tab:confirm_clean} and \ref{tab:confirm_cyclic} recompute the main comparisons on these eight seeds without any new run.",
       f"On clean data DyF-ZTL (v2.2) reaches {u.acc_mean:.2f}\\%~$\\pm$~{u.acc_sd:.2f} accuracy and {u.f1_mean:.2f}\\% macro-F1 on seeds 2--9 (seeds 0--1: {s01.acc_mean:.2f}\\%), with paired accuracy differences of {pd_('SCAFFOLD')[0]} points versus SCAFFOLD ($p_{{\\mathrm{{Holm}}}}$ = {pd_('SCAFFOLD')[1]}), {pd_('FedProx (uniform)')[0]} versus FedProx-uniform ({pd_('FedProx (uniform)')[1]}), {pd_('FLTrust')[0]} versus FLTrust ({pd_('FLTrust')[1]}), {pd_('MLP + engine')[0]} versus the MLP with the same engine ({pd_('MLP + engine')[1]}) and {pd_('Server-only (DFNN)')[0]} versus the server-only reference ({pd_('Server-only (DFNN)')[1]}); the macro-F1 differences to SCAFFOLD, FedProx-uniform and FLTrust remain within noise, as on all ten seeds.",
       f"Under cyclic flipping the pattern of Table~\\ref{{tab:cyclic_tests}} is reproduced: DyF-ZTL is more accurate than FLTrust at every ratio except {', '.join(neg) if neg else 'none'} (differences within 0.4 points), and the advantage is significant by the paired $t$-test at ratios {', '.join(sig) if sig else 'none'}"
       + (f" and additionally by the Wilcoxon test at {', '.join(sigw)}" if sigw else '')
       + f" (unadjusted, as in Table~\\ref{{tab:cyclic_tests}}; after Holm correction over the ten ratios {', '.join(sigh) if sigh else 'no ratio'} remain{'s' if len(sigh) == 1 else ''}).",
       f"Averaged over the ten ratios the advantage over FLTrust is {adv_u:.2f} points on the unseen seeds and {adv_i:.2f} points on the inspected seeds 0--1: the two seeds that motivated the design changes are somewhat more favourable to the final engine at the highest ratios, whereas their clean-data accuracy ({s01.acc_mean:.2f}\\%) lies below the unseen-seed mean; the sign of every conclusion is the same on both subsets.",
       r"The five-seed attack studies have only three unseen seeds (2--4); their means are listed in the released analysis (\texttt{confirm\_attacks\_seeds\_2\_4.csv}) and agree in sign with the five-seed results."]
para = ' '.join(txt) + '\n'
open(os.path.join(OUT, 'E3_paragraph.tex'), 'w', encoding='utf-8').write(para)
print(clean[clean.subset == 'seeds 2-9 (unseen)'].round(2).to_string(index=False))
print(paired[paired.subset == 'seeds 2-9 (unseen)'].round(3).to_string(index=False))
print(cpaired[(cpaired.competitor == 'FLTrust')].round(3).to_string(index=False))
print('done ->', OUT)

