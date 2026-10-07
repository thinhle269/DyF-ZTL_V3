import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
RES = os.path.join(ROOT, 'results_r02')
OUT = os.path.join(RES, 'analysis', 'reviewer')
os.makedirs(OUT, exist_ok=True)


def to_md(df, index=False):
    f = df.reset_index() if index else df
    cols = list(map(str, f.columns))
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]

    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ''
            return f"{v:.4f}" if c.startswith('p_') else f"{v:.2f}"
        return str(v)
    for row in f.itertuples(index=False):
        lines.append('| ' + ' | '.join(cell(c, v) for c, v in zip(cols, row)) + ' |')
    return '\n'.join(lines)


def r1_1():

    rows, per_client = [], []
    for seed in range(10):
        p = os.path.join(RES, 'partitions', f'random_a0.5_smote_s{seed}.json')
        if not os.path.exists(p):
            continue
        part = json.load(open(p))
        k_norm = part['classes'].index('normal')
        pre = np.array([c['n_original'] for c in part['clients']], float)
        post = np.array([c['n_post'] for c in part['clients']], float)
        share_n = np.array([c['hist_pre'][k_norm] / max(1, sum(c['hist_pre'])) for c in part['clients']])
        nd = share_n >= 0.8
        status = pd.Series([c['smote_status'] for c in part['clients']]).value_counts()
        rows.append({'seed': seed, 'clients': len(pre),
                     'pre-SMOTE min': int(pre.min()), 'pre median': int(np.median(pre)), 'pre max': int(pre.max()),
                     'post-SMOTE min': int(post.min()), 'post median': int(np.median(post)), 'post max': int(post.max()),
                     'largest weight, pre (%)': 100 * pre.max() / pre.sum(),
                     'largest weight, post (%)': 100 * post.max() / post.sum(),
                     'clients >=80% normal': int(nd.sum()),
                     'their weight, pre (%)': 100 * pre[nd].sum() / pre.sum(),
                     'their weight, uniform (%)': 100 * nd.mean(),
                     'SMOTE applied/partial/skipped': f"{status.get('applied', 0)}/{status.get('partial', 0)}/{status.get('skipped', 0)}"})
        for i, c in enumerate(part['clients']):
            per_client.append(dict(seed=seed, client=i, n_pre=c['n_original'], n_post=c['n_post'],
                                   normal_share=round(share_n[i] * 100, 1), smote=c['smote_status'],
                                   hist_pre=' '.join(map(str, c['hist_pre'])), hist_post=' '.join(map(str, c['hist_post']))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, 'R1.1_client_sizes_by_seed.csv'), index=False)
    pd.DataFrame(per_client).to_csv(os.path.join(OUT, 'R1.1_client_sizes_per_client.csv'), index=False)
    agg = os.path.join(RES, 'analysis', 'report_clean_aggregation.csv')
    md = ["# R1.1 — Client sample counts before/after SMOTE and aggregation weights\n",
          "Random split, Dirichlet α = 0.5, K = 20 clients, seeds 0–9 (the clean-comparison partitions). "
          "Pre-SMOTE counts are the |D_k| used by the canonical Eq. 1 weights. Classes: "
          "ddos, dos, injection, normal, password, scanning, xss.\n", to_md(t),
          f"\n**Range over seeds:** largest client weight under pre-SMOTE |D_k| weighting "
          f"{t['largest weight, pre (%)'].min():.1f}–{t['largest weight, pre (%)'].max():.1f}% "
          f"(uniform: 5.0%); clients with ≥ 80% normal records hold "
          f"{t['their weight, pre (%)'].min():.1f}–{t['their weight, pre (%)'].max():.1f}% of the |D_k| weight "
          f"versus {t['their weight, uniform (%)'].min():.0f}–{t['their weight, uniform (%)'].max():.0f}% under uniform averaging.\n",
          "Per-client counts and class histograms (pre/post SMOTE): `R1.1_client_sizes_per_client.csv`.\n"]
    if os.path.exists(agg):
        a = pd.read_csv(agg)
        a = a[['pair', 'metric', 'n', 'diff', 'ci_low', 'ci_high', 'p_t']].round(4)
        md += ["## Aggregation ablation (uniform − |D_k|-weighted, paired by seed)\n", to_md(a), "\n"]
    open(os.path.join(OUT, 'R1.1.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


ENGINES = {'DyF-ZTL-V': 'DyF-ZTL (v2.2)', 'DyF-ZTL-R': 'DyF-ZTL v2.1 (earlier)', 'DyF-ZTL': 'DyF-ZTL v2 (earlier)',
           'DyF-ZTL-v1': 'Round-1 engine (v1)', 'DeepTrust-V': 'DeepTrust (MLP + v2.2)', 'ServerOnly': 'server-only reference'}


def _episodes(h):

    out, start = [], None
    for _, row in h.iterrows():
        if row.Admitted == 0 and start is None:
            start = row.Round
        elif row.Admitted == 1 and start is not None:
            out.append(row.Round - start)
            start = None
    if start is not None:
        out.append(np.nan)
    return out


def r1_2():

    import analyze_r02 as A
    df, _ = A.load(RES)
    rows, reasons = [], []
    for m, label in ENGINES.items():
        g = df[(df.exp == 'main') & (df.method == m)]
        for _, r in g.iterrows():
            p = os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            if 'Phase' in d.columns:
                d = d[d.Phase != 'grace']
            cfg = json.load(open(os.path.join(r['dir'], f"final_{r['run_id']}.json")))['config']
            part = A._partition(RES, cfg)
            share = None
            if part:
                k = part['classes'].index('normal')
                share = [np.asarray(c['hist_pre'], float)[k] / max(1.0, float(np.sum(c['hist_pre']))) for c in part['clients']]
            eps = []
            for c, h in d.groupby('Client'):
                eps += _episodes(h.sort_values('Round'))
            ex = d[d.Admitted == 0]
            nd_share = (100 * ex.Client.map(lambda i: share[int(i)] >= 0.8).mean()) if (share and len(ex)) else np.nan
            rows.append(dict(engine=label, seed=r['seed'], honest_rejection=100 * (1 - d.Admitted.mean()),
                             episodes=len(eps), rehab_rounds=np.nanmean(eps) if eps and not all(np.isnan(eps)) else np.nan,
                             never_back=int(np.sum(np.isnan(eps))) if eps else 0, excluded_from_normal_dominated=nd_share,
                             accuracy=r['Accuracy']))
            if 'Evidence' in d.columns:
                ev = d[d.Evidence.isin(['mild', 'strong'])]
                for reason, n in ev.Reason.fillna('').value_counts().items():
                    reasons.append(dict(engine=label, reason=reason, n=n))
    t = pd.DataFrame(rows)
    md = ["# R1.2 — Honest clients excluded without any attacker (clean runs, 10 seeds)\n",
          "Every client is honest in these runs, so every exclusion is a *false* rejection. Exclusion removes an "
          "update from one round's aggregate; it is not a verdict on the client (revocation needs persistent strong "
          "evidence; rehabilitation is automatic).\n"]
    if len(t):
        s = t.groupby('engine').agg(seeds=('seed', 'count'), honest_rejection_pct=('honest_rejection', 'mean'),
                                    sd=('honest_rejection', 'std'), episodes_per_run=('episodes', 'mean'),
                                    rounds_to_rehabilitation=('rehab_rounds', 'mean'),
                                    episodes_never_rehabilitated=('never_back', 'mean'),
                                    pct_from_normal_dominated_clients=('excluded_from_normal_dominated', 'mean'),
                                    accuracy=('accuracy', 'mean'))
        s = s.reindex([v for v in ENGINES.values() if v in s.index])
        s.to_csv(os.path.join(OUT, 'R1.2_honest_exclusion_clean.csv'))
        md += [to_md(s, index=True), "\n"]
    if reasons:
        rr = pd.DataFrame(reasons).groupby(['engine', 'reason']).n.sum().reset_index()
        rr['share_%'] = 100 * rr.n / rr.groupby('engine').n.transform('sum')
        rr = rr.sort_values(['engine', 'n'], ascending=[True, False])
        rr.to_csv(os.path.join(OUT, 'R1.2_evidence_reasons_clean.csv'), index=False)
        md += ["## Which evidence lowered an honest client's trust (mild/strong events, clean runs)\n", to_md(rr), "\n"]

    arows = []
    cy = df[df.exp == 'cyclic']
    for m, label in list(ENGINES.items()) + [('FLTrust', 'FLTrust (zero weight)')]:
        for _, r in cy[(cy.method == m) & (cy.ratio > 0)].iterrows():
            p = os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")
            if os.path.exists(p):
                am = A.admission_metrics(p)
                if am:
                    arows.append(dict(engine=label, ratio=r['ratio'], seed=r['seed'],
                                      malicious_rejected=100 * am['malicious_rejection'],
                                      honest_rejected=100 * am['honest_rejection'],
                                      precision_of_reject=100 * am['precision_reject'] if am['precision_reject'] == am['precision_reject'] else np.nan))
    if arows:
        a = pd.DataFrame(arows).groupby(['engine', 'ratio']).agg(seeds=('seed', 'count'), malicious_rejected=('malicious_rejected', 'mean'),
                                                                 honest_rejected=('honest_rejected', 'mean'),
                                                                 precision_of_reject=('precision_of_reject', 'mean')).reset_index()
        a.to_csv(os.path.join(OUT, 'R1.2_admission_under_cyclic.csv'), index=False)
        md += ["## Under cyclic label flipping: who is rejected (mean over available seeds)\n", to_md(a), "\n"]
    open(os.path.join(OUT, 'R1.2.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


MAIN, MAIN_LABEL = 'DyF-ZTL-V', 'DyF-ZTL (v2.2)'


def _load():
    import analyze_r02 as A
    df, pc = A.load(RES)
    try:
        sc = json.load(open(os.path.join(RES, 'calibration', 'scaffold.json')))
        for frame in (df, pc):
            m = frame.exp.isin(['main', 'heterogeneity']) & (frame.method == sc['method'])
            frame.loc[m, 'method'] = 'SCAFFOLD'
    except (OSError, ValueError, KeyError):
        pass
    for c in ('schedule', 'participation', 'attack'):
        if c in df.columns:
            df[c] = df[c].astype(object)
    df['schedule'] = df['schedule'].where(df['schedule'].notna(), 'static')
    df['participation'] = df['participation'].where(df['participation'].notna(), 1.0)
    df['attack'] = df['attack'].where(df['attack'].notna(), 'cyclic_flip')
    return A, df, pc


def _msd(s):
    s = s.dropna()
    return '' if not len(s) else (f"{s.mean():.2f} ± {s.std(ddof=1):.2f}" if len(s) > 1 else f"{s.iloc[0]:.2f}")


def _paired_row(A, a, b, metric, label):
    r = A.paired(a[metric], b[metric]) if len(a) and len(b) else None
    if not r:
        return None
    return dict(comparison=label, metric=metric, n=r['n'], diff=r['diff'], ci_low=r['ci_low'], ci_high=r['ci_high'],
                p_t=r['p_t'], p_w=r['p_w'])


def r1_3():

    A, df, _ = _load()
    md = ["# R1.3 / R2.3 — Robustness (100 rounds; mean ± SD over seeds, n per cell)\n"]
    cy = pd.concat([df[df.exp == 'cyclic'], df[(df.exp == 'agg_v21') & (df.ratio > 0)]])
    if len(cy):
        for metric, name in (('Accuracy', 'accuracy'), ('F1-Score', 'macro-F1')):
            t = cy.groupby(['method', 'ratio'])[metric].apply(_msd).unstack('ratio')
            n = cy.groupby(['method', 'ratio']).seed.count().unstack('ratio')
            t.to_csv(os.path.join(OUT, f'R1.3_cyclic_{name}.csv'))
            n.to_csv(os.path.join(OUT, 'R1.3_cyclic_n.csv'))
            md += [f"## Cyclic label flipping — {name}\n", to_md(t, index=True), "\n"]
        tests = []
        for ratio, g in cy.groupby('ratio'):
            ref = g[g.method == MAIN].set_index('seed')
            for o in ('FLTrust', 'DyF-ZTL-v1', 'Median', 'TrimmedMean', 'MultiKrum', 'Krum', 'FedAvg', 'FedProx'):
                row = _paired_row(A, ref, g[g.method == o].set_index('seed'), 'Accuracy', f"{MAIN_LABEL} − {o}")
                if row:
                    tests.append(dict(ratio=ratio, **row))
        if tests:
            tt = pd.DataFrame(tests)
            tt.to_csv(os.path.join(OUT, 'R1.3_cyclic_tests.csv'), index=False)
            md += ["## Paired accuracy differences per ratio\n", to_md(tt), "\n"]
    at = df[df.exp == 'attacks']
    if len(at):
        keys = ['attack', 'schedule', 'participation', 'ratio', 'method']
        t = at.groupby(keys).agg(n=('seed', 'count'), accuracy=('Accuracy', _msd), macro_f1=('F1-Score', _msd),
                                 asr_conditional=('asr_cond', _msd), attack_to_normal=('attack_to_normal', _msd))
        t.to_csv(os.path.join(OUT, 'R1.3_attacks.csv'))
        md += ["## Targeted flip, backdoor, boosted backdoor (model replacement), adaptive, schedules, partial participation\n",
               to_md(t, index=True), "\n"]
    adm = []
    for exp in ('cyclic', 'attacks'):
        for _, r in df[(df.exp == exp) & (df.method == MAIN) & (df.ratio > 0)].iterrows():
            p = os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")
            if os.path.exists(p):
                m = A.admission_metrics(p)
                if m:
                    adm.append(dict(exp=exp, attack=r['attack'], schedule=r['schedule'], participation=r['participation'],
                                    ratio=r['ratio'], seed=r['seed'], **{k: m[k] for k in (
                                        'malicious_rejection', 'honest_rejection', 'precision_reject', 'recall_reject',
                                        'mean_admitted', 'empty_rounds', 'time_to_exclusion', 'time_to_rehabilitation')}))
    if adm:
        a = pd.DataFrame(adm)
        a.to_csv(os.path.join(OUT, 'R1.3_R1.4_admission_per_run.csv'), index=False)
        s = a.groupby(['exp', 'attack', 'schedule', 'participation', 'ratio']).mean(numeric_only=True).drop(columns=['seed'])
        md += [f"## {MAIN_LABEL}: admission decisions (mean over seeds; per-run file R1.3_R1.4_admission_per_run.csv)\n",
               to_md(s.round(3), index=True), "\n"]
    open(os.path.join(OUT, 'R1.3.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md[:6]))


def r1_4():

    A, df, _ = _load()
    md = ["# R1.4 — Trusted corpus and admission modes\n"]
    gates = []
    for p in sorted(glob.glob(os.path.join(RES, 'calibration', 'gates_*-V-R-U.json'))):
        g = json.load(open(p))
        gates.append(dict(condition=g['condition'], complete=g.get('complete'), n_client_rounds=g.get('n_client_rounds'),
                          **{k: round(g[k], 4) for k in ('rho_s', 'rho_m', 'c_s', 'c_m', 'm_s')}))
    if gates:
        gt = pd.DataFrame(gates)
        gt.to_csv(os.path.join(OUT, 'R1.4_gates_v21.csv'), index=False)
        md += ["## Gates recalibrated per trusted-set condition (v2.2 gate sets `<condition>-V-R-U`; quantiles of honest evidence, seeds 100–102)\n", to_md(gt), "\n"]
    ts = df[df.exp == 'trusted_set'].copy()
    if len(ts):
        ts['condition'] = ts.trust_condition.fillna('') + ts.trust_val_size.map(lambda x: f"size{int(x)}" if x == x and x else '')
        rows = []
        for (cond, ratio, m), g in ts.groupby(['condition', 'ratio', 'method']):
            hon = mal = np.nan
            vals = [A.admission_metrics(os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")) for _, r in g.iterrows()
                    if os.path.exists(os.path.join(r['dir'], f"decisions_{r['run_id']}.csv"))]
            vals = [v for v in vals if v]
            if vals:
                hon = 100 * np.nanmean([v['honest_rejection'] for v in vals])
                mal = 100 * np.nanmean([v['malicious_rejection'] for v in vals])
            rows.append(dict(condition=cond, ratio=ratio, method=m, n=len(g), accuracy=_msd(g.Accuracy),
                             macro_f1=_msd(g['F1-Score']), honest_rejected=hon, malicious_rejected=mal))
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(OUT, 'R1.4_trusted_set.csv'), index=False)
        md += ["## Trusted-set conditions (DyF-ZTL-V = recalibrated gates; DyF-ZTL-V-dg = default gates)\n", to_md(t), "\n"]
    tm = pd.concat([df[df.exp == 'trust_modes'], df[(df.exp == 'cyclic') & (df.method == MAIN) & df.ratio.isin([0.2, 0.4, 0.6, 0.8])]])
    if len(tm):
        rows = []
        for (m, ratio), g in tm.groupby(['method', 'ratio']):
            vals = [A.admission_metrics(os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")) for _, r in g.iterrows()
                    if os.path.exists(os.path.join(r['dir'], f"decisions_{r['run_id']}.csv"))]
            vals = [v for v in vals if v]
            f = lambda k: 100 * np.nanmean([v[k] for v in vals]) if vals else np.nan
            rows.append(dict(mode={MAIN: 'hard', MAIN + '-soft': 'soft (trust-weighted)', MAIN + '-prob': 'probation'}.get(m, m),
                             ratio=ratio, n=len(g), accuracy=_msd(g.Accuracy), honest_rejected=f('honest_rejection'),
                             malicious_rejected=f('malicious_rejection'),
                             attacker_weight_share=f('malicious_weight_share'),
                             rehabilitation_rounds=np.nanmean([v['time_to_rehabilitation'] for v in vals]) if vals else np.nan))
        t = pd.DataFrame(rows).sort_values(['ratio', 'mode'])
        t.to_csv(os.path.join(OUT, 'R1.4_modes.csv'), index=False)
        md += ["## Hard exclusion vs trust-weighted (soft) vs probation (integrity–participation trade-off)\n", to_md(t), "\n"]
    open(os.path.join(OUT, 'R1.4.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md[:4]))


def _js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def r1_5():

    A, df, _ = _load()
    md = ["# R1.5 — Data heterogeneity, splits and temporal drift\n"]
    het = []
    for p in sorted(glob.glob(os.path.join(RES, 'partitions', 'random_a*_s[0-4].json'))):
        part = json.load(open(p))
        H = np.array([c['hist_pre'] for c in part['clients']], float)
        glob_h = H.sum(0)
        het.append(dict(alpha='IID' if part['alpha'] is None else part['alpha'], smote=part['smote'], seed=part['seed'],
                        mean_JS_to_global=np.mean([_js(h, glob_h) for h in H]),
                        mean_classes_per_client=np.mean((H > 0).sum(1)),
                        size_cv=H.sum(1).std() / H.sum(1).mean()))
    if het:
        h = pd.DataFrame(het).groupby(['alpha', 'smote']).mean(numeric_only=True).drop(columns=['seed']).round(3)
        h.to_csv(os.path.join(OUT, 'R1.5_heterogeneity_measures.csv'))
        md += ["## Heterogeneity of the client partitions (label skew; mean over seeds)\n",
               "JS = Jensen–Shannon divergence (bits) between a client's class distribution and the global one.\n",
               to_md(h, index=True), "\n"]
    he = df[df.exp.isin(['heterogeneity', 'scaffold_check'])].copy()
    if len(he):
        he = he[~((he.exp == 'scaffold_check') & he.alpha.eq(0.5) & he.smote.eq(True))]
        he['condition'] = [f"α={'IID' if a is None or a != a else a}, {'SMOTE' if s else 'no SMOTE'}"
                           + (f", {l}" if l and l != 'ce' else '') for a, s, l in zip(he.alpha, he.smote, he.loss)]
        t = he.groupby(['condition', 'method']).agg(n=('seed', 'count'), accuracy=('Accuracy', _msd),
                                                    macro_f1=('F1-Score', _msd)).reset_index()
        t.to_csv(os.path.join(OUT, 'R1.5_heterogeneity_results.csv'), index=False)
        md += ["## Results across heterogeneity levels\n", to_md(t), "\n"]
    ch = df[df.exp.isin(['chrono', 'drift'])].copy()
    if len(ch):
        ch['features'] = ch.get('features', pd.Series(index=ch.index, dtype=object)).fillna('original') \
            if 'features' in ch.columns else 'original'
        t = ch.groupby(['split', 'features', 'method']).agg(n=('seed', 'count'), accuracy=('Accuracy', _msd),
                                                           macro_f1=('F1-Score', _msd),
                                                           FAR_normal_to_attack=('FAR Benign->Attack', _msd)).reset_index()
        t.to_csv(os.path.join(OUT, 'R1.5_chrono_drift.csv'), index=False)
        md += ["## Chronological split and drift-robust feature sets (§6c.16)\n", to_md(t), "\n"]
    for f in ('drift_audit.md', 'split_report.md'):
        if os.path.exists(os.path.join(RES, 'audit', f)):
            md.append(f"See also `results_r02/audit/{f}`.")
    open(os.path.join(OUT, 'R1.5.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md[:4]))


def r1_9():

    A, df, pc = _load()
    p = pc[pc.exp.isin(['main', 'agg_ablation', 'scaffold_check']) & pc.method.isin(
        [MAIN, 'DyF-ZTL-R', 'SCAFFOLD', 'FedProx-U', 'FedProx', 'FedAvg', 'FedAvg-U', 'DeepTrust-V', 'FuzzyNoTrust', 'DyF-ZTL-v1', 'ServerOnly'])]
    md = ["# R1.9 — Per-class results across seeds (clean, test split, evaluated once after round 100)\n"]
    if len(p):
        t = p.groupby(['Class', 'method']).agg(n=('seed', 'count'), precision=('Precision', _msd), recall=('Recall', _msd),
                                               f1=('F1', _msd)).reset_index()
        t.to_csv(os.path.join(OUT, 'R1.9_per_class.csv'), index=False)
        md += [to_md(t), "\n"]
    ref = df[(df.exp == 'main') & (df.method == MAIN)].set_index('seed')['F1-Score'].sort_values()
    if len(ref) >= 3:
        seed = int(ref.index[(len(ref) - 1) // 2])
        md.append(f"Representative run for the confusion matrix: seed **{seed}** (median macro-F1 of {MAIN_LABEL}, "
                  "pre-registered neutral rule).\n")
    open(os.path.join(OUT, 'R1.9.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md[:3]))


def r1_4b():

    A, df, _ = _load()
    sub = df[((df.exp == 'server_only_sizes') | ((df.exp == 'main') & df.method.isin(['ServerOnly', 'ServerOnly-MLP', MAIN])))
             & (df.ratio == 0)].copy()
    if not len(sub):
        return
    sub['trust_size'] = sub.trust_val_size.fillna(5280).astype(int)
    t = sub.groupby(['trust_size', 'method']).agg(n=('seed', 'count'), accuracy=('Accuracy', _msd), macro_f1=('F1-Score', _msd),
                                                  FAR_normal_to_attack=('FAR Benign->Attack', _msd)).reset_index()
    t.to_csv(os.path.join(OUT, 'R1.4b_server_only_crossover.csv'), index=False)
    diff = []
    for s_, g in sub.groupby('trust_size'):
        a = g[g.method == MAIN].set_index('seed')
        b = g[g.method == 'ServerOnly'].set_index('seed')
        for metric in ('Accuracy', 'F1-Score'):
            row = _paired_row(A, a, b, metric, f"{MAIN_LABEL} - server-only (|D_trust| = {s_})")
            if row:
                diff.append(row)
    md = ["# R1.4b - Federation vs server-only across trusted-set sizes (clean; protocol 6c.24)\n",
          "`ServerOnly` trains the same DFNN on D_trust alone for the same 100 rounds x 5 epochs (no client). "
          f"{MAIN_LABEL} at smaller |D_trust| uses the gates calibrated for that size.\n", to_md(t), "\n"]
    if diff:
        md += ["## Paired differences (common seeds)\n", to_md(pd.DataFrame(diff)), "\n"]
    open(os.path.join(OUT, 'R1.4b.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md[:3]))


def r1_8():

    def key(c):
        return json.dumps({k: v for k, v in c.items() if k != 'exp'}, sort_keys=True, default=str)
    runs = {}
    for p in glob.glob(os.path.join(RES, '*', 'final_*.json')):
        if os.path.basename(os.path.dirname(p)).startswith('smoke'):
            continue
        r = json.load(open(p))
        runs.setdefault(key(r['config']), []).append((r['config']['exp'], r['config']['method'], r['config']['seed'],
                                                      r['test'], r.get('code_hash'), r.get('finished')))
    rows = []
    for v in runs.values():
        if len(v) < 2:
            continue
        a, b = v[0], v[1]
        same = all(abs(a[3][m] - b[3][m]) < 1e-9 for m in ('Accuracy', 'F1-Score', 'Precision', 'Detection Rate (Recall)',
                                                           'False Alarm Rate (FAR)'))
        rows.append(dict(method=a[1], seed=a[2], first=f"{a[0]} ({a[5]})", second=f"{b[0]} ({b[5]})",
                         accuracy_first=a[3]['Accuracy'], accuracy_second=b[3]['Accuracy'], identical=same))
    t = pd.DataFrame(rows)
    md = ["# R1.8 — Reproducibility\n",
          "* Run identity = hash of the full configuration; manifests store SHA-256 of code and data (`results_r02/manifests/`).",
          "* Grace rounds passed explicitly; `warmup_rounds = N` means exactly N rounds (unit-tested for N = 0, 2, 3, 4: "
          "`tests/check_invariants.py`).",
          "* The test split is evaluated once, after the predetermined final round; monitoring uses validation only.",
          "* One configuration source (`run_experiments.py --list`); README mirrors it.\n"]
    if len(t):
        t.to_csv(os.path.join(OUT, 'R1.8_duplicate_runs.csv'), index=False)
        md += [f"**Independent re-runs:** {len(t)} configurations were executed twice by different batches/processes "
               f"at different times; **{int(t.identical.sum())}/{len(t)} give identical test metrics** (to machine "
               "precision).\n", to_md(t.groupby('method').identical.agg(['count', 'sum']).rename(
                   columns={'count': 'pairs', 'sum': 'identical'}), index=True), "\n"]
    open(os.path.join(OUT, 'R1.8.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


BLOCKS = {'R1.1': r1_1, 'R1.2': r1_2, 'R1.3': r1_3, 'R1.4': r1_4, 'R1.4b': r1_4b, 'R1.5': r1_5, 'R1.8': r1_8, 'R1.9': r1_9}

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in BLOCKS.items():
        if which in ('all', name):
            fn()

