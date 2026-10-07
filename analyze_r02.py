import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS = ['Accuracy', 'F1-Score', 'Detection Rate (Recall)', 'Precision', 'False Alarm Rate (FAR)', 'FAR Benign->Attack']
MAIN = 'DyF-ZTL-V'
CONFIG_KEYS = ['exp', 'method', 'seed', 'ratio', 'rounds', 'attack', 'schedule', 'participation', 'trust_condition',
               'trust_val_size', 'gate_set', 'aggregation', 'weight_basis', 'loss', 'alpha', 'smote', 'split', 'features']


def _uniform_family():


    try:
        import run_experiments as rx
        return {m for m, cfg in rx.METHODS.items()
                if (cfg.get('sim') or {}).get('aggregation') == 'uniform' and not m.endswith('-U')}
    except Exception:
        return {'DyF-ZTL', 'DyF-ZTL-v1', 'FuzzyNoTrust', 'DeepTrust'}


def load(outdir):
    rows, per_class = [], []
    fam = _uniform_family()
    for p in glob.glob(os.path.join(outdir, '*', 'final_*.json')):
        exp_dir = os.path.basename(os.path.dirname(p))
        if exp_dir in ('smoke', 'smoke_v2'):
            continue
        r = json.load(open(p))
        c = dict(r['config'])
        if (c['method'] in fam or c['method'] == 'DyF-ZTL-V') and c.get('aggregation') == 'weighted':
            c['method'] = c['method'] + '-W'
        row = {k: c.get(k) for k in CONFIG_KEYS}
        row['trust_params'] = json.dumps({k: v for k, v in c.get('trust_params', {}).items() if k != 'gates'}, sort_keys=True)
        row.update({k: r['test'].get(k) for k in METRICS})
        row.update(run_id=r['run_id'], asr=r.get('test_trigger_asr'), asr_cond=r.get('test_trigger_asr_cond'),
                   attack_to_normal=r.get('test_attack_to_normal'), wall_time_s=r.get('wall_time_s'),
                   device=r.get('device'), dir=os.path.dirname(p))
        for k in ('asr', 'asr_cond', 'attack_to_normal'):
            if row[k] is not None:
                row[k] = row[k] * 100
        rows.append(row)
        for pc in r.get('per_class') or []:
            per_class.append(dict(exp=c['exp'], method=c['method'], seed=c['seed'], ratio=c['ratio'],
                                  split=c.get('split'), run_id=r['run_id'], **pc))
    df = pd.DataFrame(rows)
    pcd = pd.DataFrame(per_class)
    if len(df):
        df = df.drop_duplicates('run_id')

        ok = df.exp.str.startswith('pilot_') | (df.exp == 'calibrate') | (df.seed < 100)
        if (~ok).any():
            print(f"[WARN] {int((~ok).sum())} runs with seed >= 100 in evaluation presets were dropped from the tables")
        df = df[ok]
        if len(pcd):
            pcd = pcd[pcd.run_id.isin(df.run_id)]
    return df, pcd


def to_md(frame, index=True):

    f = frame.reset_index() if index else frame
    cols = [str(c) for c in f.columns]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else ('' if pd.isna(v) else str(v))
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in f.itertuples(index=False)]
    return '\n'.join(lines)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b):

    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None
    x, y = a.loc[common].astype(float).values, b.loc[common].astype(float).values
    d = x - y
    sd = d.std(ddof=1)
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    t_p = stats.ttest_rel(x, y).pvalue if sd > 0 else 1.0
    try:
        w_p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        w_p = 1.0
    return dict(n=len(d), mean_a=x.mean(), mean_b=y.mean(), diff=d.mean(),
                ci_low=d.mean() - tcrit * sd / np.sqrt(len(d)), ci_high=d.mean() + tcrit * sd / np.sqrt(len(d)),
                p_t=t_p, p_w=w_p, d_z=(d.mean() / sd) if sd > 0 else np.nan)


def summarize(df, keys, metrics=METRICS):
    g = df.groupby(keys, dropna=False)[metrics]
    out = g.agg(['mean', 'std', 'count'])
    out.columns = [f"{m} ({s})" for m, s in out.columns]
    return out.round(3)


def admission_metrics(path, attack_rounds_only=True):


    full = pd.read_csv(path)
    if 'Admitted' not in full.columns or 'Malicious' not in full.columns:
        return None
    d = full[full.Phase != 'grace'] if 'Phase' in full.columns else full
    if not len(d):
        return None
    mal, hon = d[d.Malicious == 1], d[d.Malicious == 0]
    rej = d[d.Admitted == 0]
    res = dict(malicious_rejection=1 - mal.Admitted.mean() if len(mal) else np.nan,
               honest_rejection=1 - hon.Admitted.mean() if len(hon) else np.nan,
               precision_reject=(rej.Malicious.mean() if len(rej) else np.nan),
               mean_admitted=d.groupby('Round').Admitted.sum().mean(),
               empty_rounds=int((d.groupby('Round').Admitted.sum() == 0).sum()))
    res['recall_reject'] = res['malicious_rejection']
    if 'Weight' in d.columns:
        per_round = d.groupby('Round').apply(lambda g: g.loc[g.Malicious == 1, 'Weight'].sum() / max(g.Weight.sum(), 1e-12))
        res['malicious_weight_share'] = float(per_round.mean()) if len(per_round) else np.nan
    else:
        res['malicious_weight_share'] = np.nan
    t_ex = []
    for cl, g in full[full.Malicious == 1].groupby('Client'):
        g = g.sort_values('Round')
        first_att = g.Round.iloc[0]
        ex = g[g.Admitted == 0]
        t_ex.append((ex.Round.iloc[0] - first_att) if len(ex) else np.nan)
    res['time_to_exclusion'] = np.nanmean(t_ex) if t_ex and not all(np.isnan(t_ex)) else np.nan
    res['attackers_never_excluded'] = int(np.sum(np.isnan(t_ex))) if t_ex else 0
    rehab = []
    for cl, g in hon.groupby('Client'):
        g = g.sort_values('Round').reset_index(drop=True)
        excluded_at = None
        for _, row in g.iterrows():
            if row.Admitted == 0 and excluded_at is None:
                excluded_at = row.Round
            elif row.Admitted == 1 and excluded_at is not None:
                rehab.append(row.Round - excluded_at)
                excluded_at = None
        if excluded_at is not None:
            rehab.append(np.nan)
    res['honest_exclusion_episodes'] = len(rehab)
    res['time_to_rehabilitation'] = np.nanmean(rehab) if rehab and not all(np.isnan(rehab)) else np.nan
    res['episodes_never_rehabilitated'] = int(np.sum(np.isnan(rehab))) if rehab else 0
    return res


def _partition(outdir, cfg):
    alpha = cfg.get('alpha')
    feat = {'drift_robust': '_dr', 'drift_oracle': '_do'}.get(cfg.get('features'), '')
    name = f"{cfg.get('split', 'random')}_a{alpha}_{'smote' if cfg.get('smote', True) else 'nosmote'}_s{cfg['seed']}{feat}.json"
    p = os.path.join(outdir, 'partitions', name)
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


WINDOWS = ((1, 20), (21, 50), (51, 100))
NORMAL_DOMINATED = 0.8


def admission_windows(df, outdir):


    rows = []
    for _, r in df.iterrows():
        p = os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if 'Admitted' not in d.columns:
            continue
        cfg = json.load(open(os.path.join(r['dir'], f"final_{r['run_id']}.json")))['config']
        part = _partition(outdir, cfg)
        if part is None:
            continue
        normal = part['classes'].index('normal')
        share = [np.asarray(c['hist_pre'], dtype=float)[normal] / max(1.0, float(np.sum(c['hist_pre'])))
                 for c in part['clients']]
        d['group'] = np.where(d.Malicious == 1, 'attacker',
                              np.where(d.Client.map(lambda i: share[int(i)]) >= NORMAL_DOMINATED,
                                       'honest_normal_dominated', 'honest_other'))
        for lo, hi in WINDOWS:
            w = d[(d.Round >= lo) & (d.Round <= hi)]
            for g, gg in w.groupby('group'):
                rows.append(dict(exp=r['exp'], method=r['method'], attack=r['attack'], ratio=r['ratio'], seed=r['seed'],
                                 schedule=r.get('schedule'), participation=r.get('participation'),
                                 window=f"{lo}-{hi}", group=g, admission_rate=gg.Admitted.mean(), n=len(gg)))
    return pd.DataFrame(rows)


def client_admission(df, outdir):


    rows, clean = [], []
    for _, r in df.iterrows():
        p = os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if 'Admitted' not in d.columns:
            continue
        if 'Phase' in d.columns:
            d = d[d.Phase != 'grace']
        cfg = json.load(open(os.path.join(r['dir'], f"final_{r['run_id']}.json")))['config']
        part = _partition(outdir, cfg)
        if part is None or not len(d):
            continue
        normal = part['classes'].index('normal')
        rate = d.groupby('Client').Admitted.mean()
        mal = d.groupby('Client').Malicious.max()
        for cl, a_rate in rate.items():
            h = np.asarray(part['clients'][int(cl)]['hist_pre'], dtype=float)
            rows.append(dict(exp=r['exp'], method=r['method'], seed=r['seed'], ratio=r['ratio'], attack=r['attack'],
                             run_id=r['run_id'], client=int(cl), malicious=int(mal[cl]), admission_rate=a_rate,
                             n_pre_smote=int(h.sum()), normal_share=h[normal] / h.sum(), n_classes=int((h > 0).sum())))
        if not r['ratio']:
            sub = pd.DataFrame([x for x in rows if x['run_id'] == r['run_id']])
            clean.append(dict(exp=r['exp'], method=r['method'], seed=r['seed'],
                              honest_rejection=1 - d.Admitted.mean(),
                              mean_admitted=d.groupby('Round').Admitted.sum().mean(),
                              clients_below_50pct=int((sub.admission_rate < 0.5).sum()),
                              spearman_adm_normal_share=stats.spearmanr(sub.admission_rate, sub.normal_share)[0]
                              if sub.admission_rate.nunique() > 1 else np.nan,
                              spearman_adm_size=stats.spearmanr(sub.admission_rate, sub.n_pre_smote)[0]
                              if sub.admission_rate.nunique() > 1 else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(clean)


def completeness(outdir):
    try:
        import argparse as ap_
        import run_experiments as rx
    except Exception as e:
        return pd.DataFrame([dict(preset='?', error=str(e))])
    rows = []
    ns = ap_.Namespace(seeds=None, rounds=None, methods=None, ratios=None)
    for name, preset in rx.PRESETS.items():
        if name.startswith('smoke'):
            continue
        jobs = rx.expand(preset, ns)
        if any(j[3] is None for j in jobs):
            rows.append(dict(preset=name, expected=len(jobs), finished=None, missing=None, note='R_attack not set'))
            continue
        fin = miss = 0
        note = ''
        for m, s, r, R, d, ex in jobs:
            cfg_ns = ap_.Namespace(clients=20, epochs=preset.get('epochs', 5), batch=32,
                                   aggregation='weighted', weight_basis='original')
            try:
                rid = rx.run_id_of(rx.run_config(name, m, s, r, R, d, cfg_ns, ex.get('trust'), ex.get('sim'),
                                                 ex.get('gate_set')))
            except FileNotFoundError:
                note = 'gates/trigger not calibrated yet'
                miss += 1
                continue
            if os.path.exists(os.path.join(outdir, name, f'final_{rid}.json')):
                fin += 1
            else:
                miss += 1
        rows.append(dict(preset=name, expected=len(jobs), finished=fin, missing=miss, note=note))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results_r02'))
    args = ap.parse_args()
    out = os.path.join(args.outdir, 'analysis')
    os.makedirs(out, exist_ok=True)
    df, pc = load(args.outdir)
    md = [f"# Round-2 analysis ({pd.Timestamp.now():%Y-%m-%d %H:%M})\n", f"Finished runs: {len(df)}\n"]

    try:
        sc = json.load(open(os.path.join(args.outdir, 'calibration', 'scaffold.json')))
    except (OSError, ValueError):
        sc = None
    if sc and sc.get('method', 'SCAFFOLD') != 'SCAFFOLD' and len(df):
        for frame in (df, pc):
            if len(frame):
                mask = frame.exp.isin(['main', 'heterogeneity']) & (frame.method == sc['method'])
                frame.loc[mask, 'method'] = 'SCAFFOLD'
        md.append(f"SCAFFOLD = `{sc['method']}` (algo {sc['algo']}, lr {sc['lr']}), chosen by the pilot rule.\n")
    comp = completeness(args.outdir)
    comp.to_csv(os.path.join(out, 'completeness.csv'), index=False)
    md += ["## Completeness\n", comp.pipe(to_md, index=False), "\n"]
    if not len(df):
        open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(md))
        print('\n'.join(md))
        return
    df.to_csv(os.path.join(out, 'runs_flat.csv'), index=False)


    main_df = df[df.exp == 'main']
    if len(main_df):
        s = summarize(main_df, ['method'])
        s.to_csv(os.path.join(out, 'main_summary.csv'))
        md += ["## Clean comparison (main)\n", s[[c for c in s.columns if c.split(' (')[0] in ('Accuracy', 'F1-Score', 'FAR Benign->Attack')]].pipe(to_md), "\n"]
        if len(pc):
            pcm = pc[pc.exp == 'main'].groupby(['method', 'Class'])[['Precision', 'Recall', 'F1']].agg(['mean', 'std'])
            pcm.columns = [f"{m} ({s_})" for m, s_ in pcm.columns]
            pcm.round(2).to_csv(os.path.join(out, 'main_per_class.csv'))
        tests = []
        ref = main_df[main_df.method == MAIN].set_index('seed')
        for metric in ('Accuracy', 'F1-Score'):
            fam = []
            for other in sorted(set(main_df.method) - {MAIN}):
                o = main_df[main_df.method == other].set_index('seed')
                if len(ref) and len(o):
                    res = paired(ref[metric], o[metric])
                    if res:
                        fam.append(dict(metric=metric, comparison=f"{MAIN} vs {other}", **res))
            planned = [i for i, r in enumerate(fam) if r['comparison'].split(' vs ')[1] in ('FedAvg', 'FedProx', 'SCAFFOLD')]
            if planned:
                adj = holm([fam[i]['p_t'] for i in planned])
                for i, a in zip(planned, adj):
                    fam[i]['p_t_holm'] = a
            tests += fam
        if tests:
            t = pd.DataFrame(tests).round(4)
            t.to_csv(os.path.join(out, 'main_tests.csv'), index=False)
            md += ["### Paired tests (DyF-ZTL minus other; Holm over FedAvg/FedProx/SCAFFOLD)\n", t.pipe(to_md, index=False), "\n"]
        if len(ref) >= 3:
            f1 = ref['F1-Score'].sort_values()
            med_seed = int(f1.index[(len(f1) - 1) // 2])
            json.dump(dict(seed=med_seed, rule=f'seed with the median {MAIN} macro-F1 over the main seeds',
                           f1_by_seed={int(k): float(v) for k, v in f1.items()}),
                      open(os.path.join(out, 'representative.json'), 'w'), indent=1)


    pairs = [('FedAvg', 'FedAvg-U'), ('FedProx', 'FedProx-U'), ('DyF-ZTL', 'DyF-ZTL-U'), ('DyF-ZTL', 'DyF-ZTL-PS'), ('DyF-ZTL-R', 'DyF-ZTL-R-W'), ('DyF-ZTL-V', 'DyF-ZTL-V-W')]
    base = df[df.exp.isin(['main', 'agg_ablation'])]
    agg_rows = []
    for a, b in pairs:
        A, B = base[base.method == a].set_index('seed'), base[base.method == b].set_index('seed')
        for metric in ('Accuracy', 'F1-Score'):
            if len(A) and len(B):
                res = paired(A[metric], B[metric])
                if res:
                    agg_rows.append(dict(comparison=f"{a} (weighted) vs {b}", metric=metric, **res))
    if agg_rows:
        t = pd.DataFrame(agg_rows).round(4)
        t.to_csv(os.path.join(out, 'aggregation.csv'), index=False)
        md += ["## Aggregation ablation (R1.1)\n", t.pipe(to_md, index=False), "\n"]


    reg = df[df.exp == 'regression']
    if len(reg):
        s = summarize(reg, ['method'], ['Accuracy', 'F1-Score'])
        s.to_csv(os.path.join(out, 'regression.csv'))
        md += ["## Fuzzy layer regression (F01)\n", s.pipe(to_md), "\n"]


    ch = df[df.exp == 'chrono']
    if len(ch):
        s = summarize(pd.concat([ch, main_df[main_df.method.isin(ch.method.unique())]]), ['split', 'method'],
                      ['Accuracy', 'F1-Score', 'FAR Benign->Attack'])
        s.to_csv(os.path.join(out, 'chrono.csv'))
        md += ["## Per-class chronological vs random split (R1.5)\n", s.pipe(to_md), "\n"]


    het = df[df.exp == 'heterogeneity']
    if len(het):
        het = het.assign(alpha=het.alpha.fillna('IID'))
        s = summarize(het, ['alpha', 'smote', 'loss', 'method'], ['Accuracy', 'F1-Score'])
        s.to_csv(os.path.join(out, 'heterogeneity.csv'))
        md += ["## Heterogeneity (R1.5)\n", s.pipe(to_md), "\n"]


    cy = df[df.exp == 'cyclic']
    if len(cy):
        for stat, name in (('mean', 'cyclic_mean'), ('std', 'cyclic_sd'), ('count', 'cyclic_n')):
            cy.pivot_table(index='method', columns='ratio', values='Accuracy', aggfunc=stat).round(2) \
              .to_csv(os.path.join(out, f'{name}.csv'))
        md += ["## Cyclic label-flip sweep: mean accuracy (R1.3)\n",
               cy.pivot_table(index='method', columns='ratio', values='Accuracy', aggfunc='mean').round(2).pipe(to_md), "\n"]
        tests = []
        for ratio, g in cy.groupby('ratio'):
            ref = g[g.method == MAIN].set_index('seed')
            for other in sorted(set(g.method) - {MAIN}):
                res = paired(ref['Accuracy'], g[g.method == other].set_index('seed')['Accuracy']) if len(ref) else None
                if res:
                    tests.append(dict(ratio=ratio, comparison=f"{MAIN} vs {other}", **res))
        if tests:
            pd.DataFrame(tests).round(4).to_csv(os.path.join(out, 'cyclic_tests.csv'), index=False)


    at = df[df.exp == 'attacks']
    if len(at):
        at = at.assign(schedule=at.schedule.fillna('static'), participation=at.participation.fillna(1.0))
        s = summarize(at, ['attack', 'schedule', 'participation', 'ratio', 'method'],
                      ['Accuracy', 'F1-Score', 'asr_cond', 'asr', 'attack_to_normal'])
        s.to_csv(os.path.join(out, 'attacks.csv'))
        md += ["## Other attacks (R1.3, R2.3)\n", s.pipe(to_md), "\n"]


    adm = []
    for _, r in df.iterrows():
        p = os.path.join(r['dir'], f"decisions_{r['run_id']}.csv")
        if os.path.exists(p) and r['ratio'] and r['ratio'] > 0:
            m = admission_metrics(p)
            if m:
                adm.append(dict({k: r[k] for k in ('exp', 'method', 'seed', 'ratio', 'attack', 'schedule', 'participation',
                                                  'trust_condition', 'trust_val_size', 'gate_set', 'trust_params')}, **m))
    if adm:
        a = pd.DataFrame(adm)
        a.to_csv(os.path.join(out, 'admission_runs.csv'), index=False)
        keys = ['exp', 'method', 'attack', 'ratio']
        agg = a.groupby(keys, dropna=False)[['malicious_rejection', 'honest_rejection', 'precision_reject', 'mean_admitted',
                                             'empty_rounds', 'time_to_exclusion', 'time_to_rehabilitation']].mean().round(3)
        agg.to_csv(os.path.join(out, 'admission.csv'))
        md += ["## Admission metrics (R1.4), mean over seeds\n", agg.pipe(to_md), "\n"]


    pcl, clean = client_admission(df, args.outdir)
    if len(pcl):
        pcl.to_csv(os.path.join(out, 'admission_by_client.csv'), index=False)
    if len(clean):
        clean.to_csv(os.path.join(out, 'admission_clean_runs.csv'), index=False)
        cagg = clean.groupby(['exp', 'method'])[['honest_rejection', 'mean_admitted', 'clients_below_50pct',
                                                 'spearman_adm_normal_share', 'spearman_adm_size']].agg(['mean', 'std', 'count'])
        cagg.columns = [f"{m} ({s_})" for m, s_ in cagg.columns]
        cagg = cagg.round(3)
        cagg.to_csv(os.path.join(out, 'admission_clean.csv'))
        md += ["## Honest clients excluded without any attacker (clean runs; R1.2)\n",
               "Spearman correlations are between a client's post-grace admission rate and its share of "
               "`normal` records / its pre-SMOTE size.\n", cagg.pipe(to_md), "\n"]
    aw = admission_windows(df, args.outdir)
    if len(aw):
        aw.to_csv(os.path.join(out, 'admission_windows_runs.csv'), index=False)
        piv = aw.groupby(['exp', 'method', 'attack', 'ratio', 'group', 'window']).admission_rate.mean() \
                .unstack('window').round(3)
        piv.to_csv(os.path.join(out, 'admission_windows.csv'))
        md += ["## Admission rate by round window and client group (closed-loop dynamics)\n",
               f"Groups: attackers; honest clients with >= {NORMAL_DOMINATED:.0%} normal records; other honest clients.\n",
               piv.pipe(to_md), "\n"]


    for exp, keys in (('trusted_set', ['trust_condition', 'trust_val_size', 'gate_set', 'ratio', 'method']),
                      ('trust_modes', ['ratio', 'method']),
                      ('sensitivity', ['trust_params', 'gate_set', 'ratio'])):
        sub = df[df.exp == exp]
        if exp == 'sensitivity':
            sub = sub[sub.method == MAIN]
        if len(sub):
            s = summarize(sub, keys, ['Accuracy', 'F1-Score'])
            s.to_csv(os.path.join(out, f'{exp}.csv'))
            md += [f"## {exp}\n", s.pipe(to_md), "\n"]

    open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()

