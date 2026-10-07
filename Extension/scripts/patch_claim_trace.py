import io

P = 'D:/Samia/Round_02/audit/claim_trace.py'
s = io.open(P, encoding='utf-8').read()
B = chr(92)
NL = chr(10)


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("            fl = M.trusted_runs('FLTrust', cond, ratio)" + NL + "            for col, s, metric in ((3, v_, 'Accuracy'), (4, v_, 'F1-Score'), (7, dg, 'Accuracy'), (8, fl, 'Accuracy')):",
    "            fl = M.trusted_runs('FLTrust', cond, ratio)" + NL
    + "            if cond == 'size300' and len(fl) == 0:        # FLTrust always uses a 300-record root set: its cyclic-sweep runs, seeds 0-4" + NL
    + "                fl = M.DF[(M.DF.exp == 'cyclic') & (M.DF.method == 'FLTrust') & (M.DF.ratio == ratio) & (M.DF.seed <= 4) & (M.DF.attack == 'cyclic_flip')" + NL
    + "                          & M.DF.schedule.isna() & M.DF.participation.isna()].drop_duplicates('seed')" + NL
    + "            for col, s, metric in ((3, v_, 'Accuracy'), (4, v_, 'F1-Score'), (7, dg, 'Accuracy'), (8, fl, 'Accuracy')):")


EXT = r'''

# ===================================================================================================== Extension (protocol §6c.26)
def tex_rows(name):
    """data rows of a generated LaTeX table: list of cell lists"""
    rows = []
    for line in open(os.path.join(RES, 'paper', 'tables', name + '.tex'), encoding='utf-8'):
        line = line.strip()
        if ' & ' not in line or 'makecell' in line or 'textbf' in line:
            continue
        line = line[:-2] if line.endswith(chr(92) * 2) else line
        rows.append([c.strip() for c in line.split('&')])
    return rows


def seeds_of(rows, lo, hi):
    return rows[(rows.seed >= lo) & (rows.seed <= hi)]


def holm_adj(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, p[i] * (len(p) - rank))
        adj[i] = min(1.0, running)
    return adj


def parse_delta(cell):
    m = re.match(r'^([+-]\d+\.\d+)(?:\s*\((\d\.\d+)\))?$', str(cell).strip())
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else None


def t_extension():
    E3 = {'DyF-ZTL (v2.2)': 'DyF-ZTL-V', 'SCAFFOLD': 'SCAFFOLD', 'FedProx (uniform)': 'FedProx-U', 'FLTrust': 'FLTrust', 'FedAvg (uniform)': 'FedAvg-U',
          'FedProx': 'FedProx', 'FedAvg': 'FedAvg', 'MLP + engine': 'DeepTrust-V', 'DFNN, no engine': 'FuzzyNoTrust', 'Round-1 engine': 'DyF-ZTL-v1',
          'Server-only (DFNN)': 'ServerOnly'}
    # --- Table C3: clean comparison on seeds 0-1 / 2-9 / 0-9 and paired differences on seeds 2-9
    ref = M.clean('DyF-ZTL-V')
    ref_acc, ref_f1 = jvals(ref, 'Accuracy'), jvals(ref, 'F1-Score')
    rows = tex_rows('tab_confirm_clean')
    deltas = {}
    for r in rows:
        key = E3.get(r[0])
        if key is None:
            continue
        g = M.clean(key)
        acc, f1 = jvals(g, 'Accuracy'), jvals(g, 'F1-Score')
        for col, lo, hi, vals in ((1, 0, 1, acc), (2, 2, 9, acc), (3, 0, 9, acc), (4, 2, 9, f1)):
            sub = vals[(vals.index >= lo) & (vals.index <= hi)]
            v, rec = check_msd(r[col], sub, 2)
            add('Table C3 confirmation clean', 'tab:confirm_clean', r[0], f"col {col} (seeds {lo}-{hi})", r[col], rec, v, len(sub), ids(seeds_of(g, lo, hi)), files(seeds_of(g, lo, hi)))
        if key != 'DyF-ZTL-V':
            idx = [k for k in acc.index if 2 <= k <= 9 and k in ref_acc.index]
            d_acc = (ref_acc[idx] - acc[idx]).astype(float)
            d_f1 = (ref_f1[idx] - f1[idx]).astype(float)
            deltas[r[0]] = (r, d_acc, d_f1, stats.ttest_1samp(d_acc, 0).pvalue, stats.ttest_1samp(d_f1, 0).pvalue, g)
    names = list(deltas)
    p_acc = holm_adj([deltas[n][3] for n in names])
    p_f1 = holm_adj([deltas[n][4] for n in names])
    for n, pa, pf in zip(names, p_acc, p_f1):
        r, d_acc, d_f1, _, _, g = deltas[n]
        for col, d, p in ((5, d_acc, pa), (6, d_f1, pf)):
            cell = parse_delta(r[col])
            rec = f"{d.mean():+.2f} ({p:.3f})"
            ok = cell is not None and close(cell[0], d.mean(), 2) and (cell[1] is None or close(cell[1], p, 3))
            add('Table C3 confirmation clean', 'tab:confirm_clean', r[0], f"col {col} (paired, Holm)", r[col], rec, 'yes' if ok else 'no', len(d), ids(seeds_of(g, 2, 9)) + '|' + ids(seeds_of(ref, 2, 9)), files(seeds_of(g, 2, 9)))
    # --- Table C4: cyclic sweep on seeds 2-9 and paired differences vs FLTrust
    rows = tex_rows('tab_confirm_cyclic')
    ratios = [i / 10 for i in range(10)]
    meth = {'DyF-ZTL (v2.2) (seeds 2--9)': 'DyF-ZTL-V', 'FLTrust (seeds 2--9)': 'FLTrust', 'Round-1 engine (seeds 2--9)': 'DyF-ZTL-v1'}
    for r in rows:
        if r[0] in meth:
            for col, ratio in enumerate(ratios, start=1):
                g = M.cyclic(meth[r[0]], ratio)
                vals = jvals(g, 'Accuracy')
                sub = vals[(vals.index >= 2) & (vals.index <= 9)]
                v, rec = check_msd(r[col], sub, 1)
                add('Table C4 confirmation cyclic', 'tab:confirm_cyclic', r[0], f"ratio {ratio:.1f}", r[col], rec, v, len(sub), ids(seeds_of(g, 2, 9)), files(seeds_of(g, 2, 9)))
        elif 'vs FLTrust' in r[0]:
            lo, hi = (2, 9) if '2--9' in r[0] else (0, 1)
            for col, ratio in enumerate(ratios, start=1):
                a, b = M.cyclic('DyF-ZTL-V', ratio), M.cyclic('FLTrust', ratio)
                va, vb = jvals(a, 'Accuracy'), jvals(b, 'Accuracy')
                idx = [k for k in va.index if lo <= k <= hi and k in vb.index]
                d = (va[idx] - vb[idx]).astype(float)
                p = stats.ttest_1samp(d, 0).pvalue if len(idx) > 2 else None
                cell = parse_delta(r[col])
                rec = f"{d.mean():+.2f}" + (f" ({p:.3f})" if p is not None else '')
                ok = cell is not None and close(cell[0], d.mean(), 2) and ((cell[1] is None and p is None) or (cell[1] is not None and p is not None and close(cell[1], p, 2)))
                add('Table C4 confirmation cyclic', 'tab:confirm_cyclic', r[0], f"ratio {ratio:.1f}", r[col], rec, 'yes' if ok else 'no', len(idx), ids(seeds_of(a, lo, hi)) + '|' + ids(seeds_of(b, lo, hi)), files(seeds_of(a, lo, hi)))
    # --- Table 19: firing-strength distribution (computed from the saved models by Extension/scripts/firing_distribution.py)
    fs = pd.read_csv(os.path.join(ROOT, 'Extension', 'E4_firing', 'firing_summary.csv'))
    cols = ['neff_mean', 'neff_median', 'records_maxshare_gt_0p5_pct', 'active_share_gt_0p05_mean', 'active_raw_gt_1e3_mean', 'nodes_usage_ge_1pct', 'nodes_never_top1', 'top5_mass_pct', 'nodes_for_90pct', 'usage_gini']
    nd = [2, 2, 1, 2, 1, 0, 0, 1, 0, 2]
    main_runs = M.clean('DyF-ZTL-V')
    for r in tex_rows('tab_firing'):
        if r[0].isdigit():
            row = fs[fs.seed == int(r[0])].iloc[0]
            run = main_runs[main_runs.seed == int(r[0])]
            for col, (c, d) in enumerate(zip(cols, nd), start=1):
                v = close(float(r[col]), row[c], d)
                add('Table 19 firing distribution', 'tab:firing', f"seed {r[0]}", c, r[col], f"{row[c]:.{d}f}", 'pipeline' if v else 'no', 1, ids(run), ';'.join(rel(os.path.join(x.dir, f"global_{x.run_id}.pt")) for x in run.itertuples()),
                    'Extension/E4_firing/firing_summary.csv (model inference on the validation split)')
        elif r[0].startswith('mean'):
            for col, (c, d) in enumerate(zip(cols, nd), start=1):
                v, rec = check_msd(r[col], fs[c], d)
                add('Table 19 firing distribution', 'tab:firing', 'mean±SD', c, r[col], rec, 'yes' if v == 'yes' else 'no', len(fs), ids(main_runs), '', 'recomputed from Extension/E4_firing/firing_summary.csv')
    # --- statements
    def f(*parts):
        return ';'.join(rel(os.path.join(RES, *p.split('/'))) for p in parts)
    # rounds without an admitted client (§6.4, §6.6, §4.2.6)
    def empty_rounds(rows):
        out = []
        for r in rows.itertuples():
            d = pd.read_csv(os.path.join(r.dir, f"decisions_{r.run_id}.csv"), usecols=['Round', 'Admitted'])
            g = d.groupby('Round').Admitted.sum()
            z = g[g == 0]
            out.append((int(r.seed), len(z), int(z.index.min()) if len(z) else None))
        return out
    per_ratio = {ratio: empty_rounds(M.cyclic('DyF-ZTL-V', ratio)) for ratio in (0.7, 0.8, 0.9)}
    n_runs = {ratio: sum(1 for _, k, _ in v if k) for ratio, v in per_ratio.items()}
    n_rounds = {ratio: sum(k for _, k, _ in v) for ratio, v in per_ratio.items()}
    longest = max(((k, s, first) for v in per_ratio.values() for s, k, first in v if k), default=(0, None, None))
    rec = f"runs {n_runs[0.7]}/{n_runs[0.8]}/{n_runs[0.9]}, rounds {n_rounds[0.7]}/{n_rounds[0.8]}/{n_rounds[0.9]}, longest {longest[0]} rounds from round {longest[2]} (seed {longest[1]}, 0.9)"
    ok = (n_runs[0.7], n_runs[0.8], n_runs[0.9]) == (1, 3, 6) and (n_rounds[0.7], n_rounds[0.8], n_rounds[0.9]) == (2, 55, 169) and longest[2] == 24
    cyc = pd.concat([M.cyclic('DyF-ZTL-V', r) for r in (0.7, 0.8, 0.9)])
    addt('Text §6.4', 'sec:cyclic', 'rounds without an admitted client', '1/3/6 runs at 0.7/0.8/0.9; 2/55/169 rounds; one seed at 0.9 from round 24', rec, 'yes' if ok else 'no', len(cyc), ids(cyc), files(cyc, 'decisions'))
    cont = pd.concat([M.trusted_runs('DyF-ZTL-V', c, r) for c in ('contam5', 'contam10') for r in (0.2, 0.4)])
    er = [k for _, k, _ in empty_rounds(cont) if k]
    addt('Text §6.6', 'sec:trusted', 'rounds without an admitted client, contaminated corpora', '7 of 10 runs; 9-86 rounds per run', f"{len(er)} of {len(cont)} runs; {min(er) if er else '-'}-{max(er) if er else '-'} rounds",
         'yes' if len(er) == 7 and len(cont) == 10 and min(er) == 9 and max(er) == 86 else 'no', len(cont), ids(cont), files(cont, 'decisions'))
    # class time windows (§6.11)
    tw = pd.read_csv(os.path.join(RES, 'audit', 'time_windows.csv')).set_index('type')
    exp = {'scanning': ('04-23', '04-24'), 'dos': ('04-24', '04-25'), 'injection': ('04-25', '04-25'), 'ddos': ('04-25', '04-26'), 'password': ('04-26', '04-27'), 'xss': ('04-27', '04-27'), 'normal': ('04-02', '04-29')}
    ok = all(str(tw.loc[c, 'min'])[5:10] == a and str(tw.loc[c, 'max'])[5:10] == b for c, (a, b) in exp.items())
    addt('Text §6.11', 'sec:temporal', 'class time windows', 'scanning 23-24, DoS 24-25, injection 25, DDoS 25-26, password 26-27, XSS 27 April; Normal 2-29 April',
         '; '.join(f"{c} {str(tw.loc[c, 'min'])[5:10]}..{str(tw.loc[c, 'max'])[5:10]}" for c in exp), 'yes' if ok else 'no', '', '', f('audit/time_windows.csv'), 'audit/split_report.md')
    # size-matched comparison with FLTrust (§6.4 / §6.6 text and Table 14)
    for ratio, claim in ((0.2, '96.7 vs 95.9'), (0.4, '96.5 vs 95.1')):
        dy = M.trusted_runs('DyF-ZTL-V', 'size300', ratio)
        fl = M.DF[(M.DF.exp == 'cyclic') & (M.DF.method == 'FLTrust') & (M.DF.ratio == ratio) & (M.DF.seed <= 4) & (M.DF.attack == 'cyclic_flip') & M.DF.schedule.isna() & M.DF.participation.isna()].drop_duplicates('seed')
        a, b = jvals(dy, 'Accuracy').mean(), jvals(fl, 'Accuracy').mean()
        want = [float(x) for x in claim.split(' vs ')]
        addt('Text §6.4', 'sec:cyclic', f"size-matched 300-record comparison, ratio {ratio}", claim, f"{a:.1f} vs {b:.1f}", 'yes' if close(a, want[0], 1) and close(b, want[1], 1) else 'no', len(dy) + len(fl), ids(dy) + '|' + ids(fl), files(dy) + ';' + files(fl))
    mk = jvals(M.cyclic('MultiKrum', 0.7), 'Accuracy').mean()
    addt('Text §6.4', 'sec:cyclic', 'Multi-Krum at 70% attackers', '24.4%', f"{mk:.1f}", 'yes' if close(mk, 24.4, 1) else 'no', 10, ids(M.cyclic('MultiKrum', 0.7)), files(M.cyclic('MultiKrum', 0.7)))
    a, b = M.cyclic('DyF-ZTL-V', 0.3), M.cyclic('FLTrust', 0.3)
    d = (jvals(a, 'Accuracy') - jvals(b, 'Accuracy')).dropna()
    addt('Text §6.4', 'sec:cyclic', 'DyF-ZTL minus FLTrust at 30% attackers', '-0.16 points (95.8 vs 95.9)', f"{d.mean():+.2f} ({jvals(a, 'Accuracy').mean():.1f} vs {jvals(b, 'Accuracy').mean():.1f})",
         'yes' if close(d.mean(), -0.16, 2) else 'no', len(d), ids(a) + '|' + ids(b), files(a) + ';' + files(b))
    # E4 statements (same source as Table 19)
    m = fs.mean(numeric_only=True)
    addt('Text §6.12', 'sec:interpretability', 'firing distribution', 'n_eff 7.1±1.2; 15% records >0.5 to one node; 18.9 nodes ≥1%; top-5 61%; 16 nodes for 90%; 13±7 never top-1; purity 47%',
         f"{m.neff_mean:.1f}±{fs.neff_mean.std():.1f}; {m.records_maxshare_gt_0p5_pct:.0f}%; {m.nodes_usage_ge_1pct:.1f}; {m.top5_mass_pct:.0f}%; {m.nodes_for_90pct:.0f}; {m.nodes_never_top1:.0f}±{fs.nodes_never_top1.std():.0f}; {100 * m.top1_purity_weighted:.0f}%",
         'pipeline', len(fs), ids(main_runs), '', 'Extension/E4_firing/firing_summary.csv')
    # figures of the extension
    add('Figures firing distribution', 'fig:firing_usage/fig:firing_class_node/fig:firing_neff', 'seed 4 (usage, class x node); all seeds (n_eff)', 'firing shares on the validation split', 'model inference', '', 'pipeline', 10, ids(main_runs),
        ';'.join(rel(os.path.join(x.dir, f"global_{x.run_id}.pt")) for x in main_runs.itertuples()), 'Extension/E4_firing/firing_per_node.csv, firing_class_by_node_seed4.csv, firing_neff_records.csv')
'''
rep(NL + NL + '# ===================================================================================================== main', EXT + NL + '# ===================================================================================================== main')
rep("t_server_only, t_heterogeneity, t_drift, t_rules, t_systems, t_sensitivity, t_gates, figures, text_claims):",
    "t_server_only, t_heterogeneity, t_drift, t_rules, t_systems, t_sensitivity, t_gates, figures, text_claims, t_extension):")
io.open(P, 'w', encoding='utf-8', newline=NL).write(s)
import ast
ast.parse(s)
print('claim_trace.py extended')

