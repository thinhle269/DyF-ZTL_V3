import ast
import io

P = 'D:/Samia/Round_02/audit/claim_trace.py'
s = io.open(P, encoding='utf-8').read()
NL = chr(10)
B = chr(92)
EXT2 = r'''

def t_extension_runs():
    import glob as _glob
    LABEL = {'DyF-ZTL-V': 'DyF-ZTL (v2.2)', 'SCAFFOLD2-SGD05': 'SCAFFOLD', 'SCAFFOLD': 'SCAFFOLD', 'FedProx-U': 'FedProx (uniform)', 'FLTrust': 'FLTrust'}

    def finals(preset):
        out = []
        for f in sorted(_glob.glob(os.path.join(RES, preset, 'final_*.json'))):
            j = json.load(open(f, encoding='utf-8'))
            c = j['config']
            out.append(dict(run_id=os.path.basename(f)[6:-5], dir=os.path.join(RES, preset), method=LABEL.get(c['method'], c['method']), seed=c['seed'], ratio=c['ratio'],
                            attack=c['attack'], acc=j['test']['Accuracy'], f1=j['test']['F1-Score'],
                            asr_cond=100 * j['test_trigger_asr_cond'] if j.get('test_trigger_asr_cond') is not None else np.nan,
                            a2n=100 * j['test_attack_to_normal'] if j.get('test_attack_to_normal') is not None else np.nan))
        return pd.DataFrame(out)

    def adm(rows):
        rej, wgt, blend = [], [], []
        for r in rows.itertuples():
            d = pd.read_csv(os.path.join(r.dir, f"decisions_{r.run_id}.csv"))
            att = d[d.Malicious == 1]
            rej.append(100 * (1 - att.Admitted.mean()))
            wgt.append(100 * att.groupby('Round').Weight.sum().mean())
            blend.append(att.AttackMix.mean() if 'AttackMix' in d.columns and att.AttackMix.notna().any() else np.nan)
        return np.array(rej), np.array(wgt), np.array(blend)

    # ---- Table C5 (E1)
    e1 = finals('ext_confirm')
    e1['cond'] = np.where(e1.attack.eq('backdoor_boost'), 'boosted backdoor 0.4', np.where(e1.ratio == 0, 'clean', 'cyclic ' + e1.ratio.astype(str)))
    ref = e1[e1.method == 'DyF-ZTL (v2.2)']
    for r in tex_rows('tab_e1'):
        cond, meth = r[0], r[1]
        g = e1[(e1.cond == cond) & (e1.method == meth)].drop_duplicates('seed').set_index('seed')
        cell = re.sub(r'\s*\(\$n\$=\d+\)\s*$', '', r[2])
        v, rec = check_msd(cell, g.acc, 2)
        add('Table C5 replication seeds 10-14', 'tab:e1_confirm', f"{cond} / {meth}", 'accuracy seeds 10-14', r[2], rec, v, len(g), ';'.join(sorted(g.run_id)), ';'.join(rel(os.path.join(x.dir, f"final_{x.run_id}.json")) for x in g.itertuples()))
        v, rec = check_msd(r[4], g.f1, 2)
        add('Table C5 replication seeds 10-14', 'tab:e1_confirm', f"{cond} / {meth}", 'macro-F1 seeds 10-14', r[4], rec, v, len(g), ';'.join(sorted(g.run_id)), '')
        if cond == 'boosted backdoor 0.4' and r[5] != '--':
            v = close(float(r[5].split('/')[0]), g.asr_cond.mean(), 0)
            add('Table C5 replication seeds 10-14', 'tab:e1_confirm', f"{cond} / {meth}", 'conditional ASR seeds 10-14', r[5].split('/')[0].strip(), f"{g.asr_cond.mean():.0f}", 'yes' if v else 'no', len(g), ';'.join(sorted(g.run_id)), '')
        if meth != 'DyF-ZTL (v2.2)' and r[6] != '--':
            a = ref[ref.cond == cond].drop_duplicates('seed').set_index('seed')
            idx = a.index.intersection(g.index)
            d = (a.loc[idx, 'acc'] - g.loc[idx, 'acc']).astype(float)
            p = stats.ttest_1samp(d, 0).pvalue if len(idx) > 2 else None
            cell = parse_delta(r[6])
            ok = cell is not None and close(cell[0], d.mean(), 2) and (cell[1] is None or (p is not None and close(cell[1], p, 3)))
            add('Table C5 replication seeds 10-14', 'tab:e1_confirm', f"{cond} / {meth}", 'paired difference vs DyF-ZTL', r[6], f"{d.mean():+.2f} ({p:.3f})" if p is not None else f"{d.mean():+.2f}", 'yes' if ok else 'no', len(idx), ';'.join(sorted(g.run_id)) + '|' + ';'.join(sorted(a.run_id)), '')
    # ---- Table 13 (E2)
    e2 = finals('ext_attacks')
    e2['knowledge'] = np.where(e2.attack.str.startswith('oracle'), 'oracle (white-box server)', 'surrogate (own data)')
    e2['payload'] = np.where(e2.attack.str.endswith('backdoor'), 'backdoor', 'label flip')
    main = M.DF[(M.DF.exp == 'attacks') & M.DF.attack.isin(['adaptive_flip', 'adaptive_backdoor']) & M.DF.method.isin(['DyF-ZTL-V', 'FLTrust']) & (M.DF.seed < 5)]
    for r in tex_rows('tab_e2'):
        k, pl, meth, ratio = r[0], r[1], r[2], float(r[3])
        if k.startswith('form of the rule'):
            g = main[(main.attack == ('adaptive_backdoor' if pl == 'backdoor' else 'adaptive_flip')) & (main.ratio == ratio) & (main.method == ('DyF-ZTL-V' if meth.startswith('DyF') else 'FLTrust'))].drop_duplicates('seed')
            acc = jvals(g, 'Accuracy'); a2n = jvals(g, 'attack_to_normal'); asr = jvals(g, 'asr_cond')
            ids_, fl = ids(g), files(g)
        else:
            g = e2[(e2.knowledge == k) & (e2.payload == pl) & (e2.method == meth) & (e2.ratio == ratio)].drop_duplicates('seed')
            acc, a2n, asr = g.acc, g.a2n, g.asr_cond
            ids_, fl = ';'.join(sorted(g.run_id)), ';'.join(rel(os.path.join(x.dir, f"final_{x.run_id}.json")) for x in g.itertuples())
        v, rec = check_msd(r[5], acc, 2)
        add('Table 13 policy-aware attackers', 'tab:e2_knowledge', f"{k} / {pl} / {meth} / {ratio}", 'accuracy', r[5], rec, v, len(g), ids_, fl)
        for col, vals, name in ((6, a2n, 'attack->normal'), (7, asr, 'conditional ASR')):
            if r[col] != '--':
                v = close(float(r[col]), np.nanmean(vals), 1)
                add('Table 13 policy-aware attackers', 'tab:e2_knowledge', f"{k} / {pl} / {meth} / {ratio}", name, r[col], f"{np.nanmean(vals):.1f}", 'yes' if v else 'no', len(g), ids_, fl)
        if not k.startswith('form of the rule'):
            rej, wgt, blend = adm(g)
            for col, vals, name, nd in ((8, rej, 'attacker rejection', 1), (9, wgt, 'attacker weight', 1), (10, blend, 'mean blend', 2)):
                if r[col] != '--':
                    v = close(float(r[col]), np.nanmean(vals), nd)
                    add('Table 13 policy-aware attackers', 'tab:e2_knowledge', f"{k} / {pl} / {meth} / {ratio}", name, r[col], f"{np.nanmean(vals):.{nd}f}", 'yes' if v else 'no', len(g), ids_, ';'.join(rel(os.path.join(x.dir, f"decisions_{x.run_id}.csv")) for x in g.itertuples()))
'''
s = s.replace(NL + '# ===================================================================================================== main', EXT2 + NL + '# ===================================================================================================== main')
old = "t_sensitivity, t_gates, figures, text_claims, t_extension):"
assert s.count(old) == 1
s = s.replace(old, "t_sensitivity, t_gates, figures, text_claims, t_extension, t_extension_runs):")
ast.parse(s)
io.open(P, 'w', encoding='utf-8', newline=NL).write(s)
print('claim trace: E1/E2 tables added')

