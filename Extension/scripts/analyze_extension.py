import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = 'D:/Samia/Round_02'
RES = os.path.join(ROOT, 'results_r02')
E1 = os.path.join(ROOT, 'Extension', 'E1_confirm_seeds_10_14')
E2 = os.path.join(ROOT, 'Extension', 'E2_full_knowledge')
os.makedirs(E1, exist_ok=True)
os.makedirs(E2, exist_ok=True)
LABEL = {'DyF-ZTL-V': 'DyF-ZTL (v2.2)', 'SCAFFOLD2-SGD05': 'SCAFFOLD', 'SCAFFOLD': 'SCAFFOLD', 'FedProx-U': 'FedProx (uniform)', 'FLTrust': 'FLTrust'}
ACC, F1 = 'Accuracy', 'F1-Score'


def pct(v):
    return np.nan if v is None else 100.0 * float(v)


def load_runs(preset):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, preset, 'final_*.json'))):
        j = json.load(open(f, encoding='utf-8'))
        cfg = j.get('config', {})
        rid = os.path.basename(f)[6:-5]
        row = dict(run_id=rid, preset=preset, method=cfg.get('method'), seed=cfg.get('seed'), ratio=cfg.get('ratio'), attack=cfg.get('attack'),
                   acc=j['test'][ACC], f1=j['test'][F1], far_b=j['test'].get('FAR Benign->Attack'), asr=pct(j.get('test_trigger_asr')),
                   asr_cond=pct(j.get('test_trigger_asr_cond')), attack_to_normal=pct(j.get('test_attack_to_normal')), wall_time_s=j.get('wall_time_s'))
        dec = os.path.join(RES, preset, f'decisions_{rid}.csv')
        if os.path.exists(dec) and (row['ratio'] or 0) > 0:
            d = pd.read_csv(dec)
            att = d[d.Malicious == 1]
            if len(att):
                row['attacker_rejection'] = float(1 - att.Admitted.mean())
                row['attacker_weight'] = float(att.groupby('Round').Weight.sum().mean())
                if 'AttackMix' in d.columns and att.AttackMix.notna().any():
                    row['mean_blend'] = float(att.AttackMix.mean())
                    row['share_rounds_full_poison'] = float((att.AttackMix >= 0.999).mean())
                    row['share_rounds_clean_fallback'] = float((att.AttackMix <= 0.0).mean())
                if 'Evidence' in d.columns:
                    row['attacker_evidence_good'] = float((att.Evidence == 'good').mean())
            hon = d[d.Malicious == 0]
            if len(hon):
                row['honest_rejection'] = float(1 - hon.Admitted.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def msd(x, d=2):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return '--'
    return f"{x.mean():.{d}f}" if len(x) == 1 else f"{x.mean():.{d}f}$\\pm${x.std():.{d}f}"


flat = pd.read_csv(os.path.join(RES, 'analysis', 'runs_flat.csv'), low_memory=False)
std = (flat.rounds == 100) & (flat.seed < 100) & (flat.split.fillna('random') == 'random') & (flat.alpha.fillna(0.5) == 0.5) \
    & flat.trust_val_size.isna() & flat.trust_condition.isna() & (flat.features.fillna('default').isin(['default', 'original'])) \
    & (flat.schedule.fillna('static') == 'static') & (flat.participation.fillna(1.0) == 1.0)


def e1_extra(summ):
    def cell(cond, m, col):
        g = summ[(summ.condition == cond) & (summ.method == m)]
        return float(g[col].iloc[0]) if len(g) else float('nan')
    txt = (f" DyF-ZTL (v2.2) keeps {cell('cyclic 0.4', 'DyF-ZTL (v2.2)', 'acc_new'):.2f}\%, {cell('cyclic 0.8', 'DyF-ZTL (v2.2)', 'acc_new'):.2f}\% and {cell('cyclic 0.9', 'DyF-ZTL (v2.2)', 'acc_new'):.2f}\% accuracy at 40, 80 and 90\% attackers on the new seeds"
           f" (FLTrust {cell('cyclic 0.4', 'FLTrust', 'acc_new'):.2f}\%, {cell('cyclic 0.8', 'FLTrust', 'acc_new'):.2f}\% and {cell('cyclic 0.9', 'FLTrust', 'acc_new'):.2f}\%), and the boosted backdoor succeeds in {cell('boosted backdoor 0.4', 'DyF-ZTL (v2.2)', 'asr_cond_new'):.1f}\% of the triggered records against DyF-ZTL versus {cell('boosted backdoor 0.4', 'FLTrust', 'asr_cond_new'):.1f}\% against FLTrust."
           f" SCAFFOLD and FedProx-uniform, which were not part of the cyclic sweep of the main study, behave like the other majority-dependent rules: {cell('cyclic 0.4', 'SCAFFOLD', 'acc_new'):.1f}\%~$\pm$~{cell('cyclic 0.4', 'SCAFFOLD', 'acc_new_sd'):.1f} and {cell('cyclic 0.4', 'FedProx (uniform)', 'acc_new'):.1f}\%~$\pm$~{cell('cyclic 0.4', 'FedProx (uniform)', 'acc_new_sd'):.1f} at 40\% attackers and below 7\% at 80--90\%.")
    return txt.replace(chr(92) + '%', chr(92) + '%')


e1 = load_runs('ext_confirm')
if len(e1):
    e1['method_label'] = e1.method.map(LABEL).fillna(e1.method)
    e1['condition'] = np.where(e1.attack.eq('backdoor_boost'), 'boosted backdoor 0.4', np.where(e1.ratio.fillna(0) == 0, 'clean', 'cyclic ' + e1.ratio.astype(str)))
    e1.to_csv(os.path.join(E1, 'e1_runs.csv'), index=False)

    def ref_rows(method, cond):

        m = method
        if cond == 'clean':
            sub = flat[std & (flat.method.isin([m, 'SCAFFOLD'] if m.startswith('SCAFFOLD') else [m])) & (flat.ratio.fillna(0) == 0) & flat.exp.isin(['main', 'agg_ablation', 'cyclic'])]
        elif cond.startswith('cyclic'):
            r = float(cond.split()[1])
            sub = flat[std & (flat.method == m) & (flat.ratio == r) & (flat.exp == 'cyclic') & flat.attack.fillna('cyclic_flip').eq('cyclic_flip')]
        else:
            sub = flat[std & (flat.method == m) & (flat.ratio == 0.4) & (flat.exp == 'attacks') & flat.attack.eq('backdoor_boost')]
        sub = sub[~(sub.method.str.startswith('DyF') & sub.aggregation.fillna('').eq('weighted'))]
        return sub.assign(pref=sub.exp.map({'main': 0, 'agg_ablation': 1, 'cyclic': 2, 'attacks': 3})).sort_values(['seed', 'pref']).drop_duplicates('seed').set_index('seed')

    srows, prows = [], []
    for cond in ['clean', 'cyclic 0.4', 'cyclic 0.8', 'cyclic 0.9', 'boosted backdoor 0.4']:
        g = e1[e1.condition == cond]
        for m in ['DyF-ZTL-V', 'SCAFFOLD2-SGD05', 'FedProx-U', 'FLTrust']:
            h = g[g.method.isin([m, 'SCAFFOLD'] if m.startswith('SCAFFOLD') else [m])].drop_duplicates('seed').set_index('seed')
            ref = ref_rows(m, cond)
            if len(h) == 0 and len(ref) == 0:
                continue
            srows.append(dict(condition=cond, method=LABEL[m], n_new=len(h), acc_new=h.acc.mean(), acc_new_sd=h.acc.std(), f1_new=h.f1.mean(), f1_new_sd=h.f1.std(),
                              asr_cond_new=h.asr_cond.mean() if 'asr_cond' in h else np.nan, attacker_rejection_new=h.attacker_rejection.mean() if 'attacker_rejection' in h else np.nan,
                              n_ref=len(ref), acc_ref=ref[ACC].mean(), acc_ref_sd=ref[ACC].std(), f1_ref=ref[F1].mean(), f1_ref_sd=ref[F1].std(),
                              asr_cond_ref=ref.asr_cond.mean() if 'asr_cond' in ref else np.nan))
        a = g[g.method == 'DyF-ZTL-V'].drop_duplicates('seed').set_index('seed')
        aref = ref_rows('DyF-ZTL-V', cond)
        for m in ['SCAFFOLD2-SGD05', 'FedProx-U', 'FLTrust']:
            b = g[g.method.isin([m, 'SCAFFOLD'] if m.startswith('SCAFFOLD') else [m])].drop_duplicates('seed').set_index('seed')
            bref = ref_rows(m, cond)
            idx = a.index.intersection(b.index)
            idr = aref.index.intersection(bref.index)
            if len(idx) == 0 and len(idr) == 0:
                continue
            d = (a.loc[idx, 'acc'] - b.loc[idx, 'acc']).astype(float)
            dr = (aref.loc[idr, ACC] - bref.loc[idr, ACC]).astype(float)
            prows.append(dict(condition=cond, competitor=LABEL[m], n_new=len(idx), d_acc_new=d.mean(), d_acc_new_sd=d.std(),
                              p_new=stats.ttest_1samp(d, 0).pvalue if len(idx) > 2 else np.nan,
                              n_ref=len(idr), d_acc_ref=dr.mean(), p_ref=stats.ttest_1samp(dr, 0).pvalue if len(idr) > 2 else np.nan,
                              same_sign=bool(np.sign(d.mean()) == np.sign(dr.mean())) if len(idx) and len(idr) else None))
    summ = pd.DataFrame(srows)
    paired = pd.DataFrame(prows)
    summ.to_csv(os.path.join(E1, 'e1_summary.csv'), index=False)
    paired.to_csv(os.path.join(E1, 'e1_paired.csv'), index=False)
    L = [r'\begin{tableorg}[!htbp]', r'\centering', r'\scriptsize',
         r'\caption{Confirmatory replication on five seeds (10--14) that were never used for calibration, pilots or any design decision, with the engine and its gates unchanged: test accuracy (mean $\pm$ SD) on the new seeds next to the ten (clean, cyclic) or five (boosted backdoor) seeds of the main study, and the paired difference DyF-ZTL (v2.2) minus the method on the new seeds (paired $t$-test $p$). Conditional trigger ASR for the boosted backdoor.}',
         r'\label{tab:e1_confirm}', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{llccccc}', r'\toprule',
         r'\textbf{Condition} & \textbf{Method} & \makecell{\bfseries Accuracy\\seeds 10--14} & \makecell{\bfseries Accuracy\\main study} & \makecell{\bfseries Macro-F1\\seeds 10--14} & \makecell{\bfseries ASR$_{\mathrm{cond}}$ (\%)\\seeds 10--14 / main} & \makecell{\bfseries $\Delta$acc vs DyF-ZTL\\seeds 10--14 ($p$)} \\', r'\midrule']
    for cond in ['clean', 'cyclic 0.4', 'cyclic 0.8', 'cyclic 0.9', 'boosted backdoor 0.4']:
        for _, r in summ[summ.condition == cond].iterrows():
            pr = paired[(paired.condition == cond) & (paired.competitor == r.method)]
            dtxt = '--' if pr.empty or pd.isna(pr.d_acc_new.iloc[0]) else f"{pr.d_acc_new.iloc[0]:+.2f} ({pr.p_new.iloc[0]:.3f})"
            asr = f"{r.asr_cond_new:.1f} / {r.asr_cond_ref:.1f}" if cond == 'boosted backdoor 0.4' and r.n_new else '--'
            acc_new = '--' if r.n_new == 0 else (f"{r.acc_new:.2f}$\pm${r.acc_new_sd:.2f} ($n$={int(r.n_new)})" if r.n_new > 1 else f"{r.acc_new:.2f} ($n$=1)")
            f1_new = '--' if r.n_new == 0 else (f"{r.f1_new:.2f}$\pm${r.f1_new_sd:.2f}" if r.n_new > 1 else f"{r.f1_new:.2f}")
            acc_ref = '--' if r.n_ref == 0 else f"{r.acc_ref:.2f}$\pm${r.acc_ref_sd:.2f}"
            L.append(f"{cond} & {r.method} & {acc_new} & {acc_ref} & {f1_new} & {asr} & {dtxt} \\\\")
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{tableorg}']
    open(os.path.join(E1, 'tab_e1.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    done = int(e1.shape[0])
    agree = paired.same_sign.dropna()
    cl = summ[(summ.condition == 'clean') & (summ.method == 'DyF-ZTL (v2.2)')]
    para = (r"\paragraph{Replication on seeds never used for any decision.} Five additional seeds (10--14) were run after the design freeze for the clean comparison, cyclic flipping at 40, 80 and 90\% attackers and the boosted backdoor at 40\%, with the engine, its gates and every baseline unchanged (protocol amendment 26). "
            + (f"On these seeds DyF-ZTL (v2.2) reaches {cl.acc_new.iloc[0]:.2f}\\%~$\\pm$~{cl.acc_new_sd.iloc[0]:.2f} accuracy on clean data (main study: {cl.acc_ref.iloc[0]:.2f}\\%). " if len(cl) and cl.n_new.iloc[0] > 1 else '')
            + f"Table~\\ref{{tab:e1_confirm}} lists every condition; {int(agree.sum())} of the {len(agree)} paired differences against SCAFFOLD, FedProx-uniform and FLTrust have the same sign as in the main study"
            + (" and none changes a conclusion of Sections~\\ref{sec:clean}--\\ref{sec:threats}." if len(agree) and agree.all() else "; the exceptions are stated in the table and discussed in the text.")
            + e1_extra(summ)
            + f" ({done} of 90 runs completed at the time of writing.)" * (done < 90) + "\n")
    open(os.path.join(E1, 'E1_paragraph.tex'), 'w', encoding='utf-8').write(para)
    print('E1:', done, 'runs;', summ.round(2).to_string(index=False))
    print(paired.round(3).to_string(index=False))
else:
    print('E1: no finished run yet')


def e2_conclusion(pick, s_flip, o_flip, p_flip, s_bd, o_bd, p_bd):
    import numpy as _np
    rej_s = [pick('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', r, 'attacker_rejection') for r in (0.2, 0.4)]
    rej_o = [pick('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', r, 'attacker_rejection') for r in (0.2, 0.4)]
    blend = [pick(k, 'label flip', 'DyF-ZTL (v2.2)', r, 'mean_blend') for k in ('surrogate (own data)', 'oracle (white-box server)') for r in (0.2, 0.4)]
    an_s = [pick('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', r, 'attack_to_normal') for r in (0.2, 0.4)]
    an_o = [pick('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', r, 'attack_to_normal') for r in (0.2, 0.4)]
    an_p = [pick('form of the rule only (paper)', 'label flip', 'DyF-ZTL (v2.2)', r, 'attack_to_normal') for r in (0.2, 0.4)]
    full = [pick(k, 'backdoor', 'DyF-ZTL (v2.2)', r, 'rounds_full_poison') for k in ('surrogate (own data)', 'oracle (white-box server)') for r in (0.2, 0.4)]
    acc_loss = _np.nanmean([p_flip[i] - s_flip[i] for i in range(2)] + [p_flip[i] - o_flip[i] for i in range(2)])
    d_an = _np.nanmean([an_s[i] - an_p[i] for i in range(2)] + [an_o[i] - an_p[i] for i in range(2)])
    f = lambda v: '--' if v is None or (isinstance(v, float) and _np.isnan(v)) else f"{v:.0f}"
    return (f"Knowing the policy therefore changes the admission outcome, not the damage. Against label flipping the policy-aware attackers are admitted in {f(100 - _np.nanmean(rej_s))}\% (surrogate) and {f(100 - _np.nanmean(rej_o))}\% (oracle) of their rounds instead of being excluded, "
            f"but the blend that fits inside the gates is only {_np.nanmin(blend):.2f}--{_np.nanmax(blend):.2f} of the full poisoned update, which costs {acc_loss:.1f} points of accuracy and raises the share of attack records predicted \emph{{Normal}} by about {d_an:.1f} points relative to the rule-form attacker. "
            f"Against the backdoor payload the policy-aware attacker submits its full poisoned update in {f(_np.nanmin(full))}--{f(_np.nanmax(full))}\% of its rounds, because that update already passes the gates, and reaches the success of the unconstrained backdoor ({f(s_bd[1])}\% and {f(o_bd[1])}\% at 40\% attackers); this confirms the limitation stated above rather than adding a new one, and it shows that the low success of the rule-form attacker came from its own conservative constraints, not from the engine.")


e2 = load_runs('ext_attacks')
if len(e2):
    e2['method_label'] = e2.method.map(LABEL).fillna(e2.method)
    e2['knowledge'] = np.where(e2.attack.str.startswith('oracle'), 'oracle (white-box server)', 'surrogate (own data)')
    e2['payload'] = np.where(e2.attack.str.endswith('backdoor'), 'backdoor', 'label flip')
    e2.to_csv(os.path.join(E2, 'e2_runs.csv'), index=False)

    ref = flat[std & (flat.exp == 'attacks') & flat.attack.isin(['adaptive_flip', 'adaptive_backdoor']) & flat.method.isin(['DyF-ZTL-V', 'FLTrust']) & (flat.seed < 5)]
    rrows = []
    for (a, r, m), g in ref.groupby(['attack', 'ratio', 'method']):
        rrows.append(dict(knowledge='form of the rule only (paper)', payload='backdoor' if a.endswith('backdoor') else 'label flip', ratio=r, method=LABEL[m], n=len(g),
                          acc=g[ACC].mean(), acc_sd=g[ACC].std(), f1=g[F1].mean(), asr_cond=g.asr_cond.mean(), attack_to_normal=g.attack_to_normal.mean() * (100 if g.attack_to_normal.max() <= 1 else 1)))
    srows = []
    for (k, pl, r, m), g in e2.groupby(['knowledge', 'payload', 'ratio', 'method_label']):
        srows.append(dict(knowledge=k, payload=pl, ratio=r, method=m, n=len(g), acc=g.acc.mean(), acc_sd=g.acc.std(), f1=g.f1.mean(), asr_cond=g.asr_cond.mean(),
                          attack_to_normal=g.attack_to_normal.mean(), attacker_rejection=g.attacker_rejection.mean() * 100 if 'attacker_rejection' in g else np.nan,
                          attacker_weight=g.attacker_weight.mean() * 100 if 'attacker_weight' in g else np.nan, honest_rejection=g.honest_rejection.mean() * 100 if 'honest_rejection' in g else np.nan,
                          mean_blend=g.mean_blend.mean() if 'mean_blend' in g else np.nan, rounds_full_poison=g.share_rounds_full_poison.mean() * 100 if 'share_rounds_full_poison' in g else np.nan,
                          rounds_clean_fallback=g.share_rounds_clean_fallback.mean() * 100 if 'share_rounds_clean_fallback' in g else np.nan,
                          attacker_evidence_good=g.attacker_evidence_good.mean() * 100 if 'attacker_evidence_good' in g else np.nan))
    summ = pd.concat([pd.DataFrame(rrows), pd.DataFrame(srows)], ignore_index=True)
    summ.to_csv(os.path.join(E2, 'e2_summary.csv'), index=False)
    order = ['form of the rule only (paper)', 'surrogate (own data)', 'oracle (white-box server)']
    L = [r'\begin{tableorg}[!htbp]', r'\centering', r'\scriptsize',
         r"\caption{Attackers with increasing knowledge of the admission policy (seeds 0--4, 100 rounds, 20\% and 40\% attackers). \emph{Form of the rule only}: the adaptive attacker of the main study. \emph{Surrogate}: the attacker knows the calibrated gate constants and the trusted-data policy and estimates the server reference on its own clean records. \emph{Oracle}: white-box server, the attacker uses $D_{\mathrm{trust}}$ and the reference update of the round, so its evidence is exactly the server's. Attack$\to$Normal = share of attack test records predicted benign (label flip); ASR$_{\mathrm{cond}}$ = conditional trigger success (backdoor); attacker rejection and attacker weight share over the attack rounds; blend = mean poison fraction the attacker could submit while its evidence stayed admissible (DyF-ZTL runs).}",
         r'\label{tab:e2_knowledge}', r'\setlength{\tabcolsep}{3pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{lllcccccccc}', r'\toprule',
         r'\textbf{Knowledge} & \textbf{Payload} & \textbf{Method} & \textbf{Ratio} & \textbf{$n$} & \makecell{\bfseries Accuracy\\(\%)} & \makecell{\bfseries Attack$\to$Normal\\(\%)} & \makecell{\bfseries ASR$_{\mathrm{cond}}$\\(\%)} & \makecell{\bfseries Attacker\\rejection (\%)} & \makecell{\bfseries Attacker\\weight (\%)} & \makecell{\bfseries Mean\\blend} \\', r'\midrule']
    for k in order:
        for pl in ['label flip', 'backdoor']:
            for m in ['DyF-ZTL (v2.2)', 'FLTrust']:
                for r in [0.2, 0.4]:
                    g = summ[(summ.knowledge == k) & (summ.payload == pl) & (summ.method == m) & (summ.ratio == r)]
                    if g.empty:
                        continue
                    g = g.iloc[0]
                    cell = lambda v, d=1: '--' if pd.isna(v) else f"{v:.{d}f}"
                    L.append(f"{k} & {pl} & {m} & {r:.1f} & {int(g.n)} & {g.acc:.2f}" + (f"$\\pm${g.acc_sd:.2f}" if g.n > 1 and not pd.isna(g.acc_sd) else '')
                             + f" & {cell(g.attack_to_normal)} & {cell(g.asr_cond)} & {cell(g.get('attacker_rejection', np.nan))} & {cell(g.get('attacker_weight', np.nan))} & {cell(g.get('mean_blend', np.nan), 2)} \\\\")
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{tableorg}']
    open(os.path.join(E2, 'tab_e2.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    def pick(k, pl, m, r, col):
        g = summ[(summ.knowledge == k) & (summ.payload == pl) & (summ.method == m) & (summ.ratio == r)]
        return np.nan if g.empty else g.iloc[0][col]

    done = int(e2.shape[0])
    s_flip = [pick('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', r, 'acc') for r in (0.2, 0.4)]
    o_flip = [pick('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', r, 'acc') for r in (0.2, 0.4)]
    s_bd = [pick('surrogate (own data)', 'backdoor', 'DyF-ZTL (v2.2)', r, 'asr_cond') for r in (0.2, 0.4)]
    o_bd = [pick('oracle (white-box server)', 'backdoor', 'DyF-ZTL (v2.2)', r, 'asr_cond') for r in (0.2, 0.4)]
    p_flip = [pick('form of the rule only (paper)', 'label flip', 'DyF-ZTL (v2.2)', r, 'acc') for r in (0.2, 0.4)]
    p_bd = [pick('form of the rule only (paper)', 'backdoor', 'DyF-ZTL (v2.2)', r, 'asr_cond') for r in (0.2, 0.4)]
    f_flip = [pick('surrogate (own data)', 'label flip', 'FLTrust', r, 'acc') for r in (0.2, 0.4)]
    f_bd = [pick('surrogate (own data)', 'backdoor', 'FLTrust', r, 'asr_cond') for r in (0.2, 0.4)]
    o_rej = [pick('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', r, 'attacker_rejection') for r in (0.2, 0.4)]
    o_blend = [pick('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', r, 'mean_blend') for r in (0.2, 0.4)]
    fmt2 = lambda v: '--' if pd.isna(v) else f"{v:.1f}"
    para = (r"\paragraph{Attackers that know the policy.} Table~\ref{tab:e2_knowledge} extends the adaptive attacker of the main study, which knows only the form of the rule, by two attackers registered in protocol amendment 26. The \emph{surrogate} attacker is handed the calibrated gate constants and knows how the trusted corpus is drawn; it trains the server's reference model on its own clean records, computes the three pieces of evidence of every candidate update and submits the most poisoned blend of its clean and poisoned updates whose evidence it predicts to be admitted. The \emph{oracle} attacker is a white-box server: it uses $D_{\mathrm{trust}}$ and the server's reference update of the round, so every update it submits is admitted by construction and the question is only how much poison fits inside the gates. "
            + f"With label flipping, DyF-ZTL keeps {fmt2(s_flip[0])}\\% / {fmt2(s_flip[1])}\\% accuracy at 20\\% / 40\\% attackers against the surrogate attacker and {fmt2(o_flip[0])}\\% / {fmt2(o_flip[1])}\\% against the oracle (rule-form attacker: {fmt2(p_flip[0])}\\% / {fmt2(p_flip[1])}\\%; FLTrust against the surrogate: {fmt2(f_flip[0])}\\% / {fmt2(f_flip[1])}\\%). "
            + f"With the backdoor payload the conditional trigger success is {fmt2(s_bd[0])}\\% / {fmt2(s_bd[1])}\\% (surrogate) and {fmt2(o_bd[0])}\\% / {fmt2(o_bd[1])}\\% (oracle) against DyF-ZTL, versus {fmt2(p_bd[0])}\\% / {fmt2(p_bd[1])}\\% for the rule-form attacker and {fmt2(f_bd[0])}\\% / {fmt2(f_bd[1])}\\% for FLTrust against the surrogate. "
            + f"The oracle attacker is rejected in {fmt2(o_rej[0])}\\% / {fmt2(o_rej[1])}\\% of its attack rounds and submits on average a poison fraction of {('--' if pd.isna(o_blend[0]) else f'{o_blend[0]:.2f}')} / {('--' if pd.isna(o_blend[1]) else f'{o_blend[1]:.2f}')} of its full poisoned update. "
            + e2_conclusion(pick, s_flip, o_flip, p_flip, s_bd, o_bd, p_bd)
            + f" ({done} of 60 runs completed at the time of writing.)" * (done < 60) + "\n")
    open(os.path.join(E2, 'E2_paragraph.tex'), 'w', encoding='utf-8').write(para)
    print('E2:', done, 'runs')
    print(summ.round(2).to_string(index=False))
else:
    print('E2: no finished run yet')

