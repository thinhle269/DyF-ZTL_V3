import io
import os

ROOT = 'D:/Samia/Round_02'
CANON = os.path.join(ROOT, 'Extension', 'DyF_ZTL_R2_v22_reconciled')
B = chr(92)
NL = chr(10)
PCT = B + '%'


def load(p):
    return io.open(p, encoding='utf-8').read()


def save(p, s):
    io.open(p, 'w', encoding='utf-8', newline=NL).write(s)


e1 = load(os.path.join(ROOT, 'Extension', 'E1_confirm_seeds_10_14', 'E1_paragraph.tex')).strip()
e2 = load(os.path.join(ROOT, 'Extension', 'E2_full_knowledge', 'E2_paragraph.tex')).strip()


p = os.path.join(CANON, 'src_tex', '70_discussion.tex')
s = load(p)
marker = '%%E1_PLACEHOLDER%% the five-seed replication (seeds 10--14) is inserted here when the extension runs finish'
if marker in s:
    s = s.replace(marker, e1 + NL + NL + '%%TABLE:tab_e1%%' + NL + B + 'FloatBarrier')
    print('E1 inserted into Appendix C')

old = "The tested boosted model-replacement and partial-knowledge validation-aware attacks had low attack success; this does not establish robustness to full-knowledge or unconstrained adaptive attackers."
new = ("The tested boosted model-replacement and rule-form validation-aware attacks had low attack success. Attackers that know the calibrated gates and the trusted-data policy, including a white-box attacker that uses the server's own reference, are admitted in most rounds: with label flipping the poison that fits inside the gates costs 0.3--0.4 points of accuracy, whereas the backdoor payload passes the gates unchanged and reaches the success of the unconstrained backdoor (87--90" + PCT + " at 40" + PCT + " attackers).")
if old in s:
    s = s.replace(old, new)
    print('discussion sentence updated')
save(p, s)


p = os.path.join(CANON, 'src_tex', '60_results.tex')
s = load(p)
if '%%TABLE:tab_e2%%' not in s:
    i = s.index('%%TABLE:tab_attacks%%')
    j = s.index(B + 'end{figure}', i) + len(B + 'end{figure}')
    s = s[:j] + NL + NL + e2 + NL + NL + '%%TABLE:tab_e2%%' + s[j:]
    print('E2 inserted into §6.5')
save(p, s)


p = os.path.join(CANON, 'src_tex', '50_setup.tex')
s = load(p)
row_anchor = "Sleeper & honest for the first 20 rounds, then cyclic flip & nothing & accuracy; time to exclusion " + B + B
if 'Policy-aware flip / backdoor (surrogate)' not in s and row_anchor in s:
    rows = ("Policy-aware flip / backdoor (surrogate) & mixes clean and poisoned updates so that its own estimate of the three pieces of evidence passes the calibrated gates (blend grid 1.0--0.1; clean update if none passes); estimates the server reference on its own clean records & the calibrated gate constants, the evidence rule and the trusted-data policy & accuracy; attack$" + B + "to$\\emph{Normal}; conditional ASR; admission rate; admitted poison fraction " + B + B + NL
            + "Policy-aware flip / backdoor (oracle) & as surrogate, with the server's $D_{" + B + "mathrm{trust}}$ and reference update of the round (white-box server) & everything the server knows & as surrogate " + B + B + NL)
    rows = rows.replace('\\\\emph', B + 'emph')
    s = s.replace(row_anchor, row_anchor + NL + rows.rstrip(NL))
    print('threat-model rows added')
save(p, s)


p = os.path.join(CANON, 'make_response_letter.py')
s = load(p)
E1 = load(os.path.join(ROOT, 'Extension', 'E1_confirm_seeds_10_14', 'e1_paired.csv'))
import pandas as pd
pr = pd.read_csv(os.path.join(ROOT, 'Extension', 'E1_confirm_seeds_10_14', 'e1_paired.csv'))
sm = pd.read_csv(os.path.join(ROOT, 'Extension', 'E1_confirm_seeds_10_14', 'e1_summary.csv'))
e2s = pd.read_csv(os.path.join(ROOT, 'Extension', 'E2_full_knowledge', 'e2_summary.csv'))


def g(cond, m, col):
    r = sm[(sm.condition == cond) & (sm.method == m)]
    return float(r[col].iloc[0])


def q(k, pl, m, r, col):
    x = e2s[(e2s.knowledge == k) & (e2s.payload == pl) & (e2s.method == m) & (e2s.ratio == r)]
    return float(x[col].iloc[0])


clean = pr[(pr.condition == 'clean')].set_index('competitor')
agree = pr.same_sign.dropna()
e1_txt = (f"Five further seeds (10–14) that were never used for calibration, pilots or any design decision were then run with the engine, its gates and every baseline unchanged ({{{{tab:e1_confirm}}}}): "
          f"clean accuracy {g('clean', 'DyF-ZTL (v2.2)', 'acc_new'):.2f} % ± {g('clean', 'DyF-ZTL (v2.2)', 'acc_new_sd'):.2f} (main study {g('clean', 'DyF-ZTL (v2.2)', 'acc_ref'):.2f} %), paired differences of {clean.loc['SCAFFOLD', 'd_acc_new']:+.2f} / {clean.loc['FedProx (uniform)', 'd_acc_new']:+.2f} / {clean.loc['FLTrust', 'd_acc_new']:+.2f} points versus SCAFFOLD / FedProx-uniform / FLTrust (p = {clean.loc['SCAFFOLD', 'p_new']:.3f} / {clean.loc['FedProx (uniform)', 'p_new']:.3f} / {clean.loc['FLTrust', 'p_new']:.3f}), "
          f"{g('cyclic 0.4', 'DyF-ZTL (v2.2)', 'acc_new'):.2f} / {g('cyclic 0.8', 'DyF-ZTL (v2.2)', 'acc_new'):.2f} / {g('cyclic 0.9', 'DyF-ZTL (v2.2)', 'acc_new'):.2f} % accuracy at 40 / 80 / 90 % cyclic attackers (FLTrust {g('cyclic 0.4', 'FLTrust', 'acc_new'):.2f} / {g('cyclic 0.8', 'FLTrust', 'acc_new'):.2f} / {g('cyclic 0.9', 'FLTrust', 'acc_new'):.2f} %) and a boosted-backdoor success of {g('boosted backdoor 0.4', 'DyF-ZTL (v2.2)', 'asr_cond_new'):.1f} % versus {g('boosted backdoor 0.4', 'FLTrust', 'asr_cond_new'):.1f} % for FLTrust. "
          f"All {int(agree.sum())} of the {len(agree)} paired differences that have a counterpart in the main study keep their sign; SCAFFOLD and FedProx-uniform, added to the cyclic conditions here, collapse like the other majority-dependent rules ({g('cyclic 0.4', 'SCAFFOLD', 'acc_new'):.1f} % and {g('cyclic 0.4', 'FedProx (uniform)', 'acc_new'):.1f} % at 40 %, below 7 % at 80–90 %).")
e2_txt = (f"Results ({{{{tab:e2_knowledge}}}}, seeds 0–4, 20 % / 40 % attackers): against label flipping, DyF-ZTL keeps {q('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', 0.2, 'acc'):.2f} / {q('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', 0.4, 'acc'):.2f} % accuracy against the surrogate attacker and {q('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', 0.2, 'acc'):.2f} / {q('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', 0.4, 'acc'):.2f} % against the oracle (rule-form attacker of the paper: {q('form of the rule only (paper)', 'label flip', 'DyF-ZTL (v2.2)', 0.2, 'acc'):.2f} / {q('form of the rule only (paper)', 'label flip', 'DyF-ZTL (v2.2)', 0.4, 'acc'):.2f} %; FLTrust against the surrogate {q('surrogate (own data)', 'label flip', 'FLTrust', 0.2, 'acc'):.2f} / {q('surrogate (own data)', 'label flip', 'FLTrust', 0.4, 'acc'):.2f} %). "
          f"The policy-aware attackers are admitted in {100 - q('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', 0.2, 'attacker_rejection'):.0f}–{100 - q('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', 0.4, 'attacker_rejection'):.0f} % of their rounds (the naive flipping attacker is excluded in 97.7–100 %), but the poison that fits inside the gates is only {q('surrogate (own data)', 'label flip', 'DyF-ZTL (v2.2)', 0.4, 'mean_blend'):.2f}–{q('oracle (white-box server)', 'label flip', 'DyF-ZTL (v2.2)', 0.2, 'mean_blend'):.2f} of the full poisoned update, so accuracy drops by 0.3–0.4 points and attack→Normal rises by about 1 point. "
          f"With the backdoor payload the policy-aware attacker submits its full poisoned update, which already passes the gates, and reaches the success of the unconstrained backdoor: conditional ASR {q('surrogate (own data)', 'backdoor', 'DyF-ZTL (v2.2)', 0.2, 'asr_cond'):.0f} / {q('surrogate (own data)', 'backdoor', 'DyF-ZTL (v2.2)', 0.4, 'asr_cond'):.0f} % (surrogate) and {q('oracle (white-box server)', 'backdoor', 'DyF-ZTL (v2.2)', 0.2, 'asr_cond'):.0f} / {q('oracle (white-box server)', 'backdoor', 'DyF-ZTL (v2.2)', 0.4, 'asr_cond'):.0f} % (oracle) versus {q('form of the rule only (paper)', 'backdoor', 'DyF-ZTL (v2.2)', 0.2, 'asr_cond'):.0f} / {q('form of the rule only (paper)', 'backdoor', 'DyF-ZTL (v2.2)', 0.4, 'asr_cond'):.0f} % for the rule-form attacker and {q('surrogate (own data)', 'backdoor', 'FLTrust', 0.2, 'asr_cond'):.0f} / {q('surrogate (own data)', 'backdoor', 'FLTrust', 0.4, 'asr_cond'):.0f} % for FLTrust against the surrogate. "
          "The manuscript states this as it is: knowing the policy changes the admission outcome, not the damage, for label flipping; for the backdoor it confirms the limitation that the gates do not detect a statistically honest backdoor update, and it shows that the low success of the rule-form attacker came from that attacker's own conservative constraints ({{sec:threats}}, {{sec:limitations}}).")
pairs = [('[E1E2_OVERVIEW]', '({{app:confirm}}, {{tab:e1_confirm}}; {{sec:threats}}, {{tab:e2_knowledge}})'),
         ('[E1_PENDING]', e1_txt), ('[E2_PENDING]', e2_txt),
         ('[E1E2_TABLEROWS]', '| {{tab:e1_confirm}} | replication on five seeds never used for any decision (clean, cyclic 40/80/90 %, boosted backdoor) | R1.3, R1.8 |' + NL + '| {{tab:e2_knowledge}} | attackers that know the calibrated gates and the trusted-data policy (surrogate and white-box oracle) | R1.3, R2.3 |')]
n = 0
for a, b in pairs:
    if a in s:
        s = s.replace(a, b)
        n += 1
save(p, s)
print('letter placeholders filled:', n)

