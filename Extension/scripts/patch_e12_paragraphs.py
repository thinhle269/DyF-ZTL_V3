import ast
import io

p = 'D:/Samia/Round_02/Extension/scripts/analyze_extension.py'
s = io.open(p, encoding='utf-8').read()
B = chr(92)
NL = chr(10)


old = '            + f" ({done} of 90 runs completed at the time of writing.)" * (done < 90) + "\\n")'
assert s.count(old) == 1, s.count(old)
new = ('            + e1_extra(summ)' + NL
       + '            + f" ({done} of 90 runs completed at the time of writing.)" * (done < 90) + "' + B + 'n")')
s = s.replace(old, new)
helper = r'''

def e1_extra(summ):
    def cell(cond, m, col):
        g = summ[(summ.condition == cond) & (summ.method == m)]
        return float(g[col].iloc[0]) if len(g) else float('nan')
    txt = (f" DyF-ZTL (v2.2) keeps {cell('cyclic 0.4', 'DyF-ZTL (v2.2)', 'acc_new'):.2f}\%, {cell('cyclic 0.8', 'DyF-ZTL (v2.2)', 'acc_new'):.2f}\% and {cell('cyclic 0.9', 'DyF-ZTL (v2.2)', 'acc_new'):.2f}\% accuracy at 40, 80 and 90\% attackers on the new seeds"
           f" (FLTrust {cell('cyclic 0.4', 'FLTrust', 'acc_new'):.2f}\%, {cell('cyclic 0.8', 'FLTrust', 'acc_new'):.2f}\% and {cell('cyclic 0.9', 'FLTrust', 'acc_new'):.2f}\%), and the boosted backdoor succeeds in {cell('boosted backdoor 0.4', 'DyF-ZTL (v2.2)', 'asr_cond_new'):.1f}\% of the triggered records against DyF-ZTL versus {cell('boosted backdoor 0.4', 'FLTrust', 'asr_cond_new'):.1f}\% against FLTrust."
           f" SCAFFOLD and FedProx-uniform, which were not part of the cyclic sweep of the main study, behave like the other majority-dependent rules: {cell('cyclic 0.4', 'SCAFFOLD', 'acc_new'):.1f}\%~$\pm$~{cell('cyclic 0.4', 'SCAFFOLD', 'acc_new_sd'):.1f} and {cell('cyclic 0.4', 'FedProx (uniform)', 'acc_new'):.1f}\%~$\pm$~{cell('cyclic 0.4', 'FedProx (uniform)', 'acc_new_sd'):.1f} at 40\% attackers and below 7\% at 80--90\%.")
    return txt.replace(chr(92) + '%', chr(92) + '%')
'''
s = s.replace(NL + '# ------------------------------------------------------------------ E1', helper + NL + '# ------------------------------------------------------------------ E1')


old = '            + "These results replace the earlier restriction of the threat model to attackers that know only the form of the rule; the conclusion is stated in the text with the numbers above, whatever their direction."'
assert s.count(old) == 1, s.count(old)
new = ('            + e2_conclusion(pick, s_flip, o_flip, p_flip, s_bd, o_bd, p_bd)')
s = s.replace(old, new)
helper2 = r'''

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
'''
s = s.replace(NL + '# ------------------------------------------------------------------ E2', helper2 + NL + '# ------------------------------------------------------------------ E2')
ast.parse(s)
io.open(p, 'w', encoding='utf-8', newline=NL).write(s)
print('E1/E2 paragraph templates completed')

