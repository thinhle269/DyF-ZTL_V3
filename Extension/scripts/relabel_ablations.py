import io
import os

ROOT = 'D:/Samia/Round_02'
CANON = os.path.join(ROOT, 'Extension', 'DyF_ZTL_R2_v22_reconciled')
B = chr(92)
NL = chr(10)


def load(p):
    return io.open(p, encoding='utf-8').read()


def save(p, s):
    io.open(p, 'w', encoding='utf-8', newline=NL).write(s)


def swap(s, pairs, where):
    done = 0
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new)
            done += 1
    print(f'  {where}: {done}/{len(pairs)} replacements applied')
    return s


p = os.path.join(ROOT, 'audit', 'make_paper_assets.py')
s = load(p)
s = swap(s, [
    ("'DyF-ZTL-R': 'DyF-ZTL, engine v2.1'", "'DyF-ZTL-R': 'Ablation C (no reference update)'"),
    ("'DyF-ZTL': 'DyF-ZTL, engine v2'", "'DyF-ZTL': 'Ablation B (global-model anchor)'"),
    ("'DyF-ZTL-v1': 'DyF-ZTL, Round-1 engine (v1)'", "'DyF-ZTL-v1': 'Ablation A (accuracy gates, Round-1 design)'"),
    ("'DyF-ZTL-v1-W': 'Round-1 engine, $|D_k|$-weighted'", "'DyF-ZTL-v1-W': 'Ablation A, $|D_k|$-weighted'"),
    ("'DyF-ZTL-v1': 'Round-1 engine'", "'DyF-ZTL-v1': 'Ablation A (acc. gates)'"),
    ("('DyF-ZTL-v1', 'DyF-ZTL-v1-W', 'DyF-ZTL, Round-1 engine')", "('DyF-ZTL-v1', 'DyF-ZTL-v1-W', 'Ablation A (accuracy gates)')"),
    ("weight share is its normalised similarity weight; v1 = Round-1 engine.", "weight share is its normalised similarity weight; ablation A = accuracy-gated admission (Round-1 design)."),
    ("'v1" + B * 4 + "honest rej.'", "'Abl.~A" + B * 4 + "honest rej.'"),
    ("('DyF-ZTL-v1', 'Round-1 engine: absolute accuracy gates, uniform averaging')", "('DyF-ZTL-v1', 'Ablation A: absolute accuracy gates (0.70/0.85), grace period, no server reference (Round-1 design)')"),
    ("('DyF-ZTL', 'v2: dual evidence anchored to the global model, calibrated gates')", "('DyF-ZTL', 'Ablation B: dual evidence anchored to the current global model, calibrated gates')"),
    ("('DyF-ZTL-R', 'v2.1: functional evidence anchored to the server reference $W_{" + B * 2 + "mathrm{ref}}$')", "('DyF-ZTL-R', 'Ablation C: server-reference anchor $W_{" + B * 2 + "mathrm{ref}}$, no reference update in the aggregate')"),
    ("('DyF-ZTL-V', '" + B * 2 + "textbf{v2.2 (proposed): v2.1 + server reference update in the aggregate}')", "('DyF-ZTL-V', '" + B * 2 + "textbf{DyF-ZTL (v2.2): server-reference anchor + reference update in the aggregate}')"),
    ("('DyF-ZTL-V-F', 'v2.2, functional evidence only'), ('DyF-ZTL-V-G', 'v2.2, geometric evidence only')", "('DyF-ZTL-V-F', 'DyF-ZTL, functional evidence only'), ('DyF-ZTL-V-G', 'DyF-ZTL, geometric evidence only')"),
    ("('DeepTrust-V', 'v2.2 engine with MLP clients (w/o fuzzy layer)')", "('DeepTrust-V', 'DyF-ZTL engine with MLP clients (w/o fuzzy layer)')"),
], 'make_paper_assets.py')
save(p, s)


p = os.path.join(ROOT, 'Extension', 'scripts', 'confirm_seeds_2_9.py')
s = load(p)
s = swap(s, [("'DyF-ZTL-v1': 'Round-1 engine'", "'DyF-ZTL-v1': 'Ablation A (acc. gates)'"),
             ("of DyF-ZTL (v2.2), FLTrust and the Round-1 engine per compromised-client ratio", "of DyF-ZTL (v2.2), FLTrust and ablation A (accuracy-gated admission) per compromised-client ratio"),
             ("The candidate engines v2.1 and v2.2 were motivated by observations on the evaluation seeds 0 and 1 of the earlier engine versions",
              "The two amendments that led to the final engine (the server-reference anchor and the reference update in the aggregate, Section~" + B + "ref{sec:engine_evolution}) were motivated by observations on the evaluation seeds 0 and 1 of ablations B and C")], 'confirm_seeds_2_9.py')
save(p, s)
p = os.path.join(ROOT, 'audit', 'claim_trace.py')
s = load(p)
s = swap(s, [("'Round-1 engine': 'DyF-ZTL-v1'", "'Ablation A (acc. gates)': 'DyF-ZTL-v1'"),
             ("'Round-1 engine (seeds 2--9)': 'DyF-ZTL-v1'", "'Ablation A (acc. gates) (seeds 2--9)': 'DyF-ZTL-v1'"),
             ("of `Submit_DyF/DyF-ZTL_R2.pdf`", "of `Extension/DyF_ZTL_R2_v22_reconciled/DyF-ZTL_R2.pdf`")], 'claim_trace.py')
save(p, s)


PCT = B + '%'
tex = {
    '40_method.tex': [(
        "The engine described above is the fourth version of the admission mechanism, and the paper reports the earlier versions as ablations. The Round-1 engine (v1) used absolute validation-accuracy gates (0.70/0.85) with an effective three-round grace period. The initial Round-2 protocol replaced it with a dual-evidence engine whose functional anchor was the current global model (v2). After preliminary results on evaluation seeds 0--1 revealed failure under high cyclic-flipping ratios, a pilot on reserved seeds 100--101 compared v2 with a server-reference functional anchor (v2.1). The authors adopted v2.1 despite one of the original adoption criteria not being met; the deviation and its observed results are recorded in the protocol (amendments 6c.17--18). Preliminary evaluation of v2.1 on seeds 0--1 then revealed a majority-class failure when few clients were admitted. Three aggregation remedies were specified before a further reserved-seed pilot; that pilot selected inclusion of the server reference update (v2.2, Eq.~" + B + "ref{eq:agg}). The design was frozen before the final v2.2 matrix was run,",
        "The engine described above is the final design, and the paper reports three ablations that correspond to the stages of its development: ablation A uses absolute validation-accuracy gates (0.70/0.85) with an effective three-round grace period and no server reference (the Round-1 design); ablation B is the dual-evidence engine registered in the initial Round-2 protocol, with the functional evidence anchored to the current global model; ablation C anchors the functional evidence to the server reference but does not add the reference update to the aggregate. After preliminary results on evaluation seeds 0--1 revealed that B fails under high cyclic-flipping ratios, a pilot on reserved seeds 100--101 compared B with C. The authors adopted C despite one of the original adoption criteria not being met; the deviation and its observed results are recorded in the protocol (amendments 6c.17--18). Preliminary evaluation of C on seeds 0--1 then revealed a majority-class failure when few clients were admitted. Three aggregation remedies were specified before a further reserved-seed pilot; that pilot selected the inclusion of the server reference update (Eq.~" + B + "ref{eq:agg}), which completes the final engine. The design was frozen before the final matrix was run,")],
    '50_setup.tex': [("the v2.1 adoption and v2.2 selection are disclosed in Section~", "the adoption of the server-reference anchor and the selection of the reference update are disclosed in Section~"),
                     ("The final v2.2 design was then frozen before its full evaluation matrix.", "The final design was then frozen before its full evaluation matrix."),
                     ("the three earlier engine versions of Section~", "the three engine ablations (A--C) of Section~")],
    '60_results.tex': [("(Round-1 engine: 22.0" + PCT + ")", "(ablation A, the accuracy-gated Round-1 design: 22.0" + PCT + ")"),
                       ("The Round-1 engine reaches the same accuracy (96.54" + PCT + ") at twice the honest rejection rate.", "Ablation A reaches the same accuracy (96.54" + PCT + ") at twice the honest rejection rate."),
                       ("The Round-1 engine stays within one point of the new engine in accuracy", "Ablation A (accuracy gates) stays within one point of the final engine in accuracy"),
                       ("The Round-1 engine excludes 20--28" + PCT + " of the honest client-rounds, twice the rate of the new engine", "Ablation A excludes 20--28" + PCT + " of the honest client-rounds, twice the rate of the final engine"),
                       ("compares the four engine versions, the two single-evidence ablations and the architecture controls", "compares the final engine with its three ablations (A--C), the two single-evidence ablations and the architecture controls"),
                       ("The Round-1 engine was already robust to cyclic flipping but excluded 22", "Ablation A (accuracy gates) was already robust to cyclic flipping but excluded 22"),
                       ("Engine v2 (dual evidence anchored to the current global model) halved the honest rejection rate", "Ablation B (dual evidence anchored to the current global model) halved the honest rejection rate"),
                       ("anchoring to the server reference (v2.1) restored 95.7", "anchoring to the server reference (ablation C) restored 95.7"),
                       ("Adding the reference update to the aggregate (v2.2) raised the accuracy", "Adding the reference update to the aggregate (the final engine) raised the accuracy"),
                       ("The unboosted backdoor is not stopped by any version (68--93", "The unboosted backdoor is not stopped by any variant (68--93")],
    '70_discussion.tex': [("The final engine is the fourth version. Preliminary results on evaluation seeds 0--1 informed the subsequent v2.1 and v2.2 pilots, and v2.1 was adopted despite one original pilot criterion not being met.",
                           "The final engine was reached in two amendments of the registered design. Preliminary results on evaluation seeds 0--1 informed the pilots that introduced the server-reference anchor and the reference update, and the server-reference anchor (ablation C) was adopted despite one original pilot criterion not being met."),
                          ("The candidate engines v2.1 and v2.2 were motivated by observations on the evaluation seeds 0 and 1 of the earlier engine versions",
                           "The two amendments that led to the final engine (the server-reference anchor and the reference update in the aggregate, Section~" + B + "ref{sec:engine_evolution}) were motivated by observations on the evaluation seeds 0 and 1 of ablations B and C")],
}
for f, pairs in tex.items():
    p = os.path.join(CANON, 'src_tex', f)
    if os.path.exists(p):
        save(p, swap(load(p), pairs, f))


p = os.path.join(CANON, 'make_response_letter.py')
if os.path.exists(p):
    s = load(p)
    s = swap(s, [("we now disclose where preliminary evaluation-seed observations informed v2.1 and v2.2.", "we now disclose where preliminary evaluation-seed observations informed the two amendments of the engine (server-reference anchor; reference update in the aggregate)."),
                 ("The initial Round-2 protocol preceded evaluation of the v2 engine.", "The initial Round-2 protocol preceded the evaluation of the registered engine (ablation B in the paper)."),
                 ("Reserved-seed pilots informed the remedies, but v2.1 was adopted despite one original pilot criterion not being met; a subsequent specified pilot selected server-anchored aggregation for v2.2.",
                  "Reserved-seed pilots informed the remedies, but the server-reference anchor (ablation C) was adopted despite one original pilot criterion not being met; a subsequent specified pilot selected the reference update in the aggregate, which completes the final engine."),
                 ('the earlier engine (now "Round-1 engine (v1)") is kept as an ablation in {{tab:engine}}', 'the earlier engine (now "ablation A: accuracy-gated admission") is kept as an ablation in {{tab:engine}}'),
                 ("of DyF-ZTL against FLTrust and the Round-1 engine at every ratio.", "of DyF-ZTL against FLTrust and ablation A at every ratio."),
                 ("The same ablation is given for the Round-1 engine and for the DFNN without engine.", "The same ablation is given for ablation A (accuracy gates) and for the DFNN without engine."),
                 ("FLTrust and the Round-1 engine are reported alongside.", "FLTrust and ablation A are reported alongside."),
                 ("preliminary evaluation results on seeds 0–1 helped motivate v2.1 and v2.2, as disclosed above and in the protocol.", "preliminary evaluation results on seeds 0–1 helped motivate the two amendments of the engine, as disclosed above and in the protocol."),
                 ("v2.2 reaches 96.4 ± 0.9 % at 80 % cyclic attackers over ten seeds versus v2.1 at 95.7 ± 1.1 % over four seeds", "the final engine reaches 96.4 ± 0.9 % at 80 % cyclic attackers over ten seeds versus ablation C (no reference update) at 95.7 ± 1.1 % over four seeds"),
                 ("| {{tab:engine}} | engine evolution (v1 → v2.2) and evidence ablation | R2.1 |", "| {{tab:engine}} | engine ablations A–C (development stages) and evidence ablation | R2.1 |"),
                 ("The candidate engines v2.1 and v2.2 were motivated", "The two amendments of the engine were motivated")], 'make_response_letter.py')
    save(p, s)
print('relabelling done')

