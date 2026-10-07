import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

import analyze_r02 as A

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
RUNS = os.path.join(ROOT, 'results_r02', 'pilot_anchor')
rows = []
for p in glob.glob(os.path.join(RUNS, 'final_*.json')):
    r = json.load(open(p))
    c = r['config']
    m = A.admission_metrics(os.path.join(RUNS, f"decisions_{r['run_id']}.csv")) or {}
    rows.append(dict(method=c['method'], seed=c['seed'], ratio=c['ratio'], attack=c.get('attack', 'cyclic_flip'),
                     acc=r['test']['Accuracy'], asr=(r.get('test_trigger_asr_cond') or 0) * 100,
                     a2n=(r.get('test_attack_to_normal') or 0) * 100,
                     honest_rej=100 * m.get('honest_rejection', np.nan), mal_rej=100 * m.get('malicious_rejection', np.nan)))
d = pd.DataFrame(rows)
if not len(d):
    sys.exit("no pilot_anchor runs yet")
cond = lambda a, r: (d.attack == a) & (d.ratio == r)
need = [(m, a, r, s) for m in ('DyF-ZTL', 'DyF-ZTL-R') for a, r in (('cyclic_flip', 0.0), ('cyclic_flip', 0.4), ('cyclic_flip', 0.8),
                                                                     ('cyclic_flip', 0.9), ('backdoor', 0.4), ('targeted_flip', 0.2))
        for s in (100, 101)]
missing = [n for n in need if not ((d.method == n[0]) & cond(n[1], n[2]) & (d.seed == n[3])).any()]
print(d.sort_values(['attack', 'ratio', 'method', 'seed']).round(2).to_string(index=False))


def mean(m, a, r, col):
    x = d[(d.method == m) & cond(a, r)][col]
    return float(x.mean()) if len(x) else np.nan


def honest_clean(m):
    x = []
    for p in glob.glob(os.path.join(RUNS, f'final_{m}_s*_p0.0_*.json')):
        rec = json.load(open(p))
        if rec['config']['method'] != m:
            continue
        dd = pd.read_csv(os.path.join(RUNS, f"decisions_{rec['run_id']}.csv"))
        dd = dd[dd.Phase != 'grace'] if 'Phase' in dd else dd
        x.append(100 * (1 - dd.Admitted.mean()))
    return float(np.mean(x)) if x else np.nan


v2, v21 = 'DyF-ZTL', 'DyF-ZTL-R'
c1_vals = d[(d.method == v21) & (d.attack == 'cyclic_flip') & d.ratio.isin([0.8, 0.9])].acc
C1 = bool(len(c1_vals) == 4 and (c1_vals >= 90).all())
hon2, hon21 = honest_clean(v2), honest_clean(v21)
C2 = bool(mean(v21, 'cyclic_flip', 0.0, 'acc') >= mean(v2, 'cyclic_flip', 0.0, 'acc') - 0.5
          and hon21 <= hon2 + 5 and hon21 <= 15)
C3 = bool(mean(v21, 'cyclic_flip', 0.4, 'acc') >= mean(v2, 'cyclic_flip', 0.4, 'acc') - 1
          and mean(v21, 'backdoor', 0.4, 'asr') <= mean(v2, 'backdoor', 0.4, 'asr') + 5
          and mean(v21, 'targeted_flip', 0.2, 'a2n') <= mean(v2, 'targeted_flip', 0.2, 'a2n') + 5)
decision = ('adopt v2.1' if C1 and C2 and C3 else 'keep v2' if not C1 else 'authors decide')
out = dict(C1=C1, C2=C2, C3=C3, decision=decision if not missing else 'pilot incomplete', missing=[list(map(str, m)) for m in missing],
           clean_honest_rejection={v2: hon2, v21: hon21},
           summary={m: {f"{a}@{r}": dict(acc=mean(m, a, r, 'acc'), asr=mean(m, a, r, 'asr'), a2n=mean(m, a, r, 'a2n'),
                                         mal_rej=mean(m, a, r, 'mal_rej'))
                        for a, r in (('cyclic_flip', 0.0), ('cyclic_flip', 0.4), ('cyclic_flip', 0.8), ('cyclic_flip', 0.9),
                                     ('backdoor', 0.4), ('targeted_flip', 0.2))} for m in (v2, v21)})
if not missing:
    json.dump(out, open(os.path.join(ROOT, 'results_r02', 'calibration', 'anchor.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('C1', 'C2', 'C3', 'decision', 'clean_honest_rejection')}, indent=1))

