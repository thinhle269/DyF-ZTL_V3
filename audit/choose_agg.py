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
RES = os.path.join(ROOT, 'results_r02')
rows = []
for exp in ('pilot_agg', 'pilot_anchor'):
    for p in glob.glob(os.path.join(RES, exp, 'final_*.json')):
        r = json.load(open(p))
        c = r['config']
        if c['method'] not in ('DyF-ZTL-R', 'DyF-ZTL-RN', 'DyF-ZTL-RV', 'DyF-ZTL-RNV'):
            continue
        m = A.admission_metrics(os.path.join(RES, exp, f"decisions_{r['run_id']}.csv")) or {}
        rows.append(dict(method=c['method'], seed=c['seed'], ratio=c['ratio'], attack=c.get('attack', 'cyclic_flip'),
                         acc=r['test']['Accuracy'], asr=100 * (r.get('test_trigger_asr_cond') or 0),
                         honest_rej=100 * m.get('honest_rejection', np.nan)))
d = pd.DataFrame(rows)
if not len(d):
    sys.exit("no pilot runs yet")

for i, row in d.iterrows():
    if row.ratio == 0 and row.honest_rej != row.honest_rej:
        d.loc[i, 'honest_rej'] = np.nan
d = d.drop_duplicates(['method', 'seed', 'ratio', 'attack'])
print(d.sort_values(['attack', 'ratio', 'method', 'seed']).round(2).to_string(index=False))

CONDS = [('cyclic_flip', 0.0), ('cyclic_flip', 0.5), ('cyclic_flip', 0.8), ('cyclic_flip', 0.9),
         ('targeted_flip', 0.2), ('backdoor', 0.4), ('backdoor_boost', 0.4)]
WEAK = [('cyclic_flip', 0.5), ('cyclic_flip', 0.8), ('cyclic_flip', 0.9), ('targeted_flip', 0.2)]
SEEDS = [100, 101]


def get(m, a, r, s, col):
    x = d[(d.method == m) & (d.attack == a) & (d.ratio == r) & (d.seed == s)][col]
    return float(x.iloc[0]) if len(x) else np.nan


missing = [(m, a, r, s) for m in ('DyF-ZTL-R', 'DyF-ZTL-RN', 'DyF-ZTL-RV', 'DyF-ZTL-RNV') for a, r in CONDS for s in SEEDS
           if np.isnan(get(m, a, r, s, 'acc'))]
verdicts = {}
for v in ('DyF-ZTL-RN', 'DyF-ZTL-RV', 'DyF-ZTL-RNV'):
    gains = [get(v, a, r, s, 'acc') - get('DyF-ZTL-R', a, r, s, 'acc') for a, r in WEAK for s in SEEDS]
    d1 = bool(all(g == g and g >= -0.5 for g in gains) and np.nanmean(gains) >= 2.0)
    clean_gain = [get(v, 'cyclic_flip', 0.0, s, 'acc') - get('DyF-ZTL-R', 'cyclic_flip', 0.0, s, 'acc') for s in SEEDS]
    hr = [get(v, 'cyclic_flip', 0.0, s, 'honest_rej') - get('DyF-ZTL-R', 'cyclic_flip', 0.0, s, 'honest_rej') for s in SEEDS]
    d2 = bool(all(g == g and g >= -0.5 for g in clean_gain) and all(h == h and h <= 5 for h in hr))
    asr_b = [get(v, 'backdoor', 0.4, s, 'asr') - get('DyF-ZTL-R', 'backdoor', 0.4, s, 'asr') for s in SEEDS]
    asr_bb = [get(v, 'backdoor_boost', 0.4, s, 'asr') - get('DyF-ZTL-R', 'backdoor_boost', 0.4, s, 'asr') for s in SEEDS]
    acc_bb = [get(v, 'backdoor_boost', 0.4, s, 'acc') - get('DyF-ZTL-R', 'backdoor_boost', 0.4, s, 'acc') for s in SEEDS]
    d3 = bool(all(x == x and x <= 5 for x in asr_b + asr_bb) and all(x == x and x >= -1 for x in acc_bb))
    verdicts[v] = dict(D1=d1, D2=d2, D3=d3, mean_gain_weak=float(np.nanmean(gains)), min_gain_weak=float(np.nanmin(gains)),
                       clean_gain=float(np.nanmean(clean_gain)), honest_rej_change=float(np.nanmean(hr)),
                       backdoor_asr_change=float(np.nanmean(asr_b)), boost_asr_change=float(np.nanmean(asr_bb)))
passing = [v for v in ('DyF-ZTL-RN', 'DyF-ZTL-RV', 'DyF-ZTL-RNV') if all(verdicts[v][k] for k in ('D1', 'D2', 'D3'))]
decision = ('pilot incomplete' if missing else (f'adopt {passing[0]}' if passing else 'keep v2.1'))
out = dict(decision=decision, passing=passing, verdicts=verdicts, missing=[list(map(str, x)) for x in missing],
           rule='(all on the pilot seeds; "baseline" = v2.1 on the same seed and condition):\n  D1 effectiveness: weak conditions = cyclic 0.5 / 0.8 / 0.9 and targeted flip 0.2 —\n     every (condition, seed): accuracy >= baseline - 0.5; mean gain over the 8 cells >= +2.0 points.\n  D2 no clean cost: clean accuracy >= baseline - 0.5; clean honest rejection <= baseline + 5 points.\n  D3 integrity kept: backdoor 0.4 conditional ASR <= baseline + 5; boosted backdoor 0.4 ASR <= baseline + 5\n     and accuracy >= baseline - 1.\nChoice: among variants passing D1-D3, the simplest in the order N, V, NV. None passes -> keep v2.1.\n    python audit/choose_agg.py -> results_r02/calibration/agg_variant.json')
os.makedirs(os.path.join(RES, 'calibration'), exist_ok=True)
if not missing:
    json.dump(out, open(os.path.join(RES, 'calibration', 'agg_variant.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('decision', 'passing', 'verdicts')}, indent=1))
if missing:
    print('missing:', len(missing), missing[:6])

