import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

RUNS = os.path.join(ROOT, 'results_r02', 'pilot_horizon')
OUT = os.path.join(ROOT, 'results_r02', 'calibration')
EXPECTED = {(m, r, s) for m in ('FedAvg', 'FLTrust', 'DyF-ZTL') for r in (0.4, 0.8) for s in (100, 101)}

curves, have = {}, set()
for p in glob.glob(os.path.join(RUNS, 'final_*.json')):
    rec = json.load(open(p))
    c = rec['config']
    key = (c['method'], c['ratio'], c['seed'])
    have.add(key)
    curves[key] = pd.read_csv(os.path.join(RUNS, f"val_{rec['run_id']}.csv"))['Accuracy'].reset_index(drop=True)

missing = sorted(EXPECTED - have)
detail, chosen = [], None
for R in range(20, 101, 10):
    worst = 0.0
    for key, acc in curves.items():
        ma = acc.rolling(10).mean()
        worst = max(worst, abs(ma.iloc[R - 1] - ma.iloc[R - 11]))
    detail.append(dict(rounds=R, max_ma_change_pp=round(worst, 3)))
    if chosen is None and worst < 0.2:
        chosen = R
chosen = chosen or 100


forced = bool(missing) and all(d['max_ma_change_pp'] >= 0.2 for d in detail)
out = dict(rounds=chosen, rule='10-round moving average of validation accuracy changes < 0.2 pp for all pilot runs',
           runs=len(curves), missing=[list(m) for m in missing], complete=not missing, forced=forced,
           note=('outcome forced with the finished runs: every candidate R already fails, and the missing runs '
                 'can only increase the maximum' if forced else ''), detail=detail)
os.makedirs(OUT, exist_ok=True)
if missing and not forced:
    print("WARNING: pilot incomplete, horizon.json NOT written. Missing:", missing)
else:
    tmp = os.path.join(OUT, 'horizon.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, os.path.join(OUT, 'horizon.json'))
    if forced:
        print("Outcome forced by the finished runs (missing runs cannot change it):", missing)
print(pd.DataFrame(detail).to_string(index=False))
print("R_attack =", chosen, "(complete)" if not missing else ("(forced)" if forced else "(incomplete)"))

