import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

RUNS = os.path.join(ROOT, 'results_r02', 'pilot_scaffold')
OUT = os.path.join(ROOT, 'results_r02', 'calibration')
VARIANTS = ['SCAFFOLD', 'SCAFFOLD-SGD05', 'SCAFFOLD-SGD10', 'SCAFFOLD-SGD20',
            'SCAFFOLD2-SGD01', 'SCAFFOLD2-SGD02', 'SCAFFOLD2-SGD05']
SEEDS = [100, 101]

rows = []
for p in glob.glob(os.path.join(RUNS, 'final_*.json')):
    rec = json.load(open(p))
    c = rec['config']
    val = pd.read_csv(os.path.join(RUNS, f"val_{rec['run_id']}.csv"))
    rows.append(dict(method=c['method'], seed=c['seed'], algo=c['algo'], lr=c['lr'],
                     val_f1=float(val['F1-Score'].iloc[-1]), val_acc=float(val['Accuracy'].iloc[-1]),
                     test_f1=rec['test']['F1-Score'], rounds=c['rounds']))
df = pd.DataFrame(rows)
if not len(df):
    raise SystemExit("no pilot_scaffold runs yet")
table = df.pivot_table(index='method', columns='seed', values='val_f1')
table['mean_val_f1'] = table.mean(axis=1)
print(table.round(2).to_string())
missing = [(m, s) for m in VARIANTS + ['FedAvg'] for s in SEEDS if not ((df.method == m) & (df.seed == s)).any()]
cand = table.loc[[m for m in VARIANTS if m in table.index]]
winner = cand['mean_val_f1'].idxmax()
cfg = df[df.method == winner].iloc[0]
out = dict(method=winner, algo=cfg['algo'], lr=float(cfg['lr']),
           rule='best final validation macro-F1 averaged over pilot seeds 100-101 (50 rounds)',
           mean_val_f1={m: round(float(v), 3) for m, v in cand['mean_val_f1'].items()},
           fedavg_mean_val_f1=round(float(table.loc['FedAvg', 'mean_val_f1']), 3) if 'FedAvg' in table.index else None,
           complete=not missing, missing=[list(m) for m in missing])
if missing:
    print("WARNING: pilot incomplete, scaffold.json NOT written. Missing:", missing)
else:
    tmp = os.path.join(OUT, 'scaffold.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, os.path.join(OUT, 'scaffold.json'))
print("SCAFFOLD variant:", winner, "(complete)" if not missing else "(incomplete)")

