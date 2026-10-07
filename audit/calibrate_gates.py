import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

CAL = os.path.join(ROOT, 'results_r02', 'calibration')
RUNS = os.path.join(ROOT, 'results_r02', 'calibrate')
os.makedirs(CAL, exist_ok=True)


def condition_key(c):

    alpha, smote = c.get('alpha', 0.5), c.get('smote', True)
    suffix = ('' if alpha == 0.5 else f"-a{'iid' if alpha is None else alpha}") + ('' if smote else '-nosmote') \
        + {'drift_robust': '-dr', 'drift_oracle': '-do'}.get(c.get('features'), '') \
        + ('-N' if c.get('agg_norm') == 'reference' else '') + ('-V' if c.get('virtual_ref') else '') \
        + ('-R' if (c.get('trust_params') or {}).get('functional_ref') == 'reference' else '') \
        + ('-U' if c.get('aggregation') == 'uniform' else '')
    if c.get('arch') == 'deep':
        return 'mlp' + suffix
    if c.get('trust_condition'):
        return c['trust_condition'] + suffix
    if c.get('trust_val_size'):
        return f"size{c['trust_val_size']}" + suffix
    if c.get('split') == 'chrono_class':
        return 'chrono' + suffix
    return 'default' + suffix


def dump_atomic(obj, path):

    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def gates(ev, q_strong=0.005, q_mild=0.05):
    return dict(rho_s=float(ev.Rho.quantile(q_strong)), rho_m=float(ev.Rho.quantile(q_mild)),
                c_s=float(ev.Cos.quantile(q_strong)), c_m=float(ev.Cos.quantile(q_mild)),
                m_s=float(ev.MRatio.quantile(1 - q_strong)))


groups = {}
for p in glob.glob(os.path.join(RUNS, 'final_*.json')):
    rec = json.load(open(p))
    dec = os.path.join(RUNS, f"decisions_{rec['run_id']}.csv")
    if os.path.exists(dec):
        groups.setdefault(condition_key(rec['config']), []).append((rec['config']['seed'], rec['run_id'], dec))

summary = []
for key, runs in sorted(groups.items()):
    ev = pd.concat([pd.read_csv(d) for _, _, d in runs], ignore_index=True)
    ev = ev[ev.Malicious == 0]
    g = gates(ev)
    meta = dict(condition=key, seeds=sorted(s for s, _, _ in runs), runs=[r for _, r, _ in runs],
                n_client_rounds=int(len(ev)), quantiles=dict(strong=0.005, mild=0.05),
                complete=len(runs) == 3)
    dump_atomic(dict(g, **meta), os.path.join(CAL, f'gates_{key}.json'))
    if key in ('default', 'default-U', 'default-R-U', 'default-V-R-U'):
        suffix = key[len('default'):]
        for name, qs, qm in (('q_strong0.1', 0.001, 0.05), ('q_strong1', 0.01, 0.05),
                             ('q_mild2', 0.005, 0.02), ('q_mild10', 0.005, 0.10)):
            gv = gates(ev, qs, qm)
            dump_atomic(dict(gv, **dict(meta, quantiles=dict(strong=qs, mild=qm), condition=name + suffix)),
                        os.path.join(CAL, f'gates_{name}{suffix}.json'))
    row = dict(condition=key, runs=len(runs), client_rounds=len(ev), **{k: round(v, 4) for k, v in g.items()})
    for col in ('Rho', 'Cos', 'MRatio'):
        row.update({f'{col}_p50': round(ev[col].median(), 4), f'{col}_min': round(ev[col].min(), 4),
                    f'{col}_max': round(ev[col].max(), 4)})
    summary.append(row)

pd.DataFrame(summary).to_csv(os.path.join(CAL, 'evidence_summary.csv'), index=False)
print(pd.DataFrame(summary)[['condition', 'runs', 'client_rounds', 'rho_s', 'rho_m', 'c_s', 'c_m', 'm_s']].to_string(index=False))
if not summary:
    sys.exit("no calibration runs found")

