import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import src.models as models
import src.preprocessing as prep

torch.set_num_threads(4)
OUT = os.path.join(ROOT, 'results_r02', 'calibration')
os.makedirs(OUT, exist_ok=True)
NORMAL = 3
clients, val, test, C, D, classes = prep.load_and_process_data(
    os.path.join(ROOT, 'dataset', 'Train_Test_Windows_10.csv'), num_clients=20, partition_seed=100, verbose=False)
Xtr = torch.cat([c.tensors[0] for c in clients])
ytr = torch.cat([c.tensors[1] for c in clients])
Xv, yv = val.tensors
att = yv != NORMAL


def train(X, y, arch, seed=0, epochs=5):
    torch.manual_seed(seed)
    m = models.DeepNet(D, C) if arch == 'mlp' else models.DynamicFuzzyNet(D, C)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            F.cross_entropy(m(X[b]), y[b]).backward()
            opt.step()
    m.eval()
    return m


def pred(m, X):
    with torch.no_grad():
        return m(X).argmax(1)


def trig(X, feats, vals):
    Xt = X.clone()
    for f, v in zip(feats, vals):
        Xt[:, f] = v
    return Xt


clean = {a: train(Xtr, ytr, a) for a in ('mlp', 'dfnn')}
base = {a: float((pred(m, Xv[att]) == NORMAL).float().mean()) for a, m in clean.items()}
rows = []
feats_pool = list(range(D))
for feats in itertools.combinations(feats_pool, 2):
    for vals in itertools.product((-2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0), repeat=2):
        r = dict(features=feats, values=vals)
        for a, m in clean.items():
            r[f'clean_asr_{a}'] = float((pred(m, trig(Xv[att], feats, vals)) == NORMAL).float().mean())
        rows.append(r)
cand = pd.DataFrame(rows)
cand['clean_asr_max'] = cand[['clean_asr_mlp', 'clean_asr_dfnn']].max(axis=1)
cand = cand.sort_values('clean_asr_max').reset_index(drop=True)


learn = []
for _, r in cand.head(5).iterrows():
    feats, vals = r['features'], r['values']
    g = np.random.RandomState(0)
    idx = torch.as_tensor(g.choice(len(ytr), int(0.1 * len(ytr)), replace=False))
    Xp, yp = Xtr.clone(), ytr.clone()
    Xp[idx] = trig(Xtr[idx], feats, vals)
    yp[idx] = NORMAL
    res = dict(features=feats, values=vals, clean_asr_max=r['clean_asr_max'])
    for a in ('mlp', 'dfnn'):
        m = train(Xp, yp, a)
        res[f'poisoned_asr_{a}'] = float((pred(m, trig(Xv[att], feats, vals)) == NORMAL).float().mean())
        res[f'poisoned_clean_acc_{a}'] = float((pred(m, Xv) == yv).float().mean())
    learn.append(res)
learn = pd.DataFrame(learn)
learn['score'] = learn[['poisoned_asr_mlp', 'poisoned_asr_dfnn']].min(axis=1) - learn['clean_asr_max']
best = learn.sort_values('score', ascending=False).iloc[0]

cand.to_csv(os.path.join(OUT, 'trigger_candidates.csv'), index=False)
learn.to_csv(os.path.join(OUT, 'trigger_learnability.csv'), index=False)
spec = dict(features=[int(f) for f in best['features']], values=[float(v) for v in best['values']],
            feature_names=[str(prep.LAST_REPORT['features'][f]) for f in best['features']],
            baseline_normal_rate_on_attacks=base, clean_asr_max=float(best['clean_asr_max']),
            poisoned_asr=dict(mlp=float(best['poisoned_asr_mlp']), dfnn=float(best['poisoned_asr_dfnn'])),
            poisoned_clean_acc=dict(mlp=float(best['poisoned_clean_acc_mlp']), dfnn=float(best['poisoned_clean_acc_dfnn'])),
            data='train split (partition seed 100) and validation split only; test split not used',
            rule=f'min over {len(cand)} candidates (all feature pairs x values in +-1..3 std) of the max clean-model '
                 'ASR (MLP, DFNN); among the 5 best, maximize min poisoned ASR minus clean ASR')
with open(os.path.join(OUT, 'trigger.json'), 'w') as fh:
    json.dump(spec, fh, indent=1)
print("baseline P(pred normal | attack):", {k: round(v, 4) for k, v in base.items()})
print(cand.head(8).to_string())
print(learn.to_string())
print("CHOSEN:", json.dumps(spec, indent=1))

