import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import src.preprocessing as prep

OUT = os.path.join(ROOT, 'results_r02', 'audit')
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(ROOT, 'dataset', 'Train_Test_Windows_10.csv')


raw = pd.read_csv(CSV, low_memory=False)
raw.columns = raw.columns.str.strip()
rows = [dict(stage='raw', rows=len(raw), **raw['type'].value_counts().to_dict())]
no_mitm = raw[raw['type'] != 'mitm']
rows.append(dict(stage='drop_mitm', rows=len(no_mitm), **no_mitm['type'].value_counts().to_dict()))
X_raw, y_raw, ts, n_dirty = prep._load_clean(CSV)
rows.append(dict(stage='drop_non_numeric', rows=len(y_raw), **y_raw.value_counts().to_dict()))
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = list(le.classes_)
meta_cols = [c for c in raw.columns if c.lower() in ('ts', 'date', 'time', 'host', 'hostname', 'ip', 'src_ip',
                                                     'dst_ip', 'session', 'user', 'device', 'machine')]
for split in ('random', 'chrono_class'):
    Xtr, Xva, Xte, ytr, yva, yte = prep._split(X_raw, y, ts, split)
    for name, yy in (('train', ytr), ('val', yva), ('test', yte)):
        cnt = np.bincount(yy, minlength=len(classes))
        rows.append(dict(stage=f'{split}:{name}', rows=len(yy), **dict(zip(classes, cnt.tolist()))))
pd.DataFrame(rows).fillna(0).to_csv(os.path.join(OUT, 'data_audit.csv'), index=False)


Xtr, Xva, Xte, ytr, yva, yte = prep._split(X_raw, y, ts, 'random')
C = len(classes)
glob = np.bincount(ytr, minlength=C) / len(ytr)


def js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def smote_sizes(counts, n):
    present = counts[counts > 0]
    if len(present) <= 1 or n <= 30:
        return counts.copy(), counts.copy(), 'skipped', 'skipped'
    target = counts.max()
    r1 = counts.copy() if present.min() < 2 else np.where(counts > 0, target, 0)
    r2 = np.where((counts >= 2) & (counts < target), target, counts)
    st1 = 'failed' if present.min() < 2 else 'applied'
    st2 = 'partial' if (counts == 1).any() else 'applied'
    return r1, r2, st1, st2


crow, hrow = [], []
for seed in range(10):
    idx = prep._partition(ytr, 20, 0.5, np.random.RandomState(seed), C)
    for c, ib in enumerate(idx):
        counts = np.bincount(ytr[ib], minlength=C)
        r1, r2, st1, st2 = smote_sizes(counts, len(ib))
        crow.append(dict(seed=seed, client=c, n_original=int(counts.sum()), n_post_r1=int(r1.sum()),
                         n_post_r2=int(r2.sum()), smote_r1=st1, smote_r2=st2,
                         n_classes=int((counts > 0).sum()), singleton_classes=int((counts == 1).sum()),
                         **{f'pre_{k}': int(v) for k, v in zip(classes, counts)}))
        hrow.append(dict(seed=seed, client=c, js_pre=js(counts, glob), js_post_r2=js(r2, glob)))
cl = pd.DataFrame(crow)
cl.to_csv(os.path.join(OUT, 'client_sizes.csv'), index=False)
pd.DataFrame(hrow).to_csv(os.path.join(OUT, 'heterogeneity.csv'), index=False)


feat = X_raw.columns
key = lambda df: pd.util.hash_pandas_object(df.round(6), index=False)
tr_hash = set(key(Xtr[feat]).values)
lk = []
for name, Xp in (('val', Xva), ('test', Xte)):
    h = key(Xp[feat]).values
    lk.append(dict(check=f'exact duplicate of a train row ({name}, all {len(feat)} features)',
                   value=int(np.isin(h, list(tr_hash)).sum()), of=len(h)))

t_all = ts.values
tr_idx, te_idx = Xtr.index.values, Xte.index.values
gaps = []
for c in range(C):
    a = np.sort(t_all[tr_idx[ytr == c]])
    b = t_all[te_idx[yte == c]]
    pos = np.clip(np.searchsorted(a, b), 1, len(a) - 1)
    gaps.append(np.minimum(np.abs(b - a[pos - 1]), np.abs(b - a[pos])))
gaps = np.concatenate(gaps)
lk.append(dict(check='median gap (s) test record -> nearest same-class train record, random split',
               value=float(np.median(gaps)), of=len(gaps)))
lk.append(dict(check='share of test records within 10 s of a same-class train record, random split',
               value=round(float((gaps <= 10).mean()) * 100, 2), of=len(gaps)))
lk.append(dict(check='metadata columns available for host/session grouping',
               value=';'.join(meta_cols) or 'none', of=len(raw.columns)))
pd.DataFrame(lk).to_csv(os.path.join(OUT, 'leakage.csv'), index=False)


tw = pd.DataFrame({'type': y_raw.values, 't': pd.to_datetime(ts.values, unit='s')})
tw.groupby('type')['t'].agg(['min', 'max', 'count']).to_csv(os.path.join(OUT, 'time_windows.csv'))


print(pd.DataFrame(rows).fillna(0).to_string(index=False))
g = cl.groupby('seed')
print("\nclients per seed:", pd.DataFrame(dict(
    n_min=g.n_original.min(), n_max=g.n_original.max(),
    r1_failed=g.smote_r1.apply(lambda s: (s == 'failed').sum()),
    r2_partial=g.smote_r2.apply(lambda s: (s == 'partial').sum()))).to_string())
share = cl.assign(w=cl.n_original / cl.groupby('seed').n_original.transform('sum')).groupby('seed').w.max()
print(f"\nlargest client weight under n_k weighting: mean {share.mean():.3f}, max {share.max():.3f} (uniform = 0.050)")
h = pd.DataFrame(hrow)
print(f"JS divergence to global class mix: pre-SMOTE mean {h.js_pre.mean():.3f}, post-SMOTE mean {h.js_post_r2.mean():.3f}")
print(pd.DataFrame(lk).to_string(index=False))

