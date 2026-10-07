import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import src.preprocessing as prep

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
OUT = os.path.join(ROOT, 'results_r02', 'audit')
DATA = os.path.join(ROOT, 'dataset', 'Train_Test_Windows_10.csv')

prep.load_and_process_data(DATA, num_clients=20, partition_seed=0, split='chrono_class', verbose=False)
feats = prep.LAST_REPORT['features']
raw = pd.read_csv(DATA, low_memory=False)
raw = raw[raw['type'] != 'mitm']
for f in feats + ['ts']:
    raw[f] = pd.to_numeric(raw[f], errors='coerce')
raw = raw.dropna(subset=feats + ['ts'])
raw['is_attack'] = (raw['type'] != 'normal').astype(int)
normal = raw[raw.type == 'normal'].sort_values('ts')
n = len(normal)
tr, te = normal.iloc[:int(0.70 * n)], normal.iloc[int(0.85 * n):]
attack = raw[raw.type != 'normal']
rows = []
for f in feats:
    rho = stats.spearmanr(normal['ts'], normal[f])[0]
    sd = normal[f].std() or 1.0
    shift = (te[f].mean() - tr[f].mean()) / sd

    d_norm, d_att = abs(te[f].mean() - tr[f].mean()) / sd, abs(te[f].mean() - attack[f].mean()) / sd
    auc_lab = roc_auc_score(raw['is_attack'], raw[f]) if raw[f].nunique() > 1 else np.nan
    rows.append(dict(feature=f, spearman_time_normal=round(rho, 3), shift_late_vs_train_normal_sd=round(shift, 2),
                     late_normal_closer_to='attack' if d_att < d_norm else 'train-normal',
                     auc_feature_vs_label=round(max(auc_lab, 1 - auc_lab), 3) if auc_lab == auc_lab else np.nan))
res = pd.DataFrame(rows).sort_values('spearman_time_normal', key=lambda s: -s.abs())
auc_time = roc_auc_score(raw['is_attack'], raw['ts'])
res.to_csv(os.path.join(OUT, 'drift_audit.csv'), index=False)
md = ["# Temporal drift of the model features (R1.5)\n",
      f"Normal records: {n:,} (Apr 2–29); attack records: {len(attack):,} (Apr 23–27 windows).\n",
      f"AUC of the timestamp alone for attack vs normal (random split data): **{max(auc_time, 1 - auc_time):.3f}**.\n",
      f"Features whose late-normal mean is closer to the attack mean than to train-normal: "
      f"**{int((res.late_normal_closer_to == 'attack').sum())} of {len(res)}**.\n",
      res.to_string(index=False)]
open(os.path.join(OUT, 'drift_audit.md'), 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))

