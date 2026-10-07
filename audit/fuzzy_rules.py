import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import src.models as models
import src.preprocessing as prep

ap = argparse.ArgumentParser()
ap.add_argument('--exp', default='main')
ap.add_argument('--method', default='DyF-ZTL-V')
ap.add_argument('--aggregation', default='uniform', help="'uniform' = Round-2 method (§6c.12); 'weighted' = registered -W runs")
ap.add_argument('--runs', default=os.path.join(ROOT, 'results_r02'))
args = ap.parse_args()
OUT = os.path.join(ROOT, 'results_r02', 'fuzzy_diagnostics')
os.makedirs(OUT, exist_ok=True)

recs = []
for p in glob.glob(os.path.join(args.runs, args.exp, 'final_*.json')):
    r = json.load(open(p))
    if (r['config']['method'] == args.method and r['config'].get('aggregation') == args.aggregation
            and os.path.exists(os.path.join(args.runs, args.exp, f"global_{r['run_id']}.pt"))):
        recs.append(r)
recs.sort(key=lambda r: r['config']['seed'])
if not recs:
    sys.exit(f"no saved {args.method} models in {args.exp}")

rule_rows, approx_rows, centers_by_seed, client_rows = [], [], {}, []
for r in recs:
    seed = r['config']['seed']
    clients, val, test, C, D, classes = prep.load_and_process_data(
        os.path.join(ROOT, 'dataset', 'Train_Test_Windows_10.csv'), num_clients=20, partition_seed=seed,
        split=r['config']['split'], verbose=False)
    rep = prep.LAST_REPORT
    feats, mean, scale = rep['features'], np.array(rep['scaler_mean']), np.array(rep['scaler_scale'])
    sd = torch.load(os.path.join(args.runs, args.exp, f"global_{r['run_id']}.pt"))
    model = models.DynamicFuzzyNet(D, C, legacy='fuzzy.sigma' in sd)
    model.load_state_dict(sd)
    model.eval()
    Xv, yv = val.tensors
    with torch.no_grad():
        lf = model.fuzzy.log_firing(Xv)
        top = lf.argmax(1).numpy()
        pred = model(Xv).argmax(1).numpy()
    mu = model.fuzzy.mu.detach().numpy()
    sig = model.fuzzy.sigma.detach().numpy()
    W = model.consequent.weight.detach().numpy()
    rule_class = W.argmax(0)
    centers_by_seed[seed] = mu.T
    y = yv.numpy()
    for j in range(mu.shape[1]):
        sel = top == j
        cov = sel.mean()
        purity = float((y[sel] == rule_class[j]).mean()) if sel.any() else np.nan
        narrow = np.argsort(sig[:, j])[:3]
        rule_rows.append(dict(seed=seed, rule=j, consequent=classes[rule_class[j]], coverage=round(cov * 100, 2),
                              purity=round(purity * 100, 2) if not np.isnan(purity) else np.nan,
                              antecedents='; '.join(f"{feats[i]} ~ {mu[i, j] * scale[i] + mean[i]:.4g} +- {sig[i, j] * scale[i]:.3g}"
                                                    for i in narrow)))
    for k, cname in enumerate(classes):
        cand = [(row['coverage'] * (row['purity'] or 0), row['rule']) for row in rule_rows
                if row['seed'] == seed and row['consequent'] == cname and row['coverage'] > 0]
        if not cand:
            approx_rows.append(dict(seed=seed, cls=cname, rule=None, text='no rule with this consequent fires first'))
            continue
        j = max(cand)[1]
        narrow = np.argsort(sig[:, j])[:3]
        Xn = Xv.numpy()
        mask = np.all([np.abs(Xn[:, i] - mu[i, j]) <= sig[i, j] for i in narrow], axis=0)
        prec = float((y[mask] == k).mean()) if mask.any() else np.nan
        rec = float(mask[y == k].mean()) if (y == k).any() else np.nan
        lo_obs = Xn.min(0) * scale + mean
        hi_obs = Xn.max(0) * scale + mean
        text = ' AND '.join(f"{feats[i]} in [{max((mu[i, j] - sig[i, j]) * scale[i] + mean[i], lo_obs[i]):.4g}, "
                            f"{min((mu[i, j] + sig[i, j]) * scale[i] + mean[i], hi_obs[i]):.4g}]" for i in narrow)
        approx_rows.append(dict(seed=seed, cls=cname, rule=j, text=f"IF {text} THEN {cname}",
                                support=int(mask.sum()), precision=round(prec * 100, 2) if not np.isnan(prec) else np.nan,
                                recall=round(rec * 100, 2) if not np.isnan(rec) else np.nan,
                                model_recall=round(float((pred[y == k] == k).mean()) * 100, 2)))
    cpath = os.path.join(args.runs, args.exp, f"clients_{r['run_id']}.pt")
    if os.path.exists(cpath):
        csd = torch.load(cpath)
        for cid, s in csd.items():
            key = 'fuzzy.mu'
            drift = float(np.abs(s[key].numpy() - mu).mean())
            hist = np.array(rep['clients'][cid]['hist_pre'], dtype=float)
            client_rows.append(dict(seed=seed, client=cid, mean_center_shift=drift,
                                    normal_share=hist[classes.tolist().index('normal')] / hist.sum(),
                                    n_classes=int((hist > 0).sum()), n=int(hist.sum())))

rules = pd.DataFrame(rule_rows)
SUF = f"{args.exp}_{args.method}{'' if args.aggregation == 'uniform' else '-W'}"
rules.to_csv(os.path.join(OUT, f'rules_{SUF}.csv'), index=False)
approx = pd.DataFrame(approx_rows)
approx.to_csv(os.path.join(OUT, f'approx_rules_{SUF}.csv'), index=False)
stab = []
seeds = sorted(centers_by_seed)
for a in range(len(seeds)):
    for b in range(a + 1, len(seeds)):
        A, B = centers_by_seed[seeds[a]], centers_by_seed[seeds[b]]
        cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(cost)
        ra = rules[(rules.seed == seeds[a])].set_index('rule').consequent
        rb = rules[(rules.seed == seeds[b])].set_index('rule').consequent
        stab.append(dict(seed_a=seeds[a], seed_b=seeds[b], mean_matched_distance=float(cost[ri, ci].mean()),
                         same_consequent=float(np.mean([ra[i] == rb[j] for i, j in zip(ri, ci)]))))
pd.DataFrame(stab).to_csv(os.path.join(OUT, f'rule_stability_{SUF}.csv'), index=False)
if client_rows:
    cl = pd.DataFrame(client_rows)
    cl.to_csv(os.path.join(OUT, f'client_variation_{SUF}.csv'), index=False)
    print("client centre shift vs normal share: Spearman rho =",
          round(cl[['mean_center_shift', 'normal_share']].corr(method='spearman').iloc[0, 1], 3))
print(f"{len(recs)} models; firing-coverage of rules (mean over seeds):")
print(rules.groupby('consequent')[['coverage', 'purity']].mean().round(2).to_string())
print(approx[['seed', 'cls', 'text', 'precision', 'recall']].head(14).to_string(index=False))
if stab:
    print("stability:", pd.DataFrame(stab)[['mean_matched_distance', 'same_consequent']].mean().round(3).to_dict())

