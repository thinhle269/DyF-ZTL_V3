import io
import os
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

import src.models as models
import src.preprocessing as prep

OUT = os.path.join(ROOT, 'results_r02', 'fuzzy_diagnostics')
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clients, val, test, C, D, classes = prep.load_and_process_data(
    os.path.join(ROOT, 'dataset', 'Train_Test_Windows_10.csv'), num_clients=20, partition_seed=0, verbose=False)
Xtr = torch.cat([c.tensors[0] for c in clients]).to(dev)
ytr = torch.cat([c.tensors[1] for c in clients]).to(dev)
Xte, yte = (t.to(dev) for t in test.tensors)
buf = io.StringIO()


def report(tag, model):
    with torch.no_grad():
        m = model.fuzzy.log_firing(Xte).max(1).values.cpu().numpy()
        zero_all, denorm = m < -103.3, m < -87.3
        s = model.fuzzy.sigma.detach()
        print(f"[{tag}] all-rules-zero {zero_all.mean()*100:.2f}% | below float32 normal {denorm.mean()*100:.2f}% "
              f"| median max log-firing {np.median(m):.2f} | sigma min {s.min().item():.4f}, "
              f"non-positive {int((s <= 0).sum())}/{s.numel()}, |sigma|<0.05 {int((s.abs() < 0.05).sum())}")
        y = yte.cpu().numpy()
        worst = [(classes[c], zero_all[y == c].mean() * 100) for c in range(C)]
        print("      per class all-rules-zero %: " + ", ".join(f"{n} {v:.2f}" for n, v in worst))


with redirect_stdout(buf):
    z = Xte.abs().cpu()
    print(f"test |z|: 99th percentile {np.percentile(z.numpy(), 99):.1f}, max {z.max().item():.1f}")
    for legacy in (True, False):
        torch.manual_seed(0)
        name = 'R1 layer (unconstrained sigma)' if legacy else 'R2 layer (softplus sigma, log-space)'
        model = models.DynamicFuzzyNet(D, C, legacy=legacy).to(dev)
        report(f"{name} @init", model)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        g = torch.Generator().manual_seed(0)
        for ep in range(5):
            perm = torch.randperm(len(ytr), generator=g).to(dev)
            for i in range(0, len(ytr), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            with torch.no_grad():
                grads_ok = all(torch.isfinite(p).all() for p in model.parameters())
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"{name}: centralized test acc after 5 epochs {acc*100:.2f}%, finite parameters {grads_ok}")
        report(f"{name} @trained", model)
text = buf.getvalue()
print(text)
with open(os.path.join(OUT, 'f01_numerics.txt'), 'w', encoding='utf-8') as fh:
    fh.write(text)

