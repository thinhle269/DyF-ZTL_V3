import glob
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, 'results_r02')
NEW = os.path.join(ROOT, 'results_r02_repro')
OUT = os.path.join(REF, 'audit', 'reproduction_log.md')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

lines = [f"# Independent reproduction log ({time.strftime('%Y-%m-%d %H:%M')})\n",
         "Code: `Submit_DyF/Resource/` copied to `repro_code/` (plus the dataset file), run with `repro_run.py` into `results_r02_repro/` "
         "(a folder that contained nothing but `protocol.lock`). Reference: the stored runs of `results_r02/` with the same run identifier "
         "(hash of the full configuration). Compared: test metrics, per-class metrics, confusion matrix, per-round validation metrics, "
         "admission decisions and trust trajectories.\n",
         "| preset | run id | test metrics | per-class | confusion matrix | validation curve | decisions log | trust log | wall time ref / new (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
ok_all, n = True, 0
for p in sorted(glob.glob(os.path.join(NEW, '*', 'final_*.json'))):
    exp = os.path.basename(os.path.dirname(p))
    rid = os.path.basename(p)[len('final_'):-len('.json')]
    q = os.path.join(REF, exp, f"final_{rid}.json")
    if not os.path.exists(q):
        lines.append(f"| {exp} | {rid} | reference run not found | | | | | | |")
        ok_all = False
        continue
    new, ref = json.load(open(p)), json.load(open(q))
    same_test = new['test'] == ref['test']
    same_pc = new.get('per_class') == ref.get('per_class')

    def same_csv(kind):
        a, b = os.path.join(NEW, exp, f"{kind}_{rid}.csv"), os.path.join(REF, exp, f"{kind}_{rid}.csv")
        if not (os.path.exists(a) and os.path.exists(b)):
            return 'n/a'
        da, db = pd.read_csv(a), pd.read_csv(b)
        try:
            pd.testing.assert_frame_equal(da, db, check_dtype=False, check_exact=False, rtol=0, atol=1e-9)
            return 'identical'
        except AssertionError:
            return 'DIFFERS'
    cells = [same_csv(k) for k in ('cm', 'val', 'decisions', 'trust')]
    ok = same_test and same_pc and all(c in ('identical', 'n/a') for c in cells)
    ok_all &= ok
    n += 1
    lines.append(f"| {exp} | `{rid}` | {'identical' if same_test else 'DIFFERS'} (acc {ref['test']['Accuracy']} / {new['test']['Accuracy']}) | "
                 f"{'identical' if same_pc else 'DIFFERS'} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {ref.get('wall_time_s')} / {new.get('wall_time_s')} |")
    lines.append(f"|  | code hash ref / new | `{ref.get('code_hash', '')[:16]}` / `{new.get('code_hash', '')[:16]}` | dataset sha256 equal: {ref.get('dataset_sha256') == new.get('dataset_sha256')} | | | | | |")
lines.append(f"\n**Result:** {n} configuration(s) re-executed; {'all records identical to the stored runs' if ok_all and n else 'DIFFERENCES FOUND — see table'}.")
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if ok_all and n else 1)

