import glob
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results_r02')
RULES = {'trust_modes': lambda m: m.endswith('-prob'), 'pilot_scaffold': lambda m: m.startswith('SCAFFOLD2'),
         'main': lambda m: m == 'SCAFFOLD2-SGD05',
         'heterogeneity': lambda m: m in ('SCAFFOLD2-SGD05', 'DyF-ZTL-R', 'DyF-ZTL-R-CW'),
         'scaffold_check': lambda m: True}
apply = '--apply' in sys.argv
moved, log = 0, []
for exp, rule in RULES.items():
    d = os.path.join(RES, exp)
    live = []
    for cl in glob.glob(os.path.join(d, 'claim_*')):
        try:
            import psutil
            if psutil.pid_exists(int(open(cl).read().strip() or 0)):
                live.append(cl)
        except Exception:
            live.append(cl)
    if live:
        sys.exit(f"{exp}: runs still in progress ({len(live)} live claim files) - run this after the batch has finished")
    for p in glob.glob(os.path.join(d, 'final_*.json')):
        rec = json.load(open(p))
        if not rule(rec['config']['method']):
            continue
        rid = rec['run_id']
        files = glob.glob(os.path.join(d, f'*{rid}*'))
        dest = os.path.join(RES, '_invalid', exp)
        log.append(f"{exp}/{rid}: {len(files)} files -> _invalid/{exp}/")
        if apply:
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.move(f, os.path.join(dest, os.path.basename(f)))
        moved += 1
sc = os.path.join(RES, 'calibration', 'scaffold.json')
if os.path.exists(sc):
    log.append("calibration/scaffold.json -> scaffold_hybrid_pilot.json")
    if apply:
        shutil.move(sc, os.path.join(RES, 'calibration', 'scaffold_hybrid_pilot.json'))
print('\n'.join(log))
print(f"{moved} runs {'moved' if apply else 'would be moved'}")
if apply:
    with open(os.path.join(RES, '_invalid', 'README.md'), 'a', encoding='utf-8') as fh:
        fh.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} — archived by audit/archive_invalid.py (protocol §6c.22)\n"
                 + '\n'.join('* ' + l for l in log) + '\n')

