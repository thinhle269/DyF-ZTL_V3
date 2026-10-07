import csv
import datetime
import hashlib
import json
import os
import zipfile

ROOT = 'D:/Samia/Round_02'
RES = os.path.join(ROOT, 'results_r02')
OUT = os.path.join(ROOT, 'Extension', 'E5_release')
os.makedirs(OUT, exist_ok=True)
SKIP_DIRS = {'_invalid', 'smoke', 'logs', 'paper'}
EXTRA_DIRS = ['calibration', 'partitions', 'manifests', 'analysis', 'audit', 'fuzzy_diagnostics', 'systems', 'aggregation']
EXTRA_FILES = ['protocol.md', 'protocol.lock', 'aggregation_spec.md']


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def run_dirs():
    for name in sorted(os.listdir(RES)):
        d = os.path.join(RES, name)
        if os.path.isdir(d) and name not in SKIP_DIRS and name not in EXTRA_DIRS and any(f.startswith('final_') for f in os.listdir(d)):
            yield name, d


files_rows, run_rows = [], []
zpath = os.path.join(OUT, 'DyF-ZTL_R2_per_run_records.zip')
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for preset, d in run_dirs():
        names = sorted(os.listdir(d))
        by_rid = {}
        for f in names:
            if f.startswith('claim_'):
                continue
            prefix, _, rest = f.partition('_')
            rid = rest.rsplit('.', 1)[0]
            by_rid.setdefault(rid, []).append(f)
            full = os.path.join(d, f)
            z.write(full, f'results_r02/{preset}/{f}')
            files_rows.append((f'results_r02/{preset}/{f}', os.path.getsize(full), sha256(full)))
        for rid, fl in by_rid.items():
            fin = next((f for f in fl if f.startswith('final_')), None)
            if fin is None:
                continue
            j = json.load(open(os.path.join(d, fin), encoding='utf-8'))
            cfg = j.get('config', j)
            run_rows.append(dict(run_id=rid, preset=preset, method=cfg.get('method'), seed=cfg.get('seed'), ratio=cfg.get('ratio'),
                                 attack=cfg.get('attack'), rounds=cfg.get('rounds'), files=';'.join(sorted(fl)),
                                 sha256_final=sha256(os.path.join(d, fin)), code_hash=j.get('code_hash'),
                                 test_accuracy=(j.get('test') or {}).get('Accuracy'), device=j.get('device')))
        print(f'  {preset}: {len(by_rid)} runs, {len(names)} files')
    for sub in EXTRA_DIRS:
        d = os.path.join(RES, sub)
        if not os.path.isdir(d):
            continue
        for dirpath, _, fnames in os.walk(d):
            for f in sorted(fnames):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, ROOT).replace(os.sep, '/')
                z.write(full, rel)
                files_rows.append((rel, os.path.getsize(full), sha256(full)))
    for f in EXTRA_FILES:
        full = os.path.join(RES, f)
        if os.path.exists(full):
            z.write(full, f'results_r02/{f}')
            files_rows.append((f'results_r02/{f}', os.path.getsize(full), sha256(full)))

inv = os.path.join(RES, '_invalid')
if os.path.isdir(inv):
    with zipfile.ZipFile(os.path.join(OUT, 'DyF-ZTL_R2_superseded_runs.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dirpath, _, fnames in os.walk(inv):
            for f in sorted(fnames):
                full = os.path.join(dirpath, f)
                z.write(full, os.path.relpath(full, ROOT).replace(os.sep, '/'))

with open(os.path.join(OUT, 'MANIFEST_files.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['path', 'bytes', 'sha256'])
    w.writerows(files_rows)
with open(os.path.join(OUT, 'MANIFEST_runs.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()))
    w.writeheader()
    w.writerows(run_rows)

codes = {}
for r in run_rows:
    codes[r['code_hash']] = codes.get(r['code_hash'], 0) + 1
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
open(os.path.join(OUT, 'VERSION.txt'), 'w', encoding='utf-8').write(
    f"DyF-ZTL Round-2 (method v2.2, run name DyF-ZTL-V) - per-run record archive built {stamp}\n"
    f"runs archived: {len(run_rows)}  (presets: {', '.join(sorted(set(r['preset'] for r in run_rows)))})\n"
    f"files archived: {len(files_rows)}  ({sum(b for _, b, _ in files_rows) / 1e6:.0f} MB uncompressed)\n"
    f"code digests (code_hash of the run manifests -> number of runs): {json.dumps(codes)}\n"
    "suggested release tag for the code repository: v2.2-R2 (the SHA-256 of every src/*.py file is stored in results_r02/manifests/)\n"
    "dataset: ToN-IoT Train_Test_Windows_10.csv (not redistributed; SHA-256 in every run manifest)\n")

readme = f"""# DyF-ZTL Round 2 - per-run record archive (built {stamp})

* `DyF-ZTL_R2_per_run_records.zip` - every per-run file of the {len(run_rows)} runs under `results_r02/<preset>/`
  (final JSON with configuration, test metrics and per-class metrics; `decisions_*.csv` = one row per client and
  round with evidence, reason code, trust and admission; `trust_*.csv` = trust trajectories; `val_*.csv` =
  per-round validation metrics; `cm_*.csv` = test confusion matrix; `global_*.pt` / `clients_*.pt` = final
  model parameters), plus `calibration/` (gate constants per regime and the pre-registered choices),
  `partitions/` (client class histograms per seed), `manifests/` (SHA-256 of code and data, library versions,
  hardware and command line per launch), `protocol.md` with its amendment log, `protocol.lock`, `analysis/`
  (all analysis tables), `audit/` (claim trace, reproduction log, dataset audits), `fuzzy_diagnostics/`,
  `systems/` (emulation records).
* `DyF-ZTL_R2_superseded_runs.zip` - runs superseded by the code fixes of protocol §6c.22 (kept for transparency,
  not used in any table).
* `MANIFEST_runs.csv` - one row per run id with preset, method, seed, ratio, attack, the list of its files and the
  SHA-256 of its final JSON. `MANIFEST_files.csv` - SHA-256 and size of every archived file.
* `VERSION.txt` - code digests and the suggested repository tag.

## Locating the run behind a number of the paper
1. `results_r02/audit/claim_trace.csv` lists every number of the manuscript with the run ids it is computed from.
2. `MANIFEST_runs.csv` (or `results_r02/analysis/runs_flat.csv`) maps a run id to its preset folder and files.
3. The configuration of the run is the `config` block of `final_<run id>.json`; re-executing it with the released
   code in the stated single-threaded CPU environment reproduces every record bit for bit
   (`results_r02/audit/reproduction_log.md`).

The dataset file is not redistributed (ToN-IoT terms); its SHA-256 is recorded in every manifest.
"""
open(os.path.join(OUT, 'README_release.md'), 'w', encoding='utf-8').write(readme)
print(f'archived {len(run_rows)} runs, {len(files_rows)} files -> {zpath} ({os.path.getsize(zpath) / 1e6:.0f} MB)')

