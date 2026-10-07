import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch

torch.set_num_threads(1)
import run_experiments as rx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_r02_repro')
EXP = sys.argv[1]
_orig = rx.expand


def expand(preset, args):
    keep = []
    for job in _orig(preset, args):
        m, s, r, R, d, ex = job
        if m != 'DyF-ZTL-V' or s != 4:
            continue
        if ((ex or {}).get('sim') or {}).get('aggregation') == 'weighted':
            continue
        if EXP == 'cyclic' and r != 0.4:
            continue
        keep.append(job)
    print(f"[repro] {EXP}: {len(keep)} job(s): {[(j[0], j[1], j[2], j[3]) for j in keep]}; cuda available: {torch.cuda.is_available()}; threads: {torch.get_num_threads()}", flush=True)
    return keep


rx.expand = expand
sys.argv = ['run_experiments.py', '--exp', EXP, '--outdir', OUT, '--quiet']
rx.main()

