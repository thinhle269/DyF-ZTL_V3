import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import queue as queue_mod
import subprocess
import sys
import threading
import time
import traceback
from time import monotonic

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HOOKS = {'calibrate': 'audit/calibrate_gates.py', 'pilot_scaffold': 'audit/choose_scaffold.py',
         'pilot_horizon': 'audit/choose_horizon.py', 'pilot_anchor': 'audit/choose_anchor.py',
         'pilot_agg': 'audit/choose_agg.py'}
FIRST = ('calibrate', 'pilot_scaffold', 'pilot_horizon', 'pilot_uniform', 'pilot_anchor', 'pilot_agg')


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True


def active_runs(outdir):

    import glob
    n = 0
    for p in glob.glob(os.path.join(outdir, '*', 'claim_*')):
        try:
            with open(p) as fh:
                pid = int(fh.read().strip() or 0)
            n += _pid_alive(pid)
        except (OSError, ValueError):
            pass
    return n


def lane_main(name, device, job_q, status_q, cfg, manifest):
    os.makedirs(os.path.join(cfg['outdir'], 'logs'), exist_ok=True)
    log = open(os.path.join(cfg['outdir'], 'logs', f"lane_{cfg.get('launcher', 'x')}_{name}.log"), 'a',
               buffering=1, encoding='utf-8')
    sys.stdout = sys.stderr = log
    import random
    import torch
    torch.set_num_threads(1)
    import run_experiments as rx
    print(f"\n===== lane {name} ({device}) started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    while True:
        job = job_q.get()
        if job is None:
            status_q.put(('exit', name))
            return
        jid, (exp, method, seed, ratio, rounds, data, epochs, ex) = job
        status_q.put(('take', name, jid))
        args = argparse.Namespace(clients=cfg['clients'], epochs=epochs, batch=cfg['batch'],
                                  aggregation=cfg['aggregation'], weight_basis=cfg['weight_basis'],
                                  outdir=cfg['outdir'], quiet=True, device=device)
        tag = ','.join(f"{k}={v}" for k, v in sorted((ex.get('sim') or {}).items()))
        label = f"{exp}:{method} s{seed} p{ratio}" + (f" [{tag}]" if tag else "")
        if cfg.get('max_active'):

            waited = False
            while active_runs(cfg['outdir']) >= cfg['max_active']:
                if not waited:
                    status_q.put(('wait', name, label))
                    waited = True
                time.sleep(20 + random.random() * 20)
        status_q.put(('start', name, label))
        per_round = cfg['clients'] + 1

        def on_progress(done, total, stage, _name=name):
            status_q.put(('progress', _name, done, total, f"vong {min(done // per_round + 1, rounds)}/{rounds}"
                          if stage == 'training' else f"vong {done // per_round}/{rounds}"))
        try:
            rec = rx.run_one(exp, method, seed, ratio, rounds, data, args, manifest, progress_callback=on_progress,
                             trust_params=ex.get('trust'), sim_kwargs=ex.get('sim'), gate_set=ex.get('gate_set'))
            if rec is None:
                status_q.put(('skip', name, label, jid))
            else:
                status_q.put(('done', name, label, rec['test']['Accuracy'], rec['test']['F1-Score'],
                              rec['wall_time_s'], jid, exp))
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            status_q.put(('error', name, label, tb.strip().splitlines()[-1], jid))


class HookRunner(threading.Thread):


    def __init__(self):
        super().__init__(daemon=True)
        self.pending, self.lock, self.event = [], threading.Lock(), threading.Event()
        self.last, self.changed, self.busy = '-', threading.Event(), False

    def request(self, script):
        with self.lock:
            if script not in self.pending:
                self.pending.append(script)
        self.event.set()

    def run(self):
        while True:
            self.event.wait()
            with self.lock:
                script = self.pending.pop(0) if self.pending else None
                if not self.pending:
                    self.event.clear()
            if script is None:
                continue
            self.busy = True
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT, capture_output=True,
                                   text=True, timeout=3600)
                ok = 'ok' if r.returncode == 0 else f'LOI exit {r.returncode}: {(r.stderr or r.stdout).strip()[-120:]}'
            except Exception as e:
                ok = f'LOI {e}'
            self.last = f"{time.strftime('%H:%M')} {os.path.basename(script)} ({time.time() - t0:.0f}s): {ok}"
            self.busy = False
            self.changed.set()


def main():
    import run_experiments as rx
    ap = argparse.ArgumentParser(description="DyF-ZTL parallel runner", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', nargs='+', required=True, choices=list(rx.PRESETS))
    ap.add_argument('--cpu-lanes', type=int, default=10)
    ap.add_argument('--gpu-lanes', type=int, default=0,
                    help='default 0: CPU and GPU give slightly different floating-point results')
    ap.add_argument('--only', nargs='+',
                    help="run only these methods; 'METHOD' applies to every preset, 'PRESET:METHOD' to that preset only")
    ap.add_argument('--skip', nargs='+', default=[], help='never run these methods')
    ap.add_argument('--seed-major', dest='seed_major', action='store_true',
                    help="queue order: calibration/pilots, then 'main', then all other presets interleaved seed by seed")
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--rounds', type=int)
    ap.add_argument('--clients', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--aggregation', choices=['weighted', 'uniform'], default='weighted')
    ap.add_argument('--weight-basis', dest='weight_basis', choices=['original', 'post_smote'], default='original')
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results_r02'))
    ap.add_argument('--refresh', type=float, default=15.0, help='seconds between dashboard frames')
    ap.add_argument('--clear', action='store_true', help='redraw the dashboard in place (ANSI)')
    ap.add_argument('--allow-provisional', action='store_true')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true', help='print the queue and exit')
    ap.add_argument('--log', help='append every dashboard frame to this file (UTF-8)')
    ap.add_argument('--max-active', dest='max_active', type=int,
                    help='start a run only while fewer than this many runs (from any launcher) are in progress')
    args = ap.parse_args()

    import torch
    from src.progress import render_lanes


    log_fh = open(args.log, 'a', encoding='utf-8', buffering=1) if args.log else None
    console_q = queue_mod.Queue(maxsize=1)

    def console_writer():
        while True:
            text = console_q.get()
            try:
                print(text, flush=True)
            except Exception:
                pass

    threading.Thread(target=console_writer, daemon=True).start()

    def show(text, wait=False):
        if log_fh:
            log_fh.write(text + '\n')
            log_fh.flush()
        try:
            if wait:
                console_q.put(text, timeout=5)
            else:
                console_q.put_nowait(text)
        except queue_mod.Full:
            pass


    locked = os.path.exists(os.path.join(args.outdir, 'protocol.lock'))
    only_all = [o for o in (args.only or []) if ':' not in o]
    only_per = {}
    for o in (args.only or []):
        if ':' in o:
            p_, m_ = o.split(':', 1)
            only_per.setdefault(p_, []).append(m_)
    known = set(rx.METHODS) | {rx.SCAFFOLD_MAIN}
    unknown = [m for m in only_all + [m for v in only_per.values() for m in v] + args.skip if m not in known]
    unknown += [p for p in only_per if p not in args.exp]
    if unknown:
        sys.exit(f"unknown methods/presets in --only/--skip: {unknown}")
    jobs = []
    for exp in args.exp:
        preset = rx.PRESETS[exp]
        if preset['provisional'] and not locked and not args.allow_provisional:
            sys.exit(f"'{exp}' is provisional until {args.outdir}/protocol.lock exists (or use --allow-provisional).")
        ns = argparse.Namespace(seeds=args.seeds, rounds=args.rounds, methods=None, ratios=None)
        epochs = preset.get('epochs', 5)
        keep = only_per.get(exp, []) + only_all
        new = [(exp, m, s, r, R, d, epochs, ex) for (m, s, r, R, d, ex) in rx.expand(preset, ns)
               if (not keep or m in keep) and m not in args.skip]
        if any(j[4] is None for j in new):
            sys.exit(f"'{exp}' needs R_attack: run pilot_horizon and audit/choose_horizon.py first.")
        jobs += new
    if args.seed_major:
        def key(t):
            i, j = t
            group = 0 if j[0] in FIRST else (1 if j[0] == 'main' else 2)
            return (group, j[2] if group == 2 else 0, i)
        jobs = [j for _, j in sorted(enumerate(jobs), key=key)]

    def classify(job):

        exp, method, seed, ratio, rounds, data, epochs, ex = job
        ns = argparse.Namespace(clients=args.clients, epochs=epochs, batch=args.batch,
                                aggregation=args.aggregation, weight_basis=args.weight_basis)
        try:
            rid = rx.run_id_of(rx.run_config(exp, method, seed, ratio, rounds, data, ns, ex.get('trust'),
                                             ex.get('sim'), ex.get('gate_set')))
        except FileNotFoundError as e:
            return 'blocked', str(e).split(';')[0].split(' in ')[0]
        return ('done' if os.path.exists(os.path.join(args.outdir, exp, f"final_{rid}.json")) else 'ready'), rid

    status, reason = [], {}
    for jid, j in enumerate(jobs):
        st, info = classify(j)
        status.append(st)
        if st == 'blocked':
            reason[jid] = info
    n_total = len(jobs)
    n_pre_done = status.count('done')
    todo = n_total - n_pre_done
    n_gpu = args.gpu_lanes if torch.cuda.is_available() else 0
    lanes = [(f'gpu{i}', 'cuda') for i in range(n_gpu)] + [(f'cpu{i}', 'cpu') for i in range(args.cpu_lanes)]
    lanes = lanes[:max(1, todo)]
    title = f"Preset: {', '.join(args.exp)} | {len(lanes)} lane ({n_gpu} GPU, {len(lanes) - n_gpu} CPU)"

    def blocked_summary():
        from collections import Counter
        c = Counter(reason[j] for j, s in enumerate(status) if s == 'blocked')
        return '; '.join(f"{n} run cho: {r}" for r, n in c.most_common(4))

    show(f"{title}\n  {n_total} run trong preset, {n_pre_done} da xong truoc do, {todo} can chay "
         f"({status.count('ready')} san sang, {status.count('blocked')} cho dieu kien).", wait=True)
    if args.dry_run:
        from collections import Counter
        time.sleep(0.5)
        print('  ready per preset  :', dict(Counter(jobs[i][0] for i, s in enumerate(status) if s == 'ready')))
        print('  blocked per preset:', dict(Counter(jobs[i][0] for i, s in enumerate(status) if s == 'blocked')))
        print('  blocked because   :', blocked_summary() or '-')
        for i in [i for i, s in enumerate(status) if s in ('ready', 'blocked')][:15]:
            j = jobs[i]
            print('   ', status[i], j[0], j[1], f's{j[2]}', f'p{j[3]}', f'R{j[4]}', {k: v for k, v in j[7].items() if v})
        return
    if not todo:
        show("  Khong con run nao can chay.", wait=True)
        time.sleep(1)
        return

    os.makedirs(os.path.join(args.outdir, 'manifests'), exist_ok=True)
    margs = argparse.Namespace(clients=args.clients, epochs=5, batch=args.batch,
                               aggregation=args.aggregation, weight_basis=args.weight_basis)
    manifest = rx.build_manifest(margs)
    manifest['launcher'] = dict(exp=args.exp, lanes=[l[0] for l in lanes], pending=todo)
    with open(os.path.join(args.outdir, 'manifests', f"manifest_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.json"), 'w') as fh:
        json.dump(manifest, fh, indent=1)

    hooks = HookRunner()
    hooks.start()
    for exp in args.exp:
        if exp in HOOKS:
            hooks.request(HOOKS[exp])

    ctx = mp.get_context('spawn')
    job_q, status_q = ctx.Queue(), ctx.Queue()
    cfg = dict(clients=args.clients, batch=args.batch, aggregation=args.aggregation,
               weight_basis=args.weight_basis, outdir=args.outdir, max_active=args.max_active,
               launcher=f"{time.strftime('%m%d-%H%M')}-{os.getpid()}")
    procs = {}
    for name, dev in lanes:
        saved = {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
        os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
        if dev == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        p = ctx.Process(target=lane_main, args=(name, dev, job_q, status_q, cfg, manifest), daemon=True)
        p.start()
        procs[name] = p
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    state = {n: dict(label=None, completed=0, total=1, stage='khoi dong', last=None, status='START', t0=None)
             for n, _ in lanes}
    idle = {n for n, _ in lanes}
    queued = 0
    started = monotonic()
    done_n = fail_n = 0
    durations, recent, exited = [], [], set()
    next_frame = next_check = monotonic()
    idle_since = None
    sentinels_sent = False

    def feed():
        nonlocal queued
        free = len(idle) - queued
        while free > 0:
            nxt = next((i for i, s in enumerate(status) if s == 'ready'), None)
            if nxt is None:
                return
            status[nxt] = 'queued'
            job_q.put((nxt, jobs[nxt]))
            queued += 1
            free -= 1

    def recheck_blocked():
        for jid, s in enumerate(status):
            if s == 'deferred' and monotonic() - deferred_at.get(jid, 0) > 600:
                st_, info_ = classify(jobs[jid])
                status[jid] = st_
                if st_ == 'blocked':
                    reason[jid] = info_
                elif st_ == 'done':
                    nonlocal_done.append(jid)
            elif s == 'blocked':
                st, info = classify(jobs[jid])
                status[jid] = st
                if st == 'blocked':
                    reason[jid] = info
                else:
                    reason.pop(jid, None)
                    if st == 'done':
                        nonlocal_done.append(jid)

    nonlocal_done = []
    deferred_at = {}
    try:
        while len(exited) < len(lanes):
            try:
                msg = status_q.get(timeout=1.0)
                kind, name = msg[0], msg[1]
                st = state[name]
                if kind == 'take':
                    queued -= 1
                    idle.discard(name)
                    status[msg[2]] = 'running'
                elif kind == 'wait':
                    st.update(label=msg[2], completed=0, total=1, stage=f'cho slot (<{args.max_active} run)',
                              last=monotonic(), status='CHO', t0=None)
                elif kind == 'start':
                    st.update(label=msg[2], completed=0, total=1, stage='nap du lieu', last=monotonic(),
                              status='RUN', t0=monotonic())
                elif kind == 'progress':
                    st.update(completed=msg[2], total=msg[3], stage=msg[4], last=monotonic())
                elif kind == 'done':
                    done_n += 1
                    durations.append(msg[5])
                    status[msg[6]] = 'done'
                    recent.append(f"{time.strftime('%H:%M')} XONG {msg[2]}: acc {msg[3]:.2f} F1 {msg[4]:.2f} ({msg[5]/60:.1f} phut) [{name}]")
                    st.update(status='RANH', stage='-', completed=0, total=1, label=None)
                    idle.add(name)
                    if msg[7] in HOOKS:
                        hooks.request(HOOKS[msg[7]])
                elif kind == 'skip':
                    jid = msg[3]
                    if classify(jobs[jid])[0] == 'done':
                        done_n += 1
                        status[jid] = 'done'
                        recent.append(f"{time.strftime('%H:%M')} BO QUA {msg[2]} (da xong)")
                    else:
                        status[jid] = 'deferred'
                        deferred_at[jid] = monotonic()
                        recent.append(f"{time.strftime('%H:%M')} HOAN {msg[2]} (tien trinh khac dang chay run nay)")
                    st.update(status='RANH', stage='-', label=None)
                    idle.add(name)
                elif kind == 'error':
                    fail_n += 1
                    status[msg[4]] = 'failed'
                    recent.append(f"{time.strftime('%H:%M')} LOI {msg[2]}: {msg[3][:90]} [{name}]")
                    st.update(status='LOI', stage='-')
                    idle.add(name)
                elif kind == 'exit':
                    exited.add(name)
                    idle.discard(name)
                    st.update(status='HET VIEC', stage='-', label=None, completed=0, total=1)
            except queue_mod.Empty:
                pass
            for name, p in procs.items():
                if not p.is_alive() and name not in exited:
                    exited.add(name)
                    idle.discard(name)
                    fail_n += 1
                    state[name].update(status='CHET', stage=f'exit {p.exitcode}')
                    recent.append(f"{time.strftime('%H:%M')} LANE {name} dung bat thuong (exit {p.exitcode}) - "
                                  f"xem logs/lane_{cfg['launcher']}_{name}.log")
            if hooks.changed.is_set() or monotonic() >= next_check:
                hooks.changed.clear()
                recheck_blocked()
                done_n += len(nonlocal_done)
                nonlocal_done.clear()
                next_check = monotonic() + 60
            if not sentinels_sent:
                feed()
                running = sum(1 for s in status if s in ('queued', 'running'))
                n_ready = status.count('ready')
                n_blocked = status.count('blocked') + status.count('deferred')
                if running == 0 and n_ready == 0:
                    if n_blocked == 0 or (idle_since and monotonic() - idle_since > 1800
                                          and not hooks.busy and not hooks.pending):
                        for _ in range(len(lanes) - len(exited)):
                            job_q.put(None)
                        sentinels_sent = True
                        if n_blocked:
                            recent.append(f"{time.strftime('%H:%M')} DUNG: {n_blocked} run van cho dieu kien: {blocked_summary()}")
                    elif idle_since is None:
                        idle_since = monotonic()
                else:
                    idle_since = None
            if monotonic() >= next_frame or len(exited) == len(lanes):
                remaining = sum(1 for s in status if s in ('ready', 'blocked', 'deferred', 'queued', 'running'))
                est = (sum(durations) / len(durations)) if durations else None
                if est is None:
                    proj = [(monotonic() - s['t0']) * s['total'] / s['completed']
                            for s in state.values() if s['status'] == 'RUN' and s['completed'] > s['total'] * 0.02]
                    est = sorted(proj)[len(proj) // 2] if proj else None
                eta = None
                if est and remaining > 0:
                    active = [s for s in state.values() if s['status'] == 'RUN']
                    left_active = sum(est * (1 - s['completed'] / max(1, s['total'])) for s in active)
                    queued_n = max(0, remaining - len(active))
                    eta = (left_active + queued_n * est) / max(1, len(lanes) - len(exited))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    render_lanes(title, started, todo, done_n, fail_n, state, recent, eta, clear=args.clear)
                frame = buf.getvalue().rstrip('\n')
                n_blocked = status.count('blocked')
                n_def = status.count('deferred')
                extra = [f"  Hang doi: {status.count('ready')} san sang, {n_blocked} cho dieu kien"
                         + (f" ({blocked_summary()})" if n_blocked else '')
                         + (f", {n_def} dang chay o tien trinh khac" if n_def else '')
                         + f" | hook: {'dang chay' if hooks.busy else hooks.last}"]
                show(frame + '\n' + '\n'.join(extra))
                next_frame = monotonic() + args.refresh
    except KeyboardInterrupt:
        show("\nDung theo yeu cau: dang tat cac lane (run dang do se duoc chay lai lan sau).", wait=True)
        for p in procs.values():
            p.terminate()
        return
    show(f"\n[HOAN TAT] {done_n} run xong, {fail_n} loi, tong thoi gian {(monotonic() - started)/60:.1f} phut. "
         f"Log chi tiet: {os.path.join(args.outdir, 'logs')}", wait=True)
    time.sleep(2)


if __name__ == '__main__':
    mp.freeze_support()
    main()

