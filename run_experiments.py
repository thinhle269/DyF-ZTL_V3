import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    import msvcrt
except ImportError:
    msvcrt = None

import src.fl_core as fl
import src.preprocessing as prep
import src.utils as utils

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "dataset", "Train_Test_Windows_10.csv")


METHODS = {
    'FedAvg':         dict(arch='deep',  algo='fedavg',  defense=None),
    'FedProx':        dict(arch='deep',  algo='fedprox', defense=None),
    'SCAFFOLD':       dict(arch='deep',  algo='scaffold', defense=None),
    'SCAFFOLD-SGD05': dict(arch='deep',  algo='scaffold_sgd', defense=None, lr=0.05),
    'SCAFFOLD-SGD10': dict(arch='deep',  algo='scaffold_sgd', defense=None, lr=0.1),
    'SCAFFOLD-SGD20': dict(arch='deep',  algo='scaffold_sgd', defense=None, lr=0.2),

    'SCAFFOLD2-SGD01': dict(arch='deep', algo='scaffold2', defense=None, lr=0.01),
    'SCAFFOLD2-SGD02': dict(arch='deep', algo='scaffold2', defense=None, lr=0.02),
    'SCAFFOLD2-SGD05': dict(arch='deep', algo='scaffold2', defense=None, lr=0.05),


    'DyF-ZTL':        dict(arch='fuzzy', algo='fedavg',  defense='trust2', sim=dict(aggregation='uniform')),
    'DyF-ZTL-v1':     dict(arch='fuzzy', algo='fedavg',  defense='trust',  sim=dict(aggregation='uniform')),
    'FuzzyNoTrust':   dict(arch='fuzzy', algo='fedavg',  defense=None,     sim=dict(aggregation='uniform')),
    'DeepTrust':      dict(arch='deep',  algo='fedavg',  defense='trust2', gate_set='mlp-U',
                           sim=dict(aggregation='uniform')),
    'DeepTrust-cal':  dict(arch='deep',  algo='fedavg',  defense='trust2', trust=dict(log_only=True)),
    'DeepTrust-cal-U': dict(arch='deep', algo='fedavg',  defense='trust2', trust=dict(log_only=True),
                            sim=dict(aggregation='uniform')),

    'FedAvg-U':       dict(arch='deep',  algo='fedavg',  defense=None,     sim=dict(aggregation='uniform')),
    'FedProx-U':      dict(arch='deep',  algo='fedprox', defense=None,     sim=dict(aggregation='uniform')),
    'DyF-ZTL-U':      dict(arch='fuzzy', algo='fedavg',  defense='trust2', sim=dict(aggregation='uniform'),
                           gate_set='default'),
    'DyF-ZTL-PS':     dict(arch='fuzzy', algo='fedavg',  defense='trust2', sim=dict(weight_basis='post_smote')),

    'FedAvg-CW':      dict(arch='deep',  algo='fedavg',  defense=None,     sim=dict(loss='weighted_ce')),
    'FedProx-CW':     dict(arch='deep',  algo='fedprox', defense=None,     sim=dict(loss='weighted_ce')),
    'DyF-ZTL-CW':     dict(arch='fuzzy', algo='fedavg',  defense='trust2',
                           sim=dict(loss='weighted_ce', aggregation='uniform')),

    'DyF-ZTL-cal':    dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(log_only=True)),

    'DyF-ZTL-R':      dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           sim=dict(aggregation='uniform')),
    'DyF-ZTL-cal-R':  dict(arch='fuzzy', algo='fedavg',  defense='trust2',
                           trust=dict(log_only=True, functional_ref='reference'), sim=dict(aggregation='uniform')),

    'DyF-ZTL-R-F':    dict(arch='fuzzy', algo='fedavg',  defense='trust2',
                           trust=dict(functional_ref='reference', use_geometric=False), sim=dict(aggregation='uniform')),
    'DyF-ZTL-R-G':    dict(arch='fuzzy', algo='fedavg',  defense='trust2',
                           trust=dict(functional_ref='reference', use_functional=False), sim=dict(aggregation='uniform')),
    'DyF-ZTL-R-soft': dict(arch='fuzzy', algo='fedavg',  defense='trust2',
                           trust=dict(functional_ref='reference', mode='soft'), sim=dict(aggregation='uniform')),
    'DyF-ZTL-R-prob': dict(arch='fuzzy', algo='fedavg',  defense='trust2',
                           trust=dict(functional_ref='reference', mode='probation'), sim=dict(aggregation='uniform')),
    'DyF-ZTL-R-dg':   dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='default-R-U', sim=dict(aggregation='uniform')),
    'DyF-ZTL-R-CW':   dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           sim=dict(loss='weighted_ce', aggregation='uniform')),
    'DeepTrust-R':    dict(arch='deep',  algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='mlp-R-U', sim=dict(aggregation='uniform')),
    'DeepTrust-cal-R': dict(arch='deep', algo='fedavg',  defense='trust2',
                            trust=dict(log_only=True, functional_ref='reference'), sim=dict(aggregation='uniform')),


    'DyF-ZTL-RN':     dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='default-R-U', sim=dict(aggregation='uniform', agg_norm='reference')),
    'DyF-ZTL-RV':     dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='default-R-U', sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-RNV':    dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='default-R-U', sim=dict(aggregation='uniform', agg_norm='reference', virtual_ref=True)),

    'DyF-ZTL-V':      dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-cal-V':  dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(log_only=True, functional_ref='reference'),
                           sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-V-F':    dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference', use_geometric=False),
                           sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-V-G':    dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference', use_functional=False),
                           sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-V-soft': dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference', mode='soft'),
                           sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-V-prob': dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference', mode='probation'),
                           sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-V-dg':   dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='default-V-R-U', sim=dict(aggregation='uniform', virtual_ref=True)),
    'DyF-ZTL-V-CW':   dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           sim=dict(loss='weighted_ce', aggregation='uniform', virtual_ref=True)),
    'DeepTrust-V':    dict(arch='deep',  algo='fedavg',  defense='trust2', trust=dict(functional_ref='reference'),
                           gate_set='mlp-V-R-U', sim=dict(aggregation='uniform', virtual_ref=True)),
    'DeepTrust-cal-V': dict(arch='deep', algo='fedavg',  defense='trust2', trust=dict(log_only=True, functional_ref='reference'),
                            sim=dict(aggregation='uniform', virtual_ref=True)),

    'ServerOnly':     dict(arch='fuzzy', algo='fedavg',  defense='server_only'),
    'ServerOnly-MLP': dict(arch='deep',  algo='fedavg',  defense='server_only'),

    'DyF-ZTL-cal-U':  dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(log_only=True),
                           sim=dict(aggregation='uniform')),
    'DyF-ZTL-F':      dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(use_geometric=False),
                           sim=dict(aggregation='uniform')),
    'DyF-ZTL-G':      dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(use_functional=False),
                           sim=dict(aggregation='uniform')),
    'DyF-ZTL-soft':   dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(mode='soft'),
                           sim=dict(aggregation='uniform')),
    'DyF-ZTL-prob':   dict(arch='fuzzy', algo='fedavg',  defense='trust2', trust=dict(mode='probation'),
                           sim=dict(aggregation='uniform')),
    'DyF-ZTL-dg':     dict(arch='fuzzy', algo='fedavg',  defense='trust2', gate_set='default-U',
                           sim=dict(aggregation='uniform')),

    'FuzzyNoTrust-legacy': dict(arch='fuzzy_legacy', algo='fedavg', defense=None, sim=dict(aggregation='uniform')),

    'Krum':           dict(arch='deep',  algo='fedavg',  defense='krum'),
    'MultiKrum':      dict(arch='deep',  algo='fedavg',  defense='multikrum'),
    'Median':         dict(arch='deep',  algo='fedavg',  defense='median'),
    'TrimmedMean':    dict(arch='deep',  algo='fedavg',  defense='trimmed'),
    'FLTrust':        dict(arch='deep',  algo='fedavg',  defense='fltrust'),
    'FuzzyKrum':      dict(arch='fuzzy', algo='fedavg',  defense='krum'),
    'FuzzyMultiKrum': dict(arch='fuzzy', algo='fedavg',  defense='multikrum'),
    'FuzzyMedian':    dict(arch='fuzzy', algo='fedavg',  defense='median'),
    'FuzzyTrimmed':   dict(arch='fuzzy', algo='fedavg',  defense='trimmed'),
    'FuzzyFLTrust':   dict(arch='fuzzy', algo='fedavg',  defense='fltrust'),
}

CLEAN = dict(alpha=0.5, smote=True, split='random')
ALL_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TEN = list(range(10))
FIVE = list(range(5))
PILOT_SEEDS = [100, 101]
CAL_DIR = os.environ.get('DYF_CAL_DIR') or os.path.join(ROOT, 'results_r02', 'calibration')


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


SCAFFOLD_MAIN = 'SCAFFOLD*'


def scaffold_winner():
    s = _read_json(os.path.join(CAL_DIR, 'scaffold.json'))
    if not s or not s.get('complete') or s.get('method') not in METHODS:
        raise FileNotFoundError("SCAFFOLD solver not chosen yet: finish `pilot_scaffold` and run audit/choose_scaffold.py")
    return s['method']
HET_METHODS = ['FedAvg', 'FedProx', SCAFFOLD_MAIN, 'DyF-ZTL-V']

REGISTERED_W = dict(sim=dict(aggregation='weighted'))
MAIN_METHODS = ['FedAvg', 'FedProx', SCAFFOLD_MAIN, 'FuzzyNoTrust', 'DyF-ZTL', 'DeepTrust', 'DyF-ZTL-v1',
                'DyF-ZTL-R', 'DeepTrust-R',
                'DyF-ZTL-V', 'DeepTrust-V', 'ServerOnly', 'ServerOnly-MLP']
HET_JOBS = ([(m, 0.0, dict(CLEAN, alpha=a)) for a in (0.1, 1.0, 10.0, None) for m in HET_METHODS]
            + [(m, 0.0, dict(CLEAN, smote=False)) for m in HET_METHODS]
            + [(m, 0.0, dict(CLEAN, smote=False)) for m in ('FedAvg-CW', 'FedProx-CW', 'DyF-ZTL-V-CW')])


PRESETS = {
    'smoke': dict(provisional=False, seeds=[0], rounds=2, epochs=1, desc="pipeline + invariants",
                  jobs=[('FedAvg', 0.0, CLEAN), ('FedProx', 0.0, CLEAN), ('DyF-ZTL-v1', 0.0, CLEAN),
                        ('FuzzyNoTrust', 0.0, CLEAN), ('FuzzyNoTrust-legacy', 0.0, CLEAN),
                        ('FedAvg-U', 0.0, CLEAN), ('DyF-ZTL-v1', 0.4, CLEAN), ('Krum', 0.4, CLEAN),
                        ('MultiKrum', 0.4, CLEAN), ('Median', 0.4, CLEAN), ('TrimmedMean', 0.4, CLEAN),
                        ('FLTrust', 0.4, CLEAN), ('FuzzyFLTrust', 0.4, CLEAN),
                        ('DyF-ZTL-v1', 0.0, dict(CLEAN, split='chrono_class')),
                        ('FedAvg', 0.0, dict(CLEAN, smote=False)),
                        ('FedAvg-CW', 0.0, dict(CLEAN, smote=False)),
                        ('FedAvg', 0.0, dict(CLEAN, alpha=None))]),
    'main': dict(provisional=True, seeds=TEN, rounds=100, desc="clean comparison + ablation (R1.1, R1.5, R1.6, R1.9)",
                 jobs=[(m, 0.0, CLEAN) for m in MAIN_METHODS]
                 + [(m, 0.0, CLEAN, REGISTERED_W) for m in ('DyF-ZTL', 'FuzzyNoTrust', 'DyF-ZTL-v1')]),
    'agg_ablation': dict(provisional=True, seeds=TEN, rounds=100, desc="uniform vs weighted aggregation (R1.1)",
                         methods=['FedAvg-U', 'FedProx-U', 'DyF-ZTL-U', 'DyF-ZTL-PS'], ratios=[0.0], data=CLEAN),
    'regression': dict(provisional=True, seeds=[0, 1, 2], rounds=100, desc="Round-1 fuzzy layer vs constrained layer (F01)",
                       jobs=[(m, 0.0, CLEAN, ex) for ex in ({}, REGISTERED_W)
                             for m in ('FuzzyNoTrust-legacy', 'FuzzyNoTrust')]),
    'pilot_horizon': dict(provisional=True, seeds=PILOT_SEEDS, rounds=100, desc="choose the poisoning-sweep horizon (R1.3)",
                          jobs=[(m, r, CLEAN) for r in (0.4, 0.8) for m in ('FedAvg', 'FLTrust')]
                          + [('DyF-ZTL', r, CLEAN, REGISTERED_W) for r in (0.4, 0.8)]),
    'chrono': dict(provisional=True, seeds=TEN, rounds=100, desc="per-class chronological split (R1.5)",
                   methods=['FedAvg', 'FedProx', 'DyF-ZTL-V'], ratios=[0.0], data=dict(CLEAN, split='chrono_class')),
    'heterogeneity': dict(provisional=True, seeds=FIVE, rounds=100, desc="Dirichlet alpha, IID, no-SMOTE, class-weighted (R1.5)",
                          jobs=HET_JOBS),
}


CAL_SEEDS = [100, 101, 102]
CAL_CONDITIONS = [('default', {}), ('size100', dict(trust_val_size=100)), ('size300', dict(trust_val_size=300)),
                  ('size1200', dict(trust_val_size=1200)), ('imbalanced', dict(trust_condition='imbalanced')),
                  ('missing2', dict(trust_condition='missing2')), ('shifted', dict(trust_condition='shifted')),
                  ('contam5', dict(trust_condition='contam5')), ('contam10', dict(trust_condition='contam10'))]


R_ATTACK =(_read_json(os.path.join(CAL_DIR, 'horizon.json')) or {}).get('rounds')
ROBUST_MLP = ['FedAvg', 'FedProx', 'Krum', 'MultiKrum', 'Median', 'TrimmedMean', 'FLTrust']
ATTACK_METHODS = ['FedAvg', 'Krum', 'MultiKrum', 'Median', 'TrimmedMean', 'FLTrust',
                  'DyF-ZTL-V', 'DyF-ZTL-v1', 'DyF-ZTL-V-F', 'DyF-ZTL-V-G']
SCHEDULE_METHODS = ['FedAvg', 'FLTrust', 'DyF-ZTL-V', 'DyF-ZTL-v1']
ATTACK_CONTROLS = ['FuzzyNoTrust', 'FuzzyFLTrust']
ATTACK_JOBS = ([(m, r, CLEAN, dict(sim=dict(attack=a)))
                for a in ('targeted_flip', 'backdoor', 'backdoor_boost', 'adaptive_flip', 'adaptive_backdoor')
                for r in (0.2, 0.4) for m in ATTACK_METHODS + ATTACK_CONTROLS]
               + [(m, r, CLEAN, dict(sim=dict(schedule=s), rounds_min=40))
                  for s in ('sleeper', 'rotating') for r in (0.2, 0.4) for m in SCHEDULE_METHODS]
               + [(m, r, CLEAN, dict(sim=dict(participation=0.5)))
                  for r in (0.2, 0.4) for m in SCHEDULE_METHODS])
CYCLIC_JOBS = ([(m, r, CLEAN) for r in ALL_RATIOS for m in ROBUST_MLP + ['DyF-ZTL-V', 'DyF-ZTL-v1']]
               + [(m, r, CLEAN, dict(seeds=FIVE)) for r in ALL_RATIOS for m in ('FuzzyNoTrust', 'DeepTrust-V')]

               + [('DyF-ZTL', r, CLEAN, dict(REGISTERED_W, seeds=FIVE)) for r in (0.2, 0.4, 0.6, 0.8)])
TRUSTED_JOBS = ([(m, r, CLEAN, dict(sim=sim, gate_set=f'{key}-V-R-U' if m == 'DyF-ZTL-V' else None))
                 for key, sim in CAL_CONDITIONS[1:] for r in (0.2, 0.4)
                 for m in (['DyF-ZTL-V', 'DyF-ZTL-V-dg'] + (['FLTrust'] if 'trust_condition' in sim else []))])
SENS_GRID = ([('decay_factor', v) for v in (0.1, 0.4, 0.8)] + [('mild_decay', v) for v in (0.8, 0.95)]
             + [('recovery_factor', v) for v in (0.02, 0.1)] + [('alpha', v) for v in (0.25, 1.0)]
             + [('min_safety_threshold', v) for v in (0.2, 0.6)] + [('warmup_rounds', v) for v in (2, 3, 4)])
SENS_JOBS = ([('DyF-ZTL-V', r, CLEAN, dict(trust={p: v})) for p, v in SENS_GRID for r in (0.2, 0.4)]
             + [('DyF-ZTL-V', r, CLEAN, dict(gate_set=g)) for g in ('q_strong0.1-V-R-U', 'q_strong1-V-R-U', 'q_mild2-V-R-U', 'q_mild10-V-R-U')
                for r in (0.2, 0.4)])

PRESETS.update({
    'smoke_v2': dict(provisional=False, seeds=[0], rounds=3, epochs=1, desc="pipeline check of Round-2 engine, attacks, SCAFFOLD (needs DYF_CAL_DIR test gates)",
                     jobs=[('DyF-ZTL', 0.0, CLEAN), ('DyF-ZTL', 0.4, CLEAN), ('DyF-ZTL-cal', 0.0, CLEAN),
                           ('DyF-ZTL-F', 0.4, CLEAN), ('DyF-ZTL-G', 0.4, CLEAN), ('DyF-ZTL-soft', 0.4, CLEAN),
                           ('DyF-ZTL-prob', 0.4, CLEAN), ('DeepTrust', 0.4, CLEAN), ('SCAFFOLD', 0.0, CLEAN),
                           ('DyF-ZTL', 0.0, dict(CLEAN, split='chrono_class'))]
                     + [(m, 0.2, CLEAN, dict(sim=dict(attack=a))) for a in ('targeted_flip', 'backdoor', 'backdoor_boost',
                                                                          'adaptive_flip', 'adaptive_backdoor')
                        for m in ('FedAvg', 'DyF-ZTL', 'FLTrust', 'MultiKrum')]
                     + [(m, 0.2, CLEAN, dict(sim=sim)) for sim in (dict(schedule='rotating'), dict(schedule='sleeper'),
                                                                   dict(participation=0.5))
                        for m in ('FedAvg', 'DyF-ZTL', 'DyF-ZTL-v1')]
                     + [(m, 0.2, CLEAN, dict(sim=sim, gate_set=k if m == 'DyF-ZTL' else None))
                        for k, sim in CAL_CONDITIONS[1:] for m in ('DyF-ZTL', 'FLTrust')]),
    'pilot_scaffold': dict(provisional=True, seeds=PILOT_SEEDS, rounds=50, desc="choose the SCAFFOLD local solver on validation (D4)",
                           methods=['FedAvg', 'SCAFFOLD', 'SCAFFOLD-SGD05', 'SCAFFOLD-SGD10', 'SCAFFOLD-SGD20',
                                    'SCAFFOLD2-SGD01', 'SCAFFOLD2-SGD02', 'SCAFFOLD2-SGD05'],
                           ratios=[0.0], data=CLEAN),
    'calibrate': dict(provisional=True, seeds=CAL_SEEDS, rounds=100, job_major=True,
                      desc="log-only clean runs -> gates per trusted-set condition",
                      jobs=[('DyF-ZTL-cal', 0.0, CLEAN, dict(sim=sim)) for _, sim in CAL_CONDITIONS]
                      + [('DyF-ZTL-cal', 0.0, dict(CLEAN, split='chrono_class'))]
                      + [('DeepTrust-cal', 0.0, CLEAN)]

                      + [('DyF-ZTL-cal-U', 0.0, CLEAN, dict(sim=sim)) for _, sim in CAL_CONDITIONS]
                      + [('DyF-ZTL-cal-U', 0.0, dict(CLEAN, split='chrono_class'))]
                      + [('DeepTrust-cal-U', 0.0, CLEAN)]

                      + [('DyF-ZTL-cal-U', 0.0, dict(CLEAN, split='chrono_class', features=f))
                         for f in ('drift_robust', 'drift_oracle')]
                      + [('DyF-ZTL-cal-U', 0.0, dict(CLEAN, features='drift_oracle'))]

                      + [('DyF-ZTL-cal-R', 0.0, CLEAN, dict(sim=sim)) for _, sim in CAL_CONDITIONS]
                      + [('DyF-ZTL-cal-R', 0.0, dict(CLEAN, split='chrono_class'))]
                      + [('DyF-ZTL-cal-R', 0.0, dict(CLEAN, split='chrono_class', features=f))
                         for f in ('drift_robust', 'drift_oracle')]
                      + [('DyF-ZTL-cal-R', 0.0, dict(CLEAN, features='drift_oracle'))]
                      + [('DeepTrust-cal-R', 0.0, CLEAN)]

                      + [('DyF-ZTL-cal-R', 0.0, CLEAN, REGISTERED_W)]

                      + [('DyF-ZTL-cal-R', 0.0, dict(CLEAN, alpha=a)) for a in (0.1, 1.0, 10.0, None)]
                      + [('DyF-ZTL-cal-R', 0.0, dict(CLEAN, smote=False))]

                      + [('DyF-ZTL-cal-V', 0.0, CLEAN, dict(sim=sim)) for _, sim in CAL_CONDITIONS]
                      + [('DyF-ZTL-cal-V', 0.0, dict(CLEAN, split='chrono_class'))]
                      + [('DyF-ZTL-cal-V', 0.0, dict(CLEAN, split='chrono_class', features=f)) for f in ('drift_robust', 'drift_oracle')]
                      + [('DyF-ZTL-cal-V', 0.0, dict(CLEAN, features='drift_oracle'))]
                      + [('DyF-ZTL-cal-V', 0.0, dict(CLEAN, alpha=a)) for a in (0.1, 1.0, 10.0, None)]
                      + [('DyF-ZTL-cal-V', 0.0, dict(CLEAN, smote=False))]
                      + [('DyF-ZTL-cal-V', 0.0, CLEAN, REGISTERED_W)]
                      + [('DeepTrust-cal-V', 0.0, CLEAN)]),


    'scaffold_check': dict(provisional=True, seeds=TEN, rounds=100, desc="SCAFFOLD option II, runner-up step size (0.02)",
                           jobs=[('SCAFFOLD2-SGD02', 0.0, CLEAN)]

                           + [('SCAFFOLD2-SGD02', 0.0, dict(CLEAN, alpha=a), dict(seeds=FIVE)) for a in (0.1, 1.0, 10.0, None)]
                           + [('SCAFFOLD2-SGD02', 0.0, dict(CLEAN, smote=False), dict(seeds=FIVE))]),


    'drift': dict(provisional=True, seeds=TEN, rounds=100, desc="feature drift: drift-robust / oracle feature sets (R1.5)",
                  jobs=[(m, 0.0, dict(CLEAN, split='chrono_class', features=f))
                        for f in ('drift_robust', 'drift_oracle') for m in ('FedProx', 'FedProx-U', 'DyF-ZTL-V')]
                  + [(m, 0.0, dict(CLEAN, features='drift_oracle'), dict(seeds=FIVE)) for m in ('FedProx', 'FedProx-U', 'DyF-ZTL-V')]
                  + [('FedProx-U', 0.0, dict(CLEAN, split='chrono_class'))]),

    'pilot_anchor': dict(provisional=True, seeds=PILOT_SEEDS, rounds=100,
                         desc="pilot: functional evidence anchored to the server reference model (v2.1) vs v2",
                         jobs=[(m, r, CLEAN, ex) for m in ('DyF-ZTL', 'DyF-ZTL-R')
                               for r, ex in ((0.0, {}), (0.4, {}), (0.8, {}), (0.9, {}),
                                             (0.4, dict(sim=dict(attack='backdoor'))),
                                             (0.2, dict(sim=dict(attack='targeted_flip'))))]),


    'agg_v21': dict(provisional=True, seeds=TEN, rounds=100, desc="v2.1 engine, |D_k|-weighted admitted averaging (R1.1)",
                    jobs=[('DyF-ZTL-R', 0.0, CLEAN, REGISTERED_W)]
                    + [('DyF-ZTL-R', r, CLEAN, dict(REGISTERED_W, seeds=FIVE)) for r in (0.2, 0.4, 0.6, 0.8)]),
    'agg_v22': dict(provisional=True, seeds=TEN, rounds=100, desc="v2.2 engine, |D_k|-weighted admitted averaging (R1.1)",
                    jobs=[('DyF-ZTL-V', 0.0, CLEAN, REGISTERED_W)]
                    + [('DyF-ZTL-V', r, CLEAN, dict(REGISTERED_W, seeds=FIVE)) for r in (0.2, 0.4, 0.6, 0.8)]),

    'pilot_agg': dict(provisional=True, seeds=PILOT_SEEDS, rounds=100,
                      desc="pilot: admitted-update normalization (N) / server reference as virtual participant (V) vs v2.1",
                      jobs=[(m, r, CLEAN, ex) for m in ('DyF-ZTL-RN', 'DyF-ZTL-RV', 'DyF-ZTL-RNV')
                            for r, ex in ((0.0, {}), (0.5, {}), (0.8, {}), (0.9, {}),
                                          (0.2, dict(sim=dict(attack='targeted_flip'))),
                                          (0.4, dict(sim=dict(attack='backdoor'))),
                                          (0.4, dict(sim=dict(attack='backdoor_boost'))))]
                      + [('DyF-ZTL-R', 0.5, CLEAN), ('DyF-ZTL-R', 0.4, CLEAN, dict(sim=dict(attack='backdoor_boost')))]),


    'server_only_sizes': dict(provisional=True, seeds=TEN, rounds=100,
                              desc="server-only reference vs DyF-ZTL (v2.2) across trusted-set sizes (clean)",
                              jobs=[(m, 0.0, CLEAN, dict(sim=dict(trust_val_size=s)))
                                    for s in (100, 300, 1200) for m in ('ServerOnly', 'ServerOnly-MLP')]
                              + [('DyF-ZTL-V', 0.0, CLEAN, dict(sim=dict(trust_val_size=s), gate_set=f'size{s}-V-R-U', seeds=FIVE))
                                 for s in (100, 300, 1200)]),
    'pilot_uniform': dict(provisional=True, seeds=PILOT_SEEDS, rounds=100,
                          desc="post-hoc pilot: Round-2 engine with uniform averaging of admitted updates (§6c.11)",
                          methods=['DyF-ZTL-U'], ratios=[0.0], data=CLEAN),
    'cyclic': dict(provisional=True, seeds=TEN, rounds=R_ATTACK, desc="cyclic label-flip sweep 0-0.9 (R1.3)", jobs=CYCLIC_JOBS),
    'attacks': dict(provisional=True, seeds=FIVE, rounds=R_ATTACK, desc="targeted/backdoor/boost/adaptive, schedules, partial participation (R1.3, R2.3)",
                    jobs=ATTACK_JOBS),
    'trust_modes': dict(provisional=True, seeds=FIVE, rounds=R_ATTACK, desc="soft vs probation (hard = cyclic preset) (R1.4)",
                        methods=['DyF-ZTL-V-soft', 'DyF-ZTL-V-prob'], ratios=[0.2, 0.4, 0.6, 0.8], data=CLEAN),
    'trusted_set': dict(provisional=True, seeds=FIVE, rounds=R_ATTACK, desc="trusted-corpus conditions, recalibrated vs default gates (R1.4)",
                        jobs=TRUSTED_JOBS),
    'sensitivity': dict(provisional=True, seeds=FIVE, rounds=R_ATTACK, desc="one-at-a-time sweep of the Round-2 engine (R1.8, R2.1)",
                        jobs=SENS_JOBS),
})


def list_presets():
    for name, p in PRESETS.items():
        tag = "PROVISIONAL" if p['provisional'] else "ready"
        methods = sorted({j[0] for j in p['jobs']}) if 'jobs' in p else p['methods']
        ratios = sorted({j[1] for j in p['jobs']}) if 'jobs' in p else p['ratios']
        print(f"{name:14s} [{tag}] seeds={p['seeds']} rounds={p['rounds']} ratios={ratios}\n"
              f"{'':15s}methods={methods}\n{'':15s}{p['desc']}")


def gates_for(key):
    g = _read_json(os.path.join(CAL_DIR, f'gates_{key}.json'))
    if g is None or not g.get('complete', True):
        raise FileNotFoundError(f"calibrated gates '{key}' not found (or calibration incomplete) in {CAL_DIR}; "
                                f"run the 'calibrate' preset and audit/calibrate_gates.py first")
    return {k: g[k] for k in ('rho_s', 'rho_m', 'c_s', 'c_m', 'm_s')}


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(args):
    code_files = sorted([os.path.join('src', f) for f in os.listdir(os.path.join(ROOT, 'src')) if f.endswith('.py')]
                        + ['run_experiments.py'])
    code = {f: _sha256(os.path.join(ROOT, f)) for f in code_files}
    import sklearn, imblearn
    return dict(
        created=time.strftime('%Y-%m-%dT%H:%M:%S'), argv=sys.argv, pid=os.getpid(),
        code_sha256=code, code_hash=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()[:16],
        dataset=dict(path=os.path.relpath(DATA_PATH, ROOT), sha256=_sha256(DATA_PATH)),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      pandas=pd.__version__, sklearn=sklearn.__version__, imblearn=imblearn.__version__),
        hardware=dict(platform=platform.platform(), cpu=platform.processor(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        defaults=dict(clients=args.clients, epochs=args.epochs, batch=args.batch,
                      aggregation=args.aggregation, weight_basis=args.weight_basis))


_DATA_CACHE = {}


def get_data(args, seed, data, outdir):
    feats = data.get('features', 'default')
    key = (args.clients, seed, data['alpha'], data['smote'], data['split'], feats)
    if key not in _DATA_CACHE:
        _DATA_CACHE.clear()
        _DATA_CACHE[key] = prep.load_and_process_data(
            DATA_PATH, num_clients=args.clients, non_iid_alpha=data['alpha'], partition_seed=seed,
            smote=data['smote'], split=data['split'], verbose=not args.quiet, features=feats)
        pdir = os.path.join(outdir, 'partitions')
        os.makedirs(pdir, exist_ok=True)
        pfile = os.path.join(pdir, f"{data['split']}_a{data['alpha']}_{'smote' if data['smote'] else 'nosmote'}_s{seed}"
                                   f"{FEATURE_SUFFIX.get(feats, '').replace('-', '_')}.json")
        if not os.path.exists(pfile):
            with open(pfile, 'w') as fh:
                json.dump(prep.LAST_REPORT, fh, indent=1)
    return _DATA_CACHE[key]


def append_row(csv_path, row):
    df = pd.DataFrame([row])
    lock_path = csv_path + '.lock'
    for _ in range(50):
        lf = None
        try:
            lf = open(lock_path, 'a')
            if msvcrt is not None:
                try:
                    msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    lf.close()
                    time.sleep(0.2 + random.random() * 0.3)
                    continue
            try:
                header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
                df.to_csv(csv_path, mode='a', header=header, index=False)
                return
            finally:
                if msvcrt is not None:
                    try:
                        msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                lf.close()
        except Exception:
            if lf is not None:
                try:
                    lf.close()
                except Exception:
                    pass
            time.sleep(0.2)
    df.to_csv(f"{csv_path}.part{os.getpid()}.csv", mode='a', header=True, index=False)


SIM_KEYS = ('aggregation', 'weight_basis', 'loss', 'attack', 'schedule', 'participation',
            'trust_condition', 'trust_val_size', 'fltrust_root_size', 'trigger', 'agg_norm', 'virtual_ref')
BACKDOOR_ATTACKS = ('backdoor', 'backdoor_boost', 'adaptive_backdoor')


def trigger_spec():
    t = _read_json(os.path.join(CAL_DIR, 'trigger.json'))
    return None if t is None else dict(features=[int(f) for f in t['features']], values=[float(v) for v in t['values']])


FEATURE_SUFFIX = {'drift_robust': '-dr', 'drift_oracle': '-do'}


def het_suffix(alpha, smote):

    a = '' if alpha == 0.5 else f"-a{'iid' if alpha is None else alpha}"
    return a + ('' if smote else '-nosmote')


def condition_key(sim, data):
    u = '-U' if sim.get('aggregation') == 'uniform' else ''
    f = FEATURE_SUFFIX.get(data.get('features', 'default'), '')
    h = het_suffix(data.get('alpha', 0.5), data.get('smote', True))
    v = ('-N' if sim.get('agg_norm') == 'reference' else '') + ('-V' if sim.get('virtual_ref') else '')
    if sim.get('trust_condition'):
        return sim['trust_condition'] + h + f + v + u
    if sim.get('trust_val_size'):
        return f"size{sim['trust_val_size']}" + h + f + v + u
    if data.get('split') == 'chrono_class':
        return 'chrono' + h + f + v + u
    return 'default' + h + f + v + u


def run_config(exp, method, seed, ratio, rounds, data, args, trust_params=None, sim_kwargs=None, gate_set=None):


    if method == SCAFFOLD_MAIN:
        method = scaffold_winner()
    cfg = METHODS[method]
    sim = dict(aggregation=args.aggregation, weight_basis=args.weight_basis, loss='ce', attack='cyclic_flip')
    sim.update(cfg.get('sim', {}))
    sim.update(sim_kwargs or {})
    tp = dict(cfg.get('trust', {}))
    tp.update(trust_params or {})
    extra = {}
    if sim.get('attack') in BACKDOOR_ATTACKS:
        spec = trigger_spec()
        if spec is None:
            raise FileNotFoundError(f"trigger.json not found in {CAL_DIR}; run audit/trigger_design.py first")
        sim['trigger'] = spec
    if cfg['defense'] == 'trust2' and not tp.get('log_only'):
        key = condition_key(sim, data)
        if tp.get('functional_ref') == 'reference':
            key = key[:-2] + '-R-U' if key.endswith('-U') else key + '-R'
        extra['gate_set'] = gate_set or cfg.get('gate_set') or key
        tp['gates'] = gates_for(extra['gate_set'])
    return dict(exp=exp, method=method, arch=cfg['arch'], algo=cfg['algo'], defense=cfg['defense'],
                seed=seed, attack=sim.pop('attack'), ratio=ratio, rounds=rounds,
                epochs=args.epochs, batch=args.batch, clients=args.clients, lr=cfg.get('lr', 0.01), mu=0.01,
                alpha=data['alpha'], smote=data['smote'], split=data['split'],
                trust_params=tp, **sim, **extra,
                **({'features': data['features']} if data.get('features', 'default') != 'default' else {}))


def run_id_of(config):
    h = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{config['method']}_s{config['seed']}_p{config['ratio']}_R{config['rounds']}_{h}"


def run_one(exp, method, seed, ratio, rounds, data, args, manifest, trust_params=None, sim_kwargs=None,
            progress_callback=None, gate_set=None):
    config = run_config(exp, method, seed, ratio, rounds, data, args, trust_params, sim_kwargs, gate_set)
    rid = run_id_of(config)
    exp_dir = os.path.join(args.outdir, exp)
    os.makedirs(exp_dir, exist_ok=True)
    final_path = os.path.join(exp_dir, f"final_{rid}.json")
    if os.path.exists(final_path):
        print(f"[SKIP] {rid} already done", flush=True)
        return None
    claim = os.path.join(exp_dir, f"claim_{rid}")
    try:
        with open(claim, 'x') as fh:
            fh.write(str(os.getpid()))
    except FileExistsError:
        stale = time.time() - os.path.getmtime(claim) >= 4 * 3600
        if not stale:
            try:
                import psutil
                with open(claim) as fh:
                    stale = not psutil.pid_exists(int(fh.read().strip() or 0))
            except Exception:
                pass
        if not stale:
            print(f"[SKIP] {rid} claimed by another process", flush=True)
            return None
        with open(claim, 'w') as fh:
            fh.write(str(os.getpid()))

    client_data, val_data, test_data, num_classes, input_dim, classes = get_data(args, seed, data, args.outdir)
    sim_args = {k: config[k] for k in SIM_KEYS if k in config}
    trust = config['defense'] in ('trust', 'trust2')
    logged = config['defense'] in ('trust', 'trust2', 'krum', 'multikrum', 'fltrust')
    print(f"\n=== [{exp}] {rid} ===", flush=True)
    start = time.time()
    val_acc, y_true, y_pred, val_hist = fl.run_fl_simulation(
        client_data, val_data, test_data, 'fedavg', rounds, args.clients, args.epochs, args.batch,
        poison_ratio=ratio, seed=seed, arch=config['arch'], algo=config['algo'], defense=config['defense'],
        trust_params=config['trust_params'], lr=config['lr'],
        trust_csv_path=os.path.join(exp_dir, f"trust_{rid}.csv") if trust else None,
        decision_log_path=os.path.join(exp_dir, f"decisions_{rid}.csv") if logged else None,
        method_label=method, show_progress=not args.quiet, progress_callback=progress_callback,
        device=getattr(args, 'device', None), diag_trigger=trigger_spec(),
        client_models_path=(os.path.join(exp_dir, f"clients_{rid}.pt")
                            if exp in ('main', 'chrono') and config['arch'].startswith('fuzzy') else None),
        **sim_args)
    duration = time.time() - start

    test_metrics, cm = utils.calculate_extended_metrics(y_true, y_pred, method, num_classes=num_classes)
    per_class = utils.per_class_metrics(cm, classes)
    normal = list(classes).index('normal')
    att_rows = [i for i in range(len(classes)) if i != normal]
    attack_to_normal = float(cm[att_rows, normal].sum() / max(1, cm[att_rows].sum()))
    utils.save_confusion_matrix_csv(cm, classes, method, out_dir=exp_dir, filename=f"cm_{rid}.csv")
    pd.DataFrame(val_hist).to_csv(os.path.join(exp_dir, f"val_{rid}.csv"), index=False)
    if config['arch'].startswith('fuzzy'):
        torch.save(fl.LAST_RUN['state_dict'], os.path.join(exp_dir, f"global_{rid}.pt"))

    record = dict(run_id=rid, config=config, test=test_metrics, per_class=per_class,
                  poisoned=fl.LAST_RUN.get('poisoned'), client_sizes=fl.LAST_RUN.get('sizes'),
                  val_final=val_hist[-1] if val_hist else None, wall_time_s=round(duration, 2),
                  device=fl.LAST_RUN.get('device'), ever_attacker=fl.LAST_RUN.get('ever_attacker'),
                  test_trigger_asr=fl.LAST_RUN.get('test_trigger_asr'),
                  test_trigger_asr_cond=fl.LAST_RUN.get('test_trigger_asr_cond'), test_attack_to_normal=attack_to_normal,
                  code_hash=manifest['code_hash'], dataset_sha256=manifest['dataset']['sha256'],
                  finished=time.strftime('%Y-%m-%dT%H:%M:%S'))
    tmp = final_path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(record, fh, indent=1, default=str)
    os.replace(tmp, final_path)

    flat = {k: v for k, v in config.items() if k != 'trust_params'}
    flat['trust_params'] = json.dumps(config['trust_params'], sort_keys=True)
    flat.update({f"test_{k}": v for k, v in test_metrics.items() if k != 'Method'})
    flat.update(run_id=rid, wall_time_s=round(duration, 2), code_hash=manifest['code_hash'])
    append_row(os.path.join(args.outdir, 'runs.csv'), flat)
    try:
        os.remove(claim)
    except OSError:
        pass
    print(f"[DONE] {rid}: test Acc={test_metrics['Accuracy']} F1={test_metrics['F1-Score']} ({duration:.0f}s)", flush=True)
    return record


def expand(preset, args):


    seeds = args.seeds if args.seeds is not None else preset['seeds']
    rounds = args.rounds or preset['rounds']
    out = []
    if 'jobs' in preset:
        for job in preset['jobs']:
            m, r, d = job[:3]
            ex = job[3] if len(job) > 3 else {}
            job_seeds = seeds if args.seeds is not None else ex.get('seeds', seeds)
            R = rounds if rounds is None else max(rounds, ex.get('rounds_min', 0))
            out += [(m, s, r, R, d, ex) for s in job_seeds]
        return sorted(out, key=lambda j: j[1]) if (args.seeds is None and not preset.get('job_major')) else out
    methods = args.methods or preset['methods']
    ratios = args.ratios if args.ratios is not None else preset['ratios']
    return [(m, s, r, rounds, preset['data'], preset.get('extras', {})) for s in seeds for r in ratios for m in methods]


def main():
    ap = argparse.ArgumentParser(description="DyF-ZTL experiment driver", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', choices=list(PRESETS))
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--counts', action='store_true', help="print {preset: number of runs} as JSON")
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--rounds', type=int)
    ap.add_argument('--ratios', type=float, nargs='+')
    ap.add_argument('--methods', nargs='+')
    ap.add_argument('--clients', type=int, default=20)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--aggregation', choices=['weighted', 'uniform'], default='weighted')
    ap.add_argument('--weight-basis', dest='weight_basis', choices=['original', 'post_smote'], default='original')
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results_r02'))
    ap.add_argument('--allow-provisional', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if args.counts:
        print(json.dumps({name: len(expand(p, args)) for name, p in PRESETS.items()}))
        return
    if args.list or not args.exp:
        list_presets()
        return
    preset = PRESETS[args.exp]
    locked = os.path.exists(os.path.join(args.outdir, 'protocol.lock'))
    if preset['provisional'] and not locked and not args.allow_provisional:
        sys.exit(f"'{args.exp}' is provisional until {args.outdir}/protocol.lock exists "
                 f"(approve protocol.md first) or use --allow-provisional for a pilot.")
    if args.epochs is None:
        args.epochs = preset.get('epochs', 5)
    if args.methods:
        unknown = [m for m in args.methods if m not in METHODS]
        if unknown:
            sys.exit(f"unknown methods: {unknown}")

    jobs = expand(preset, args)
    if any(j[3] is None for j in jobs):
        sys.exit(f"'{args.exp}' needs R_attack: run the pilot_horizon preset and audit/choose_horizon.py first.")

    os.makedirs(os.path.join(args.outdir, 'manifests'), exist_ok=True)
    manifest = build_manifest(args)
    with open(os.path.join(args.outdir, 'manifests',
                           f"manifest_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.json"), 'w') as fh:
        json.dump(manifest, fh, indent=1)

    for method, seed, ratio, rounds, data, ex in jobs:
        run_one(args.exp, method, seed, ratio, rounds, data, args, manifest, trust_params=ex.get('trust'),
                sim_kwargs=ex.get('sim'), gate_set=ex.get('gate_set'))
    print("\n[ALL DONE]", flush=True)


if __name__ == '__main__':
    main()

