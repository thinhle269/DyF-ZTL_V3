# DyF-ZTL: code and evidence for the revised manuscript

This package uses the ToN-IoT Windows 10 dataset and the Round-2 v2.2
admission engine. The earlier release is <https://github.com/thinhle269/DyF-ZTL_V02>.
The Python source here has had comments and explanatory docstrings removed;
its file checksums therefore differ from the hashes recorded for the original
runs. The core training and admission expressions were checked for structural
equivalence. An auxiliary pilot script's `__doc__` dependency was replaced
with the same rule text as a normal string, and the command-line help text
was made explicit. The cleaned package has not been rerun.

**New in Round 2** (all in this package): sample-size-weighted FedAvg/FedProx with pre-SMOTE client sizes and the uniform
variants; SCAFFOLD (option-II control variates, solver chosen by a pre-registered pilot); FLTrust, Krum, Multi-Krum,
coordinate-wise median and trimmed mean inside the same pipeline; the dual-evidence admission engine `TrustEvaluatorV2`
(functional, directional and magnitude evidence against a per-round server reference, gates calibrated from honest
evidence, persistent trust, hard/soft/probation modes); server-anchored uniform aggregation of norm-clipped admitted
updates; the threat models (cyclic and targeted flipping, trigger backdoor, boosted backdoor / model replacement,
validation-aware adaptive flip and backdoor, sleeper and rotating attackers, partial participation); trusted-corpus
conditions with gate recalibration; the server-only reference; the per-class chronological split and drift-aware feature
sets; the constrained fuzzy layer (softplus widths, log-space firing) with rule extraction; the multi-process emulation;
explicit round indices and exact grace-round semantics; test-set isolation; run identifiers, manifests and claim trace.

 ## 1. Method names in the code and in the paper

| In `run_experiments.py` | In the paper | Definition |
|---|---|---|
| `DyF-ZTL-V` | DyF-ZTL (proposed, engine v2.2) | DFNN client, `TrustEvaluatorV2` with `functional_ref='reference'`, calibrated gates `<condition>-V-R-U`, uniform aggregation, `virtual_ref=True` |
| `DyF-ZTL-R`, `DyF-ZTL`, `DyF-ZTL-v1` | engine v2.1, v2, Round-1 engine (Table 13) | earlier engine versions, kept as ablations |
| `DyF-ZTL-V-F`, `DyF-ZTL-V-G` | functional-only, geometric-only evidence | `use_geometric=False` / `use_functional=False` |
| `DyF-ZTL-V-soft`, `DyF-ZTL-V-prob` | soft, probation modes | `mode='soft'` / `'probation'` |
| `DyF-ZTL-V-dg` | default gates (no recalibration) | gate set `default-V-R-U` under every trusted-corpus condition |
| `DyF-ZTL-V-CW` | class-weighted loss variant | `loss='weighted_ce'` |
| `DyF-ZTL-V` with `aggregation='weighted'` (jobs `REGISTERED_W`, presets `main`, `agg_v22`) | DyF-ZTL, $|D_k|$-weighted (Tables 7, 12) | reported as `DyF-ZTL-V-W` by the analysis |
| `DeepTrust-V` | MLP + admission engine (w/o fuzzy layer) | MLP client, same engine, own calibrated gates `mlp-V-R-U` |
| `FuzzyNoTrust`, `FuzzyNoTrust-legacy` | DFNN, plain averaging; Round-1 fuzzy layer (regression) | no engine |
| `FuzzyFLTrust` | DFNN + FLTrust rule | FLTrust aggregation with the DFNN client |
| `FedAvg`, `FedProx`, `FedAvg-U`, `FedProx-U`, `FedAvg-CW`, `FedProx-CW` | FedAvg / FedProx (Eq. 1), uniform variants, class-weighted loss | MLP client |
| `SCAFFOLD` (resolved to `SCAFFOLD2-SGD05` by `calibration/scaffold.json`) | SCAFFOLD | option-II control variates, SGD $\eta=0.05$, pilot `pilot_scaffold` |
| `FLTrust`, `Krum`, `MultiKrum`, `Median`, `TrimmedMean` | FLTrust, Krum, Multi-Krum, coordinate-wise median, trimmed mean | MLP client |
| `ServerOnly`, `ServerOnly-MLP` | server-only reference | `defense='server_only'`: the server trains on $D_{\mathrm{trust}}$ alone |
| `DyF-ZTL-cal-V`, `DeepTrust-cal-V`, … | — (calibration runs, seeds 100-102) | `log_only=True`, every client admitted; evidence feeds `audit/calibrate_gates.py` |

## 2 Environment and determinism

Python 3.13.7, PyTorch 2.8.0 (+cu128 build; CUDA was **not** used for the reported runs), NumPy 2.2.6, pandas 2.3.2,
scikit-learn 1.7.2, imbalanced-learn 0.14.0, Windows 10; one workstation with an Intel Core i9-10900X (10 cores / 20 threads),
128 GB RAM and a Quadro RTX 4000 that was hidden from every run. Every reported run executed **single-threaded on CPU**:
`run_parallel.py` sets `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `torch.set_num_threads(1)` in
each lane. A seed fixes the partition, the initialisation, the attacker set, the participation schedule and the mini-batch
order (`src/fl_core.py: seed_everything`, `_batch_generator`). The archived, unmodified code was bit-for-bit reproducible
(65 configurations executed twice by different batches at different times are identical; two configurations re-executed
from the original package are identical in every record; see `result/example_runs/`). Running on the GPU, or with several
BLAS threads, gives slightly different floating-point results that drift over 100 rounds (observed: 97.08 → 96.50 and
94.11 → 92.58 accuracy for the two re-executed runs), within the across-seed variability but not bit-identical.

## 3. How to run

The dataset is not redistributed: place the official `Train_Test_Windows_10.csv` of the ToN-IoT Windows 10 subset
(<https://research.unsw.edu.au/projects/toniot-datasets>) in `dataset/`. Its SHA-256 is recorded in every manifest.
The supplied calibration files are in `result/calibration/`, whereas the runner
defaults to `results_r02/calibration/`. Before starting a run, set
`DYF_CAL_DIR` to the absolute path of `result/calibration/`:

```powershell
$env:DYF_CAL_DIR = (Resolve-Path .\result\calibration).Path
```

`result/` contains selected archived evidence; `results_r02/` is the
default output path for new runs. `results_r02_repro/` contains the original
reproduction `protocol.lock` and receives the two seed-4 example runs.
Use the same calibration directory for every command in a reproduction session.

```
pip install -r requirements.txt
python tests/check_invariants.py                 # 8 invariant groups (aggregation, grace rounds, widths, test isolation, ...)
python audit/data_audit.py                       # cleaning counts, split sizes, client sizes, leakage checks
python run_experiments.py --list                 # every preset: methods x seeds x ratios x rounds (single source of truth)
```
 

```
python repro_run.py main        # DyF-ZTL-V, seed 4, clean   -> results_r02_repro/main/final_DyF-ZTL-V_s4_p0.0_R100_637d44f118.json
python repro_run.py cyclic      # DyF-ZTL-V, seed 4, 40 % cyclic flipping
```

`audit/reproduction_check.py` compares against the complete original
`results_r02/` archive; it cannot validate a fresh run from the selected
records in `result/` alone.

## 4. Experiment matrix and training time

All evaluation runs use 100 communication rounds, 20 clients, 5 local epochs, batch 32 (`pilot_scaffold`: 50 rounds). Wall
times below are per run with 20 runs executing concurrently on the 10-core workstation (one thread each); an isolated
single-thread run of the DFNN takes about 24 minutes (1,436 s clean, 1,462 s under attack for the two re-executed seed-4
runs) and an MLP run about 40-45 minutes (recorded experiment timings). Totals are summed CPU time of the recorded runs.

| Preset | Runs | Content | Mean wall time per run | CPU-hours |
|---|---|---|---|---|
| `main` | 160 | clean comparison, 10 seeds (incl. earlier engine versions, server-only, registered weighted jobs) | 38 min | 102 |
| `agg_ablation`, `agg_v21`, `agg_v22` | 100 | uniform baselines; |D_k|-weighted DyF-ZTL (v2.1, v2.2) | 45 min | 75 |
| `regression` | 12 | constrained vs Round-1 fuzzy layer, 3 seeds | 46 min | 9 |
| `calibrate` | 195 | log-only runs on seeds 100-102 → gate quantiles per regime | 46 min | 148 |
| `pilot_horizon`, `pilot_scaffold`, `pilot_anchor`, `pilot_agg`, `pilot_uniform` | 100 | pre-registered pilots on seeds 100-102 | 15-59 min | 68 |
| `cyclic` | 1,140 | label flipping 0-90 %, 9 methods × 10 seeds (+ earlier engine versions, controls) | 40 min | 761 |
| `attacks` | 879 | targeted, backdoor, boosted, adaptive, sleeper, rotating, partial participation; 5 seeds | 43 min | 637 |
| `trust_modes` | 56 | hard / soft / probation | 45 min | 42 |
| `trusted_set` | 338 | 8 corpus conditions × recalibrated / default gates × FLTrust | 42 min | 239 |
| `sensitivity` | 324 | 18 settings × 2 ratios × 5 seeds (incl. earlier versions) | 44 min | 240 |
| `heterogeneity` | 127 | α ∈ {0.1, 1, 10, IID}, no-SMOTE, class-weighted loss | 35 min | 74 |
| `chrono`, `drift` | 143 | per-class chronological split; drift-robust / oracle feature sets | 46-51 min | 119 |
| `server_only_sizes` | 75 | server-only and DyF-ZTL at |D_trust| = 100 / 300 / 1,200 | 9 min | 11 |
| **Total** | **3,649** | | | **2,525 CPU-hours** |

The runs were scheduled across up to 20 lanes (about 5.3 days of wall-clock time);
engine versions v2 and v2.1 were evaluated first and are retained as ablations. The 111 runs that were superseded by a code
fix (probation rule, SCAFFOLD control-variate weights, heterogeneity gates) are archived outside the
analysis (`results_r02/_invalid/`, not included here). The multi-process emulation (24 scenarios of 20 rounds, 10-40 client
processes)  

## 5. The `result/` folder

| Path | Content | Used in |
|---|---|---|
| `figures/` | paper figure images, including the three confusion matrices, three membership plots, consequent weights and firing plots copied from the manuscript source | Figs 2, 4-24 |
| `tables_csv/tab_*.csv` | 20 CSV exports from the base analysis | Most numerical tables; Tables 13 and 20 use `Extension/` evidence |
| `MANIFEST_figures_tables.csv` | source of each figure/table | — |
| `analysis/*.csv` | `runs_flat.csv` (one row per run: configuration, test metrics, run id, file location), admission metrics per run, completeness, paired tests, per-class, sensitivity, heterogeneity, chrono, regression | Sections 6-7 |
| `analysis/reviewer/R1.*.csv` | evidence tables per reviewer comment | Response letter |
| `fuzzy_diagnostics/` | extracted rules, approximate rules, rule stability, client variation, numerical check | Table 21; Figs 16-19 |
| `systems/` | per-scenario emulation records (`*_rounds.csv`, `*.json`), `summary.csv` | Table 23; Fig. 24 |
| `calibration/` | gate constants per regime (`gates_*.json`), pre-registered decisions (`horizon.json`, `scaffold.json`, `anchor.json`, `agg_variant.json`), backdoor trigger (`trigger.json`) | §4.2.2, §4.4, §5.4, Appendix A |
| `partitions/` | per-seed client partitions (class histograms before and after SMOTE, SMOTE status) | §5.1, Table 2 |
| `audit/` | `claim_trace.csv` (904 numbers of the paper → run ids and files, 797 recomputed independently), dataset and drift audits (CSV) | Provenance checks |
| `example_runs/` | the records of the two re-executed configurations, stored and reproduced versions, seed-4 checkpoints and a batch manifest | Reproduction examples |

## 6. Mapping code and evidence to the manuscript

| Paper component | Code and evidence in this package |
|---|---|
| Client model and training (Sections 3-5) | `src/models.py`, `src/fl_core.py`, `src/preprocessing.py`, `run_experiments.py` |
| Reference, evidence, trust and admission (Section 4) | `src/trust_engine.py`, `src/aggregators.py`; calibrated constants in `result/calibration/` |
| Comparisons and attacks (Section 6) | `run_experiments.py`, `analyze_r02.py`, `result/analysis/`, `result/tables_csv/` |
| Policy-aware attacks (Table 13) | `Extension/E2_full_knowledge/` contains the summary and per-run table; `Extension/scripts/analyze_extension.py` analyses those results |
| Selection-seed checks (Appendix) | `Extension/E1_seeds_10_14/` and `Extension/E3_seeds_2_9/` |
| Fuzzy memberships and consequent weights (Figs 16-19) | images in `result/figures/`; seed-4 checkpoint in `result/example_runs/reproduced/main/` |
| Rule-node firing (Table 20; Figs 20-22) | `Extension/E4_firing/` and copied images in `result/figures/` |
| Multi-process emulation (Fig. 24; Table 23) | `systems/`, `result/systems/` |

Figures 1 and 3 are drawn in the manuscript LaTeX source, and Figure 2 uses
`result/figures/F02_fuzzy.png`. Figures 16-19 are supplied as the actual
manuscript images; this package has no standalone generator for those images.
The earlier combined `fig_confusion.png` and `fig_membership.png` are retained
as archival outputs and are not the separate figures used in the revised paper.
`src/generate_paper_figures.py` and `src/fix_index_error.py` are earlier
utilities, not the current v2.2 figure-generation pipeline.
The complete per-run records and model checkpoints for all seeds are in the
<https://drive.google.com/drive/folders/10fiTUJ5MvUAJ5qKcpgoqa2mhH3MytxDM>
release. Only the two seed-4 reproduced checkpoints are copied into this
compact code folder. The historical scripts in `Extension/scripts/` also
contain workstation-specific paths and may require adaptation before reuse.
The executable runner that produced the surrogate and white-box oracle
policy-aware attacks in Table 13 was not found in the source folders supplied
with this package. Their results are available in `Extension/E2_full_knowledge/`
and the complete run archive, but `run_experiments.py` alone does not
reproduce that table. Do not infer the missing attack implementation from the
summary CSVs.

The per-run records of all 3,649 runs (about 400 MB with model files) are kept with the authors and in the archived
release; `analysis/runs_flat.csv` lists every run with its identifier and file location so that any number of the paper can be
located (`audit/claim_trace.csv` does this mapping for every table cell).

## 7. Notes and limitations

* The preset `scaffold_check` (SCAFFOLD step-size check) is defined but was not run: the pilot-selected solver was stable
  in every seed after the control-variate fix, so the check was unnecessary (`result/analysis/completeness.csv`).
* `run_experiments.py` gained presets during the batch (`server_only_sizes`, …). The archived
  manifests record the original source hashes, which differ from this comment-cleaned copy.
* Running `run_experiments.py` directly on a machine with a GPU trains on CUDA and does not reproduce the reported numbers
  bit-for-bit (§5); use `run_parallel.py` or `repro_run.py`, which set the environment.
* The dataset licence requires citing the eight ToN-IoT papers; the data are not redistributed with this package.
