# Equivalent-encoding and operational experiments

Completed on real ActivityNet-QA banks on two NVIDIA RTX 5090 GPUs,
6 November 2026. The plan was written before inspecting the new experiment
outcomes; the H/H2 cohorts had been inspected previously. This is an exploratory
extension, not an independent confirmatory test.

## Obtain and replay the saved results

Use the source tree at tag **v1.1.0** of
https://github.com/KleinChen42/source-split-reliability and extract the release
asset **CAAI_Invariance_Experiments_20261106.zip** at the repository root.
The archive retains project-relative paths. All commands below run at that root.

```sh
python revision/caai_20261106_innovation/experiments/analyze_invariance.py --results revision/caai_20261106_innovation/experiments/results --output replay_invariance --stage points --device cpu
python revision/caai_20261106_innovation/experiments/analyze_invariance.py --results revision/caai_20261106_innovation/experiments/results --output replay_invariance --stage bootstrap --device cpu
```

The first rebuilds descriptive summaries and finite queues; the second rebuilds
1,999 paired source draws and the 720-endpoint simultaneous interval family.
Append `--device cuda` in place of `--device cpu` for GPU resampling. Use a fresh
output directory for a different execution setting. The saved `analysis/`
directory already contains the completed outputs; replay is optional.

The analysis needs NumPy and the existing recovery/analyze_extensions.py module
included in the tagged repository. GPU inference additionally needs PyTorch.
No model/media files or fitted-object deserialization are needed to replay
orders and losses. The fitted estimators are supplied separately as joblib
files for inspection or prediction with a compatible fitting environment.

## Regenerate fits from reconstructed feature banks

Use the repository's recovery scripts to reconstruct `activitynetqa_D.npz`,
`activitynetqa_H.npz` and `activitynetqa_H2.npz` under
`revision/neucom_21462_20261104/recovery/rebuilt/banks/`. The original model,
benchmark acquisition and pinned checkpoint instructions remain in the base
release. The existing 30-seed split file fixes record/source membership.

Run these in separate GPU environments, assigning one visible GPU to each:

```sh
CUDA_VISIBLE_DEVICES=0 python revision/caai_20261106_innovation/experiments/run_invariance.py --output revision/caai_20261106_innovation/experiments/results --flow R
CUDA_VISIBLE_DEVICES=1 python revision/caai_20261106_innovation/experiments/run_invariance.py --output revision/caai_20261106_innovation/experiments/results --flow S
```

The recorded run used Python 3.10, NumPy 1.24.4, scikit-learn 1.2 and
PyTorch 2.13.0+cu130. Each fit uses float64 Cholesky and four CPU threads.
For Windows PowerShell, set the corresponding environment variable with
`$env:CUDA_VISIBLE_DEVICES` before invoking Python.

`run_invariance.py` saves validation selection before reading holdout utilities.
It fits 11 feature geometries x 6 penalties x 4 candidates x 60 units = 15,840
ridge models, plus 240 text/native tree models. The fixed-alpha and pooled
policies share those fits. Native confidence remains eligible in pooled panels.

## Hardware timing

After preparing decoded video frames and the pinned local Qwen and InternVL
models, run each command on its own otherwise-idle GPU:

```sh
python revision/caai_20261106_innovation/experiments/measure_cost.py --model qwen --output revision/caai_20261106_innovation/experiments/results/cost
python revision/caai_20261106_innovation/experiments/measure_cost.py --model internvl --output revision/caai_20261106_innovation/experiments/results/cost
```

The fixed sample contains one question from each of 64 real development videos
plus three warm-up videos. Timings synchronize the GPU and include preprocessing,
transfer and model execution. Disk access, frame decoding and loading are
excluded. Qwen direct generation alternates capture/no-capture order; a separate
one-token pass supplies canonical states for other generator/prompt routes.
The InternVL job also times ResNet on eight frames per video.

## Interpretation

`cell_intervals.csv` and `pairwise_intervals.csv.gz` contain the new family;
`multiplicity_method.json` defines its scope. The 720 endpoints cover 15 pooled
panels and their pairs, three outcomes and two cohorts at 70% acceptance.
Capacity curves, fixed-alpha encoding sensitivity and finite queues are
descriptive. `changed_accepted_records` counts original acceptances replaced,
not the size of a symmetric difference. H2 bootstrap draws preserve its
780/215 source mixture. Inference conditions on D and the fitted seed collection.
