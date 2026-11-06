# Source-Split Sensitivity and Accepted Loss in Frozen Multimodal Reliability Selection

This repository contains the analysis code, cohort indices and numerical inputs for
the study of fixed-capacity reliability selection on VQA v2, DocVQA and
ActivityNet-QA. The [versioned release](https://github.com/KleinChen42/source-split-reliability/releases/tag/v1.0.0)
includes the post-review thirty-seed results.

## Representation invariance and deployment extension (v1.1.0)

The [v1.1.0 release](https://github.com/KleinChen42/source-split-reliability/releases/tag/v1.1.0)
adds real ActivityNet-QA experiments on the original and additional holdouts:

- Original, fourfold replicated and orthogonally transformed visual features;
  coordinate, coordinate-balanced and block-energy normalizations.
- Thirty paired development seeds, four frozen model/prompt candidates and a
  six-penalty ridge grid: 15,840 ridge fits plus 240 text/native tree fits.
- Capacity-matched policy selection, finite source-grouped queue replay and
  synchronized hardware measurements on two RTX 5090 GPUs.
- A new 720-endpoint interval family with 1,999 paired source draws.

Use the source tree at tag **v1.1.0**, and extract
**CAAI_Invariance_Experiments_20261106.zip** at its repository root.
[Experiment and replay instructions](revision/caai_20261106_innovation/experiments/README.md)
explain how to regenerate numerical summaries without model or media files,
and how to rerun fits when reconstructed banks are available.

The extension is exploratory on inspected H/H2 cohorts. It evaluates an
invariance property of the selection procedure; block normalization and
ridge regression themselves are established methods. Its capacity/queue
curves are descriptive, and its timings cover the stated model/feature
boundary rather than end-to-end production service latency.

## Thirty-seed numerical replay

Download and extract **NEUCOM_21462_Thirty_Seed_Results.zip** from the release.
The archive contains portable policy orders and losses, selected-policy tables,
bootstrap draws, and the complete adjusted-interval tables. For each of VQA v2
and ActivityNet-QA it covers the original H and H2 cohorts. The four dataset/cohort analyses
share one maximum-deviation critical value over 604,590 endpoints. Each cohort
uses 1,999 paired source resamples at 70% acceptance. The video H2 resampling
preserves the original train/validation source counts.

Run from the repository root with Python 3.10+ and NumPy. Substitute the extracted
archive directory for THIRTY_SEED_RESULTS.

~~~sh
python revision/neucom_21462_20261104/recovery/analyze_extensions.py --replay-portable THIRTY_SEED_RESULTS/analysis --output replay
~~~

This command rebuilds the numerical source queues from the saved orders and
losses. The default CPU route is the reference implementation. With a
CUDA-compatible PyTorch installation, append --device cuda to use the GPU route.
Use a fresh output directory when changing the replay input or execution settings.

Key files in the asset include analysis/cell_intervals.csv and
analysis/pairwise_intervals.csv.gz, together with cell_points.csv,
seed_details.csv and capacity_points.csv in each cohort directory. The
multiplicity_method.json file defines the adjusted intervals. The extension is
exploratory on the inspected H and H2 cohorts, conditional on the observed
development records and fitted seed collection. The separately specified
original five-seed H2 evaluation is preserved in the historical materials.

The fixed-candidate ridge weight figure can be rendered from the interval table
with Matplotlib.

~~~sh
python revision/neucom_21462_20261104/recovery/plot_block_weights.py THIRTY_SEED_RESULTS/analysis/cell_intervals.csv block_weight_paths.pdf
~~~

## Historical numerical replay

The earlier arrays, tables and analysis code retain their project-relative
paths. With NumPy available, run these commands from the repository root.

~~~sh
python revision/neucom_21462_20261104/analyze_review_requests.py
python revision/neucom_21462_20261104/enrich_saved_results.py
~~~

## Cohort identities and model regeneration

The file revision/neucom_21462_20261104/recovery/prepared/all_record_source_indices.csv
lists the original record, source and page IDs and official split for 67,675
records. The F/V split files cover seeds 0–29. Acquisition scripts in recovery/
rejoin annotations and selected media to the released indices from the dataset
providers. The public files comprise indices, code and processed numerical
outputs. Dataset and checkpoint access is governed by their providers.

To regenerate the VQA scores, obtain the official [VQA evaluation module](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py)
as neurocomputing_transfer/vendor_sources/vqaEval.py. See the acquisition scripts
and their --help options for the benchmark inputs. For the frozen models, the
pinned ModelScope revisions are listed in pinned_model_downloads.json.
RUN_ENVIRONMENT.json in the numerical asset records the runtime of the extension.

Use rebuild_gpu_inputs.py for the visual, predictions and assemble stages after
the benchmark inputs are present. Generate each dataset/partition bank with
the corresponding isolated model environment. Qwen's canonical prompt
prediction step also captures the pre-answer question vector. The
--reuse-frozen-vision switch reuses encoder outputs for identical visual inputs.
Run run_fit_queue.py --gpu-ready --analysis-device cuda after assembly to apply
F-only fitting, V-only selection and H/H2 evaluation. Individual completed
units and partial source-bootstrap batches are resumable.

