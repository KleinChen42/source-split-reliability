# Representation-invariance extension

The real-data fitting, analysis and GPU timing instructions are in
[experiments/README.md](experiments/README.md). The full numerical outputs
are the CAAI_Invariance_Experiments_20261106.zip asset in release v1.1.0.

Render the new encoding, operational and graphical-summary plots from the
included compact data with NumPy and Matplotlib:

```sh
python revision/caai_20261106_innovation/build_extension_figures.py
```

The command writes vector PDFs to figures/ and numerical table rows to sections/.
The imported build_figures.py supplies plotting styles and the pipeline routine;
its older all-figures entry point requires the preceding manuscript plotting
inputs. To render only the updated pipeline, import and call draw_pipeline()
after the figures/ directory has been created.

The 720-endpoint interval family covers the new pooled 70% policies.
Additional capacities, fixed-candidate changes and finite queues are descriptive.
The older 604,590-endpoint family remains available in v1.0.0.
