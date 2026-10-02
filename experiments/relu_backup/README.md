# Isolated HyperGSR ReLU experiment

This directory is a code/configuration snapshot. Original project files are
unchanged. Only this copy's HyperDualLearner output becomes
`Linear -> ReLU -> per-graph min-max`, with no learnable shrink parameter.
Legacy shrink configuration fields are accepted but ignored in this copy.
Input arguments, target vectors, prediction shapes and saved result formats
remain unchanged. Other model families are unchanged.

Run from the original project root (`D:/code/HyperGSR`), so relative dataset
paths resolve to the existing data:

```powershell
python experiments/relu_backup/main.py dataset=csv dataset.n_samples=279 experiment.run_name=func160_func268_relu
python experiments/relu_backup/main.py dataset=morph35_func160 experiment.run_name=morph35_func160_relu
python experiments/relu_backup/main.py dataset=morph35_func268 experiment.run_name=morph35_func268_relu
```

Append `model.hyper_dual_learner.use_geo_priors=true` and choose a different
run name for geometry experiments. Results default to `results/relu_experiment`.
Check the resolved configuration without training with `--cfg job --resolve`.
CSV paths and sample counts should match the baseline being compared.

The snapshot imports its own `src` and Hydra configs. Data is shared, not copied.
Future edits to the original code/configuration are not propagated here.
Train from scratch for the ablation; original shrink checkpoints contain an
extra parameter and are not strictly load-compatible. Cross-modal subject/ROI
alignment assumptions are the same as in the original project.
