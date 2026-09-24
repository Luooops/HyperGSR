# HyperGSR

A brain graph super-resolution research project that predicts high-resolution connection weights from low-resolution adjacency matrices. The current CSV configuration maps **160 source nodes to 268 target nodes**. The project includes HyperGSR, STP-GSR, DirectSR, and a separate IMANGraphNet baseline.

Training and analysis use local files and do not require uploading data. Dependency installation accesses package repositories; use an existing local package repository if installation must also remain offline.

## 1. Environment setup and quick start

Run all commands from the project root. Relative paths are interpreted from the working directory.

### Create a Conda environment and install dependencies

The following commands are instructions for manual execution. They create an environment and install packages; **they do not start training or evaluation**. Conda must already be installed and available in your terminal.

Python 3.10 is suggested for the project's pinned dependencies. Compatibility with every Python/CUDA combination has not been verified. `requirements.txt` contains the dependencies for training, metric evaluation, statistical analysis, and plotting in one file.

```powershell
# Create a dedicated Conda environment with Python 3.10
conda create --name hypergsr python=3.10 pip -y

# Activate the environment
conda activate hypergsr

# Install all project dependencies: training, evaluation, analysis, and plotting
python -m pip install -r requirements.txt
```

`requirements.txt` explicitly selects `torch==2.11.0+cu130` from the official PyTorch CUDA 13.0 index, preventing the CPU build from satisfying the requirement. The official index provides a Windows x64 / Python 3.10 wheel for this version: [PyTorch CUDA 13.0 packages](https://download.pytorch.org/whl/cu130/torch/). This build successfully executed a small CUDA tensor operation on the local RTX 5080 in a separate existing environment; the complete project dependency combination and training have not yet been verified with it.

To update an existing `hypergsr` environment, stop any running training process first, then run:

```powershell
conda activate hypergsr
python -m pip install -r requirements.txt
python -m pip check

# Verify the installed build and execute a CUDA operation
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print((torch.ones(2, device='cuda') + 1).cpu())"
```

Expected output includes `2.11.0+cu130`, `13.0`, `True`, the RTX 5080 device name, and `tensor([2., 2.])`. Installing the new exact requirement replaces the old CPU-only torch package without recreating the Conda environment. No installation is performed automatically by this project. This dependency file targets CUDA-capable systems rather than macOS/CPU-only portability.

GPU support depends on the installed PyTorch build and local driver. The training code uses CUDA when available and otherwise uses the CPU. Full-size STP-GSR experiments require substantial memory. Changing PyTorch versions can affect numerical results, so run all compared ablations in the same environment.

### Inspect the configuration without training

Hydra builds the **combined experiment configuration** by merging:

1. `configs/experiment.yaml`, including its selected dataset and model defaults;
2. the selected dataset file, such as `configs/dataset/csv.yaml`;
3. the selected model file, such as `configs/model/stp_gsr.yaml`;
4. command-line overrides, such as `model=hyper_gsr` or `experiment.n_epochs=2`.

Here, "combined configuration" refers to merging settings, not generating synthetic data. `--cfg job` prints the resulting application configuration and exits without calling the training function. The environment dependencies must still be installed because Python imports the entry-point modules first.

```powershell
# Print the default configuration: CSV data and STP-GSR
python main.py --cfg job

# Print the configuration with HyperGSR selected instead
python main.py model=hyper_gsr --cfg job

# Inspect an additional override without starting training
python main.py model=hyper_gsr experiment.n_epochs=2 --cfg job
```

### Train and evaluate

```powershell
# Train HyperGSR; use a new run_name to avoid overwriting an existing run
python main.py model=hyper_gsr experiment.run_name=hyper_local_01

# Compute metrics from saved predictions; match the training run's settings
python evaluate.py model=hyper_gsr experiment.run_name=hyper_local_01
```

Running `python main.py` without overrides trains **STP-GSR**, not HyperGSR. When evaluating a run, match its model, dataset, backend mode, and run name.

## 2. Project structure and module responsibilities

```text
HyperGSR/
├── main.py                       # Hydra entry point for K-fold training
├── evaluate.py                   # Saved-prediction metrics and fold averages
├── main_iman.py                  # Separate IMANGraphNet training entry point
├── MatrixVectorizer.py           # Compatibility import for the original path
├── plot_comparison.py            # STP / Hyper EdgeAttr / Hyper Coord plots
├── statistical_analysis.py       # Pairwise statistical tests, tables, and plots
├── run_statistical_analysis.py   # Analysis entry point with default run paths
├── requirements.txt              # All training, evaluation, and analysis dependencies
├── configs/
│   ├── experiment.yaml           # Dataset/model selection, training, and folds
│   ├── hydra.yaml                # Legacy environment settings; not composed by default
│   ├── dataset/                  # csv, er, ba, sbm, kronecker
│   └── model/                    # hyper_gsr, stp_gsr, direct_sr
├── src/
│   ├── experiment.py             # Shared result paths and original seed calls
│   ├── dataset.py                # Loading, graph generation, PyG data, ROI coordinates
│   ├── train.py                  # Single-fold training, model setup, L1 validation
│   ├── matrix_vectorizer.py      # CSV vectors and symmetric adjacency matrices
│   ├── dual_graph_utils.py       # STP dual graph and edge-vector conversion
│   ├── hyper_graph_utils.py      # Sparse incidence structures and geometry
│   ├── eval_metrics.py           # Per-sample and aggregate graph errors
│   ├── plot_utils.py             # Training curves, adjacency/gradient plots, GIFs
│   ├── plotting.py               # Fold/average comparisons between two metric CSVs
│   └── models/
│       ├── build.py              # Model construction from graph sizes and model settings
│       ├── direct_sr.py          # Direct high-resolution adjacency prediction
│       ├── stp_gsr.py            # Edge initialization and explicit dual-graph learning
│       ├── hyper_gsr.py          # Full HyperGSR model and geometry cache
│       ├── hyper_layers.py       # Initialization, bipartite message passing, readout
│       └── imangraphnet/         # Separate baseline implementation; see below
├── gsr_data/                     # Local CSVs, coordinates, and parcellation files
└── results/                      # Weights, predictions, metrics, and figures
```

The IMANGraphNet implementation retains its original organization and is not registered in the main model factory:

| Module | Responsibility |
| --- | --- |
| `config.py` | Node counts, sample count, and training epochs |
| `model.py` | Aligner, Generator, and Discriminator networks |
| `load_data.py` | LR/HR CSV loading into PyG objects |
| `dataset.py` | Data-format adaptation helpers |
| `preprocess.py` | Vector, matrix, and graph conversion |
| `losses.py` | Alignment, reconstruction, and adversarial losses |
| `centrality.py` | Centrality and topological measures |
| `prediction.py` | Original training/prediction workflow and weight restoration |
| `plots.py` | Original plotting helpers |
| `main.py` | Retained original experiment script; use root-level `main_iman.py` for the project entry point |

## 3. Data and model workflow

```text
CSV / synthetic graphs
  → dataset.load_dataset(config)
  → aligned source_data / target_data lists
  → main.py: K-fold splits using matching sample indices
  → train.load_model(config)
      → models.build.build_model(graph sizes, model_config)
  → train.train(...): per-sample forward, L1 loss, gradient accumulation, Adam
  → train.evaluate_model(...): predictions reconstructed as adjacency matrices
  → per-fold .npy / .pth / figure files
  → evaluate.py: metrics.csv
  → statistical analysis and comparison plots
```

### CSV conventions

`configs/dataset/csv.yaml` specifies:

| Parameter | Default / meaning |
| --- | --- |
| `source_csv_path` | `gsr_data/lr_train_.csv` |
| `target_csv_path` | `gsr_data/hr_train_.csv` |
| `roi_coords_csv` | `gsr_data/coords_hr.csv` |
| `n_source_nodes` / `n_target_nodes` | 160 / 268 |
| `n_samples` | Use at most the first 100 samples |
| `node_feat_init` | `adj`: adjacency rows serve as node features |

The loader skips the first CSV row and removes the first column. The remaining values represent the **upper triangle, traversed column by column, excluding the diagonal**. There are 12,720 values for 160 nodes and 35,778 for 268 nodes. Source and target samples are paired by row order, not joined by ID; input files must have matching sample counts and ordering.

`load_dataset` returns two lists whose entries have the form `{'pyg': PyG Data, 'mat': Tensor}`. The main workflow uses PyG fields `x`, `pos_edge_index`, and `edge_attr`, which differ from some intermediate IMANGraphNet representations.

The ROI coordinate CSV must contain `x,y,z` columns in target-node order. The current implementation loads coordinates for every HyperGSR configuration. The NIfTI parcellation file and LR coordinate file are not used directly by the main training entry point.

### Main models

- **DirectSR** applies two graph Transformer layers and predicts the HR matrix through `XᵀX`.
- **STP-GSR** initializes HR connections, represents them as dual-graph nodes, and learns their weights.
- **HyperGSR** initializes connections and applies two-step message passing: connection nodes → ROI hyperedges → connection nodes. Optional components include distance features, coordinate features, ROI embeddings, and output shrinkage.

STP-GSR and HyperGSR originally use different initialization normalization, so their initializers remain separate. Model `forward` methods receive the target matrix to construct supervision; it is not used to compute predictions.

## 4. Configuration boundaries and Python API

`dataset` controls data, `model` controls the network, and `experiment` controls training and experiment organization. Entry points and training orchestration use the full configuration. `build_model` receives only graph sizes and `model_config`; it does not read result paths, seed RNGs, or select a device.

```python
from hydra import compose, initialize
from src.dataset import load_dataset, load_roi_coords_csv
from src.experiment import seed_experiment
from src.models.build import build_model

# Use from an interactive session in the project root.
# Scripts can alternatively use initialize_config_dir with an absolute path.
with initialize(version_base="1.3.2", config_path="configs"):
    config = compose(config_name="experiment", overrides=["model=hyper_gsr"])

seed_experiment(config.experiment.kfold.random_state)
source_data, target_data = load_dataset(config)
model = build_model(
    n_source_nodes=config.dataset.n_source_nodes,
    n_target_nodes=config.dataset.n_target_nodes,
    model_config=config.model,
)
coords = load_roi_coords_csv(config.dataset.roi_coords_csv)
prediction, supervision = model(
    source_pyg=source_data[0]["pyg"],
    target_mat=target_data[0]["mat"],
    roi_coords=coords,
)
# Both HyperGSR outputs have shape [n_target_nodes * (n_target_nodes - 1) // 2, 1].
```

Single-fold training uses named arguments. The four data lists below must already be split using matching sample indices. Create the output directory before calling `train`:

```python
from pathlib import Path
from src.train import train, evaluate_model

fold_dir = "results/manual_example/fold_1"
Path(fold_dir).mkdir(parents=True, exist_ok=True)
result = train(
    config=config,
    source_data_train=source_train,
    target_data_train=target_train,
    source_data_val=source_val,
    target_data_val=target_val,
    res_dir=fold_dir,
)
predictions, loss = evaluate_model(
    config=config,
    model=result["model"],
    source_data=source_val,
    target_data=target_val,
    criterion=result["criterion"],
    roi_coords_cpu=result["roi_coords_cpu"],
)
```

Compatibility is retained for `HyperGSR(config)`, `STPGSR(config)`, `DirectSR(config)`, `src.train.load_model(config)`, the original `eval(..., critereon=...)`, `result['critereon']`, and the root-level `MatrixVectorizer` import. New callers should use `criterion` and `evaluate_model`. `src.train.load_roi_coords_csv` remains importable; its implementation now lives in `src.dataset`.

## 5. Training and command-line overrides

Defaults: 60 epochs, learning rate 0.001, an optimizer update after accumulating 16 samples, 3-fold cross-validation, seed 42, and no per-epoch validation-loss logging.

```powershell
python main.py model=stp_gsr experiment.run_name=stp_local_01
python main.py model=direct_sr experiment.run_name=direct_local_01
python main.py model=hyper_gsr experiment.run_name=hyper_local_01

# Override epochs, accumulation frequency, learning rate, and validation logging
python main.py model=hyper_gsr experiment.run_name=hyper_short_01 experiment.n_epochs=2 experiment.batch_size=8 experiment.lr=0.001 experiment.log_val_loss=true

# Enable both distance and ROI-coordinate features
python main.py model=hyper_gsr experiment.run_name=hyper_geometry_01 model.hyper_dual_learner.use_geo_priors=true

# Select a different message-passing backend
python main.py model=hyper_gsr experiment.run_name=hyper_spmm_01 model.hyper_dual_learner.mode=spmm
```

Main `hyper_dual_learner` settings:

| Parameter | Current YAML default | Meaning |
| --- | --- | --- |
| `mode` | `trans` | `spmm` / `sage` / `gat` / `trans` |
| `hidden_dim` / `heads` | 32 / 4 | Hidden dimension and attention heads |
| `dropout` | 0.1 | Message-passing dropout |
| `edge_dim` | 1 | Transformer incidence-edge feature dimension; `trans` with 0 actually uses GAT |
| `use_hyper_emb` | true | Learn ROI embeddings; coordinate features use the coordinate MLP instead |
| `use_geo_priors` | false | Enable both distance features and coordinate-derived ROI features; disable the independent ROI embedding table when enabled |
| `edge_geo_dim` / `dist_norm` | 8 / `zscore` | Distance embedding dimension and normalization |
| `use_shrink_output` | true | Enable output shrinkage |
| `shrink_threshold` | 0.01 | Initial value of the learnable threshold, not a fixed threshold |

`use_geo_priors` is the single geometry switch. With `true`, the model creates both `edge_geo_mlp` and `roi_mlp`, concatenates distance features to connection inputs, and uses coordinate-derived ROI representations. In the `trans` branch with `edge_dim>0`, it also supplies scalar distances as incidence attributes, requiring `edge_dim=1`. With `false`, both geometry MLPs are absent, connection inputs remain one-dimensional, and ROI initialization follows `use_hyper_emb`. The non-geometric Transformer branch retains its original learned fallback attributes.

The former `use_edge_distance` and `use_hyper_coords` configuration fields and Python constructor arguments have been replaced by `use_geo_priors`; update old commands/configurations rather than adding the removed keys with Hydra's `+` syntax. Only the former both-on and both-off configurations are represented. Model parameter names and shapes for those two cases are unchanged. The existing `spmm` limitation remains: coordinate-derived ROI features are not consumed by that backend.

Although `in_dim` remains in the YAML, the full HyperGSR model determines its input dimension from the geometry-prior flag. Not all backend/geometry combinations are supported by the current implementation; see the limitations below.

### Small workflow smoke test

Use small synthetic graphs to exercise the main workflow without reading real CSV data:

```powershell
python main.py model=direct_sr dataset=er dataset.n_source_nodes=8 dataset.n_target_nodes=12 dataset.n_samples=24 experiment.n_epochs=1 experiment.run_name=smoke_direct
python evaluate.py model=direct_sr dataset=er experiment.run_name=smoke_direct
```

The target-node count must satisfy the model's attention-head divisibility constraint. With 3 folds, 24 samples avoid an out-of-range error when the original entry point plots the seventh validation sample. This is a workflow check; synthetic source and target graphs are generated separately.

### IMANGraphNet

```powershell
python main_iman.py
```

This entry point uses a separate configuration, hardcoded CSV paths, and 3-fold training. It does not accept the main entry point's Hydra overrides. Outputs go to `results/imangraphnet/fold_*`. It does not produce the same validation-prediction files as the main workflow and cannot feed directly into `evaluate.py`. This baseline was not changed or fully training-tested during the refactor.

## 6. Outputs, metrics, and weight restoration

`src.experiment.get_run_dir(config)` defines the result layout:

```text
results/<model>/<dataset>/<run_name>/                  # STP / Direct
results/hyper_gsr/<dataset>/<mode>/<run_name>/          # HyperGSR
  fold_1/
    model.pth               # state_dict
    eval_output.npy         # [N_val, n_target_nodes, n_target_nodes]
    source.npy / target.npy # Inputs and ground truth for the same validation fold
    train_losses.npy / val_losses.npy
    train_loss.png / val_loss.png
    gradient_flow.gif / train_samples.gif
    eval_sample6.png
    stats.txt               # Training time and sampled GPU memory usage
  fold_2/ ...
  fold_3/ ...
  metrics.csv               # Created by evaluate.py
```

`evaluate.py` reads each fold's `eval_output.npy` and `target.npy`; it does not reload model weights for inference. It reports eight metrics: connection-weight MAE; degree, betweenness, eigenvector, PageRank, and Katz centrality errors; clustering-coefficient difference; and Laplacian Frobenius distance. Fold averages are arithmetic means of the per-fold metrics.

To restore your saved weights, first construct the model with the training configuration:

```python
import torch
from src.dual_graph_utils import revert_dual

model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
model.eval()
with torch.no_grad():
    edge_weights, _ = model(source_data[0]["pyg"], target_data[0]["mat"], coords)
    predicted_matrix = revert_dual(edge_weights, config.dataset.n_target_nodes)
```

This example uses HyperGSR on CPU. `checkpoint_path` is a local weight file; other variables come from the earlier model-construction example. Submodule attribute names and `state_dict` keys are preserved. Geometry caches are not stored in the weights, so a newly constructed model still requires coordinates on its first forward pass. Complete configurations are not consistently stored for historical runs; directory names alone do not establish all geometry settings.

## 7. Statistical analysis and plotting

```powershell
# Analyze the original default experiment pair
python run_statistical_analysis.py

# Select two metric CSVs and an output directory explicitly
python run_statistical_analysis.py --model1_path results/stp_gsr/csv/stp_local_01/metrics.csv --model2_path results/hyper_gsr/csv/trans/hyper_local_01/metrics.csv --model1_name STP-GSR --model2_name HyperGSR --results_dir results/comparison_01

# Call the underlying analysis script directly
python statistical_analysis.py --model1_path results/stp_gsr/csv/stp_local_01/metrics.csv --model2_path results/hyper_gsr/csv/trans/hyper_local_01/metrics.csv

# Plot the three original experiment groups without opening windows
python plot_comparison.py --output_dir results/comparison_plots --no_show

# Inspect options for --stp_path, --edgeattr_path, and --coord_path
python plot_comparison.py --help

# Run the original two-run, per-fold plotting example
python -m src.plotting
```

Python API:

```python
from plot_comparison import load_data, create_bar_plots
from src.plotting import plot_metrics_compare

data = load_data(stp_gsr_path="a.csv", hyper_edgeattr_path="b.csv", hyper_coord_path="c.csv")
create_bar_plots(data=data, output_dir="results/plots", show=False)
plot_metrics_compare(csv_a="a.csv", csv_b="b.csv", labels=("A", "B"), out_dir="results/plots")
```

The statistical script excludes the `average` row, performs paired t-tests and Wilcoxon tests across folds, and exports CSV, text, LaTeX, and figures. Both inputs should have matching fold order and metric columns. Tests, effect-size calculations, and significance rules retain their original behavior; the structural refactor does not revalidate the statistical methodology. `STATISTICAL_ANALYSIS_README.md` contains historical examples whose comparison names and information-centrality metric do not fully reflect the current code; use this README and the scripts for current behavior.

## 8. Compatibility and validation

The original structural refactor changes module organization, shared helpers, explicit arguments, and documentation. It preserves network computations, parameter registration order, training loops, losses, gradient accumulation, seed-call timing, default YAML settings, evaluation formulas, and existing experiment outputs. It does not modify `gsr_data/` or `results/`.

During the refactor, structural comparisons and numerical geometry/vectorization checks passed. The temporary Git-baseline regression test was subsequently removed during cleanup. Full-model numerical checks and end-to-end training were not executed because the validation environment lacked PyG/Hydra. The requirements have not been installation-tested in a fresh environment.

The following existing behaviors remain unchanged:

- `batch_size` controls per-sample gradient accumulation. Loss is not divided by the accumulation count, and the final partial batch still triggers an update.
- Each fold saves its final-epoch model; there is no best-model selection or automatic resume.
- `main.py` plots validation index 6, and its progress message still hardcodes a total of 3 folds.
- HyperGSR always requires coordinates. Synthetic dataset configurations lack `roi_coords_csv`, so additional configuration is needed before using them with HyperGSR.
- The default non-geometric attribute branch retains `LayerNorm(1)` and its original numerical behavior.
- The `spmm` branch does not use ROI-coordinate MLP outputs. Some configurations can leave parameters without gradients, while the original gradient plotter assumes gradients exist.
- For 268 target nodes, one dense STP-GSR dual-graph intermediate requires approximately 4.77 GiB. The refactor does not change its algorithmic complexity.
- Evaluation retains fallback behavior when centrality computation fails; interpret metrics alongside terminal warnings.

## 9. Ablation runs

The commands below use `results/ablation_v1` to keep outputs separate from the existing experiments. They share the default CSV dataset, 100 samples, 3 folds, seed 42, 60 epochs, learning rate 0.001, and accumulation count 16. Reusing the same output directory and run name overwrites that run's outputs.

The four unique HyperGSR configurations form a geometry-by-shrinkage comparison:

| Run | Distance features | ROI coordinates | Learned ROI embedding | Shrinkage |
| --- | --- | --- | --- | --- |
| `hyper_gsr_baseline` | Off | Off | On | Off |
| `hyper_gsr_geo` | On | On | Off | Off |
| `hyper_gsr_shrink001` | Off | Off | On | On, initial threshold 0.01 |
| `hyper_gsr_geo_shrink001` | On | On | Off | On, initial threshold 0.01 |

Here, geometry is a bundle of changes: connection-distance features, distance-based incidence attributes, and coordinate-derived ROI features replacing learned ROI embeddings. This comparison does not isolate those three contributions individually.

```powershell
# STP-GSR
python main.py model=stp_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=stp_gsr

# DirectSR
python main.py model=direct_sr experiment.base_dir=results/ablation_v1 experiment.run_name=direct_sr

# HyperGSR baseline: learned ROI embeddings, no geometry, no shrinkage
python main.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_baseline model.hyper_dual_learner.mode=trans model.hyper_dual_learner.edge_dim=1 model.hyper_dual_learner.use_hyper_emb=true model.hyper_dual_learner.use_geo_priors=false model.hyper_dual_learner.use_shrink_output=false

# HyperGSR geometry: distances and ROI coordinates, no shrinkage
python main.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_geo model.hyper_dual_learner.mode=trans model.hyper_dual_learner.edge_dim=1 model.hyper_dual_learner.use_hyper_emb=false model.hyper_dual_learner.use_geo_priors=true model.hyper_dual_learner.use_shrink_output=false

# HyperGSR shrinkage only
python main.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_shrink001 model.hyper_dual_learner.mode=trans model.hyper_dual_learner.edge_dim=1 model.hyper_dual_learner.use_hyper_emb=true model.hyper_dual_learner.use_geo_priors=false model.hyper_dual_learner.use_shrink_output=true model.hyper_dual_learner.shrink_threshold=0.01

# HyperGSR geometry plus shrinkage
python main.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_geo_shrink001 model.hyper_dual_learner.mode=trans model.hyper_dual_learner.edge_dim=1 model.hyper_dual_learner.use_hyper_emb=false model.hyper_dual_learner.use_geo_priors=true model.hyper_dual_learner.use_shrink_output=true model.hyper_dual_learner.shrink_threshold=0.01
```

The existing first-fold checkpoints for `hyper_gsr_geo_shrink001` and `run_hyper_emb_coord_shrink001` contain the same module structure: distance MLP, coordinate MLP, one-dimensional Transformer edge attributes, and a shrinkage parameter, with no learned ROI embedding table. Checkpoints do not establish all original training settings or the initial shrinkage threshold. Under the current implementation, `use_geo_priors=true` disables `use_hyper_emb`, even if the latter is explicitly set to true.

For compatibility with the historical experiment name, the following command selects the same effective model configuration as `hyper_gsr_geo_shrink001`; it is not a separate ablation:

```powershell
python main.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_emb_coord_shrink001 model.hyper_dual_learner.mode=trans model.hyper_dual_learner.edge_dim=1 model.hyper_dual_learner.use_hyper_emb=true model.hyper_dual_learner.use_geo_priors=true model.hyper_dual_learner.use_shrink_output=true model.hyper_dual_learner.shrink_threshold=0.01
```

Likewise, the historical `run_hyper_emb_edgeattr_shrink001` checkpoint has the non-geometric learned-ROI structure with shrinkage, matching the current `hyper_gsr_shrink001` structure. A name containing `edgeattr` does not identify an additional independent configuration switch.

Evaluate the six unique runs after training:

```powershell
python evaluate.py model=stp_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=stp_gsr
python evaluate.py model=direct_sr experiment.base_dir=results/ablation_v1 experiment.run_name=direct_sr
python evaluate.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_baseline
python evaluate.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_geo
python evaluate.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_shrink001
python evaluate.py model=hyper_gsr experiment.base_dir=results/ablation_v1 experiment.run_name=hyper_gsr_geo_shrink001
```

These evaluation commands use the default `trans` result subdirectory and only read saved predictions. They do not reconstruct models, so the geometry and shrinkage overrides are not required for metric calculation. The existing limitations in Section 8 still apply, including the default fallback's `LayerNorm(1)` behavior. No ablation training was launched when preparing these commands.

## 35-ROI morphology input

The dataset presets below use all 279 converted SLIM rows, with the averaged
35-ROI morphology graph as input and functional connectivity as the target.
Generate the CSVs with `python convert_slim_mat.py` if they do not exist.
Paired training assumes the original morphology and functional MAT files share
subject order, and that LH/RH ROI indices are homologous; conversion does not
verify these metadata assumptions.

```powershell
python main.py model=hyper_gsr dataset=morph35_func268 experiment.run_name=hyper_gsr_morph35_func268
python main.py model=hyper_gsr dataset=morph35_func160 experiment.run_name=hyper_gsr_morph35_func160
```

Each source graph has node features of shape `[35, 35]` (adjacency rows).
The model's existing input layer takes its width from `n_source_nodes`; no
network computation changes are needed. `node_feat_dim` now follows
`n_source_nodes` automatically in the CSV configuration. The CSV loader checks
that each row contains exactly `N*(N-1)/2` edge values for the configured N.
The target remains 160 or 268 ROIs; the attention-head divisibility requirement
applies to that target dimension, not to the 35 input ROIs.

The presets select target coordinates from `coords_lr.csv` (160) or
`coords_hr.csv` (268). Geometry can be enabled with
`model.hyper_dual_learner.use_geo_priors=true`; coordinate-to-target ROI order
must agree. The same dataset presets can be passed to `direct_sr` and `stp_gsr`.
Existing 160-input checkpoints have different input weight shapes and must not
be loaded unchanged for 35-input training.
