# Lars – Continuous-Time Transformer Playground

This project contains an end-to-end experimentation harness for sequence interpolation using the continuous-time transformer (“ContiFormer”) introduced by Microsoft Research. It focuses on learning to recover missing regions in synthetic 1-D time-series that mix smooth, periodic, exponential, and discontinuous behaviours. Everything that is required to generate training data, build the neural architecture, and run experiments lives inside `projects/Lars/`.

## Repository Map

| Path | Purpose |
| --- | --- |
| `training.py` | CLI entry point that wires the dataset, ContiFormer model, optimizer, logging, checkpointing, and validation loop. |
| `dataset_timeSeries.py` | PyTorch `Dataset` that synthesizes random time-series, removes chunks to mimic irregular sampling, and min–max normalizes the data. |
| `create_data.py` | Library of generators (slope-based, spline, periodic, exponential, discontinuous, etc.) that `dataset_timeSeries.py` calls through `callFunction`. |
| `contiformer.py` | Transformer encoder built on top of continuous-time attention layers with Neural-ODE-based projections. |
| `linear.py`, `ode.py`, `interpolate.py`, `positional_encoding.py` | Building blocks used by the ContiFormer blocks (ODE solvers, interpolation utilities, positional encodings). |
| `useful.py` | Helper utilities for rounding and masking parts of a time-series. |

## Synthetic Data Generation

1. `TimeSeriesDataset_Interpolation_roundedInput` (in `dataset_timeSeries.py`) receives hyperparameters via `argparse`, including value bounds (`--y_lim_low`, `--y_lim_high`), the number of samples, interpolation window widths, and noise settings.
2. For each requested sequence, `create_data.callFunction` randomly chooses one of eight generator families:
   - **Piece-wise slopes** with random lengths and gradients.
   - **Random walk + spline smoothing** at different spline tensions.
   - **Exponential curves**.
   - **Single-frequency periodic signals**.
   - **High-order spline trajectories** (tighter smoothing).
   - **Periodic sums** (superposition of multiple sine waves).
   - **Discontinuous signals** generated either through injected steps or with stochastic spline offsets.
3. Gaussian or uniform noise with configurable std (`--noise_std_*`) is added. Each series is min–max scaled using the tracked extrema (`div_term` and `min_value`) before being returned.
4. `useful.remove_parts_of_graph_encoder` creates masks that drop up to `--interpolation_max_count` contiguous windows of widths sampled between `--interpolation_min_width` and `--interpolation_max_width`, constrained by `--offset` and the global x-range (`--x_lim_low`, `--x_lim_high`). These windows represent the missing measurements the model must impute.
5. The dataset returns dictionaries containing the normalized noisy observation, the clean target, a boolean mask denoting observed points, original timestamps, and scaling metadata so predictions can be rescaled after inference.

## Model Architecture

`contiformer.ContiFormer` is a Transformer encoder adapted to irregularly-sampled data:

- Each `EncoderLayer` contains a `MultiHeadAttention` block where the query, key, and value projections are implemented with `ODELinear`/`InterpLinear`. These modules integrate Neural-ODE dynamics (`ode.py`) or interpolation operators (`interpolate.py`) between pairs of timestamps, enabling attention to reason directly in continuous time.
- Temporal context is injected both through explicit timestamp encodings (`temporal_enc`) and classic sinusoidal positional encodings (`PositionalEncoding`).
- `projects/Lars/training.py` wraps the encoder with simple input/output linear layers that map the scalar observation channel to a width of 16, and back to the original dimension.
- During training the model is called through the helper `ContiFormer` class (note: this name collides with the encoder file, but the wrapper in `training.py` takes care of batching, interpolation via `torchcde`, and sampling strategies).

## Training Workflow (`training.py`)

1. Parse CLI arguments (see tables below), create output directories (`--train_dir`, `--val_dir_pictures`, `--val_dir_data`), and seed all RNGs for repeatability.
2. Instantiate the dataset via `get_ds_timeSeries`. Training is batched (`--batch_size`), while validation iterates one sequence at a time to simplify visualization.
3. For every batch:
   - Use the boolean mask to select observed timestamps and values, append the absolute time as a second feature, and send the result to the model.
   - `torchcde.LinearInterpolation` turns the irregular samples into a continuous path so the ContiFormer can evaluate it at the original high-resolution timestamps.
   - The model predicts the full trajectory, scaled back to the original units via (`pred * div_term + min_value`), and the L2 reconstruction loss is computed only on the training slices that were requested for this forward pass.
   - Adam optimizer (`--lr`) updates the model; `RunningAverageMeter` keeps a smoothed loss estimate.
4. After every iteration the script:
   - Logs progress to `train_dir/log.log`.
   - Saves checkpoints (`ckpt_Contiformer.pth`) containing both the model and optimizer state.
   - Runs one validation batch, reporting MAE/RMSE, saving an SVG overlay of noisy, ground-truth, and predicted curves, and dumping tensors to `val_dir_data/pred_{itr}.pkl` for further analysis.

## Running Experiments

### Prerequisites

- Python ≥ 3.9
- PyTorch (GPU build recommended but CPU works), torchcde, torchdiffeq, numpy, pandas, scipy, matplotlib, tqdm.
- Install the project-level dependencies once via `pip install -r requirements.txt` from the repository root.

### Basic command

```bash
cd projects/Lars
python training.py \
    --gpu 0 \
    --niters 2000 \
    --train_count 200 \
    --val_count 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --train_dir ./experiments/train_run \
    --val_dir_pictures ./experiments/val_figs \
    --val_dir_data ./experiments/val_tensors
```

All directories are created automatically. Use `--visualize True` if you want matplotlib windows (note that the script saves figures regardless).

### Key CLI arguments

**General training**

| Argument | Description | Default |
| --- | --- | --- |
| `--niters` | Number of outer epochs (validation/checkpointing frequency). | `4000` |
| `--lr` | Adam learning rate. | `0.01` |
| `--batch_size` | Training batch size (also used for sub-sampling inside the model). | `10` |
| `--gpu` | CUDA device index (set to `-1` to force CPU). | `0` |
| `--adjoint` | Whether to use the adjoint method in torchdiffeq. | `False` |
| `--model_name` | For checkpoint naming; ContiFormer is the only implemented model. | `"Contiformer"` |
| `--seed` | Reproducibility seed applied to NumPy, Python, and PyTorch RNGs. | `27` |

**ContiFormer / ODE**

| Argument | Description | Default |
| --- | --- | --- |
| `--atol`, `--rtol` | Absolute/relative tolerances for Neural-ODE solvers. | `0.1` |
| `--method` | torchdiffeq solver (`rk4`, `dopri5`, …). | `"rk4"` |
| `--dropout` | Dropout applied in attention/FFN blocks. | `0.1` |

**Synthetic data controls**

| Argument | Description | Default |
| --- | --- | --- |
| `--train_count`, `--val_count` | Number of synthetic time-series drawn per loader epoch. | `10` / `1` |
| `--number_x_values` | Length (number of timestamps) of each sequence. | `1000` |
| `--y_lim_low`, `--y_lim_high` | Range for starting values before generation. | `10` / `10000` |
| `--random_number_range_*` | Distribution used for increments inside random-walk-like generators. | `"norm"`, `0`, `5` |
| `--spline_value_low/high` | Range for spline smoothing strength. | `8e5` / `1.1e6` |
| `--noise_std_*` | Distribution parameters for observation noise scaling. | `"norm"`, `0`, `0.15` |
| `--interpolation_min/max_width`, `--interpolation_max_count` | Controls how many intervals are masked from each sequence. | `10` / `100` / `10` |
| `--offset`, `--x_lim_low/high` | Guard bands for where masked intervals may start/end. | `10`, `0`, `1000` |

These switches make it easy to stress-test the model under various kinds of sparsity, amplitude dynamics, and noise levels. Because the dataset is generated on the fly, increasing `train_count` effectively enlarges each epoch.

## Outputs and Evaluation

- **Checkpoints**: `train_dir/ckpt_Contiformer.pth` contains `state_dict`s for both the model and optimizer plus the latest iteration counter.
- **Logs**: `train_dir/log.log` captures per-iteration loss, MAE/RMSE values, and artifact paths.
- **Visualizations**: SVG plots under `val_dir_pictures/` overlay noisy input points, the hidden ground truth, and the reconstructed curve.
- **Pickle files**: `val_dir_data/pred_*.pkl` store tensors (`pred`, `target`, `samp`) for custom metrics or downstream evaluation.

To recover predictions in original units:

```python
denorm_pred = pred * batch["div_term"].unsqueeze(-1) + batch["min_value"].unsqueeze(-1)
```

## Extending the Project

- **New generators**: Add a function to `create_data.py` and register it inside `callFunction`. The dataset will automatically draw from it.
- **Different observation patterns**: Modify `remove_parts_of_graph_encoder` to change how gaps are carved out, or replace the masking strategy entirely.
- **Alternate models**: `training.py` already reserves `--model_name`. You can plug a new architecture in by following the same interface (`forward` returns predictions and, optionally, sampled indices).

With these components you can rapidly iterate on continuous-time transformer ideas, evaluate robustness to missing data, or port the synthetic pipeline to your own datasets.
