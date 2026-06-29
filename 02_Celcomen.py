#!/usr/bin/env python3
# =============================================================================
import argparse
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse as sp

from celcomen.datareaders.datareader import get_dataset_loaders
from celcomen.models.celcomen import celcomen
from celcomen.training_plan.train import train
from celcomen.utils.helpers import normalize_g2g

# =============================================================================
# CONFIGURATION
# =============================================================================

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    "--settings",
    type=Path,
    default=Path("settings.yaml"),
    help="Path to the shared pipeline settings.yaml file.",
)
arguments = argument_parser.parse_args()

SETTINGS_PATH = arguments.settings.expanduser().resolve()

if not SETTINGS_PATH.is_file():
    raise FileNotFoundError(f"Settings file not found: {SETTINGS_PATH}")

with SETTINGS_PATH.open() as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_DIR = Path(SETTINGS["project"]["project_dir"]).expanduser()

INPUT_DIR = PROJECT_DIR / "celcomen_input_rna_top_genes"
OUTPUT_DIR = INPUT_DIR / "celcomen_output_rna_top_genes"
INPUT_H5AD = INPUT_DIR / "celcomen_input_rna_top_genes.h5ad"

MODEL_FILE = "celcomen_model_rna_top_genes.pt"
PARAMETER_FILE = "simcomen_interaction_parameters_rna.pt"
LOSS_FILE = "training_loss.csv"

# --- Data columns ------------------------------------------------------------
SAMPLE_COLUMN = SETTINGS["cohort"]["sample_column"]

# --- Graph / training --------------------------------------------------------
N_NEIGHBORS = SETTINGS["celcomen"]["n_neighbors"]
DISTANCE_THRESHOLD = SETTINGS["celcomen"]["distance_threshold"]

EPOCHS = SETTINGS["celcomen"]["epochs"]
LEARNING_RATE = SETTINGS["celcomen"]["learning_rate"]
ZMFT_SCALAR = SETTINGS["celcomen"]["zmft_scalar"]
SEED = SETTINGS["celcomen"]["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# LOAD AND VALIDATE
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(INPUT_H5AD)

if SAMPLE_COLUMN not in adata.obs.columns:
    raise ValueError(f"{SAMPLE_COLUMN!r} was not found in adata.obs.")

if "spatial" not in adata.obsm:
    raise ValueError("adata.obsm['spatial'] is missing from the H5AD.")

adata.obs[SAMPLE_COLUMN] = adata.obs[SAMPLE_COLUMN].astype(str)

print(f"AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"Samples: {adata.obs[SAMPLE_COLUMN].nunique()}")
print(f"Selected device: {DEVICE}")

# =============================================================================
# PER-CELL L2 NORMALIZATION ACROSS THE MODEL GENES
# =============================================================================

X = adata.X

if sp.issparse(X):
    cell_l2 = np.sqrt(np.asarray(X.power(2).sum(axis=1)).ravel())
else:
    X = np.asarray(X, dtype=np.float64)
    cell_l2 = np.sqrt(np.square(X).sum(axis=1)).ravel()

if np.any(~np.isfinite(cell_l2)):
    raise ValueError("At least one cell has a non-finite L2 norm.")

if np.any(cell_l2 == 0):
    n_zero = int(np.sum(cell_l2 == 0))
    raise ValueError(
        f"{n_zero} cells have zero expression across all selected model genes "
        "and therefore cannot be L2-normalized."
    )

if sp.issparse(X):
    X = X.multiply((1.0 / cell_l2)[:, None]).tocsr().astype(np.float32)
else:
    X = (X / cell_l2[:, None]).astype(np.float32)

adata.X = X

if sp.issparse(adata.X):
    cell_l2_check = np.sqrt(np.asarray(adata.X.power(2).sum(axis=1)).ravel())
else:
    cell_l2_check = np.sqrt(np.square(np.asarray(adata.X)).sum(axis=1)).ravel()

if not np.allclose(cell_l2_check, 1.0, atol=1e-5, rtol=1e-5):
    raise RuntimeError(
        "Per-cell L2 normalization failed. Observed range: "
        f"{cell_l2_check.min():.8f} to {cell_l2_check.max():.8f}."
    )

print(
    "Per-cell L2 norms after normalization: "
    f"min={cell_l2_check.min():.6f}, max={cell_l2_check.max():.6f}"
)

n_genes = adata.n_vars

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# INITIALIZE AND TRAIN CELCOMEN
# =============================================================================

model = celcomen(
    input_dim=n_genes,
    output_dim=n_genes,
    n_neighbors=N_NEIGHBORS,
    seed=SEED,
)

initial_g2g = np.random.uniform(
    size=(n_genes, n_genes)
).astype(np.float32)

initial_g2g = normalize_g2g(
    (initial_g2g + initial_g2g.T) / 2
)

model.set_g2g(
    torch.from_numpy(initial_g2g)
)

model.set_g2g_intra(
    torch.from_numpy(initial_g2g.copy())
)

model.to(DEVICE)

tmp_dir = Path(tempfile.mkdtemp(prefix="celcomen_norm_", dir=OUTPUT_DIR))
normalized_h5ad = tmp_dir / "celcomen_training_normalized.h5ad"
adata.write_h5ad(normalized_h5ad, compression="gzip")

try:
    dataloader = get_dataset_loaders(
        str(normalized_h5ad),
        sample_id_name=SAMPLE_COLUMN,
        n_neighbors=N_NEIGHBORS,
        distance=DISTANCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    print("Starting CELCOMEN training.")

    losses = train(
        EPOCHS,
        LEARNING_RATE,
        model,
        dataloader,
        zmft_scalar=ZMFT_SCALAR,
        seed=SEED,
        device=DEVICE,
        verbose=False,
    )

    print("CELCOMEN training completed.")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

# =============================================================================
# SAVE RESULTS
# =============================================================================

model_state = {
    name: tensor.detach().cpu()
    for name, tensor in model.state_dict().items()
    if name != "gex"
}

model_path = OUTPUT_DIR / MODEL_FILE
parameter_path = OUTPUT_DIR / PARAMETER_FILE
loss_path = OUTPUT_DIR / LOSS_FILE

torch.save(
    {
        "model_state_dict": model_state,
        "input_dim": n_genes,
        "output_dim": n_genes,
        "n_neighbors": N_NEIGHBORS,
        "seed": SEED,
        "genes": adata.var_names.tolist(),
    },
    model_path,
)

torch.save(
    {
        "genes": adata.var_names.tolist(),
        "g2g_intercellular": model.conv1.lin.weight.detach().cpu(),
        "g2g_intracellular": model.lin.weight.detach().cpu(),
    },
    parameter_path,
)

pd.DataFrame(
    {
        "epoch": np.arange(1, len(losses) + 1),
        "loss": np.asarray(losses, dtype=float),
    }
).to_csv(
    loss_path,
    index=False,
)

print("Saved outputs:")
print(f"  {model_path}")
print(f"  {parameter_path}")
print(f"  {loss_path}")
