#!/usr/bin/env python3

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

from celcomen.models.simcomen import simcomen
from celcomen.training_plan.train import train_simcomen
from celcomen.utils.helpers import calc_gex, calc_sphex


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

BASE_DIR = PROJECT_DIR / "celcomen_input_rna_top_genes"
CELCOMEN_DIR = BASE_DIR / "celcomen_output_rna_top_genes"

TRAINING_H5AD = BASE_DIR / "celcomen_input_rna_top_genes.h5ad"
PARAMETER_FILE = CELCOMEN_DIR / "simcomen_interaction_parameters_rna.pt"

RESULTS_SUBDIR = "simcomen_perturbed"
REFERENCE_SUBDIR = "simcomen_reference"

MUT_RESULTS_H5AD_NAME = "simcomen_perturbed_results.h5ad"
MUT_LOSS_CSV_NAME = "simcomen_training_loss.csv"
REFERENCE_H5AD_NAME = "simcomen_reference_pre_states.h5ad"

# --- Cells / genes / populations ---------------------------------------------
SAMPLE_COLUMN = SETTINGS["cohort"]["sample_column"]
MUTANT_VALUE = SETTINGS["cohort"]["mutant_value"]

TARGET_GENE = SETTINGS["cohort"]["target_gene"]
ANNOTATION_COLUMN = SETTINGS["cohort"]["annotation_column"]
TARGET_POPULATIONS = SETTINGS["cohort"]["target_populations"]

# --- Graph / training --------------------------------------------------------
N_NEIGHBORS = SETTINGS["simcomen"]["n_neighbors"]
EPOCHS = SETTINGS["simcomen"]["epochs"]
LEARNING_RATE = SETTINGS["simcomen"]["learning_rate"]
ZMFT_SCALAR = SETTINGS["simcomen"]["zmft_scalar"]
SEED = SETTINGS["simcomen"]["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = CELCOMEN_DIR / RESULTS_SUBDIR

# =============================================================================
# LOAD DATA AND KEEP ONLY MUTANT CELLS
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(TRAINING_H5AD)

if SAMPLE_COLUMN not in adata.obs.columns:
    raise ValueError(f"{SAMPLE_COLUMN!r} was not found in adata.obs.")

sample_labels = adata.obs[SAMPLE_COLUMN].astype(str)

if MUTANT_VALUE not in set(sample_labels):
    raise ValueError(
        f"{MUTANT_VALUE!r} was not found in {SAMPLE_COLUMN!r}. "
        f"Available values: {sorted(sample_labels.unique())}"
    )

adata = adata[sample_labels == MUTANT_VALUE].copy()

parameters = torch.load(
    PARAMETER_FILE,
    map_location="cpu",
    weights_only=False,
)

genes = list(parameters["genes"])

if adata.var_names.tolist() != genes:
    raise ValueError(
        "The gene order in the H5AD does not match the CELCOMEN parameters."
    )

if TARGET_GENE not in genes:
    raise ValueError(f"{TARGET_GENE!r} was not found among the model genes.")

if ANNOTATION_COLUMN not in adata.obs.columns:
    raise ValueError(
        f"{ANNOTATION_COLUMN!r} was not found in adata.obs."
    )

gene_index = genes.index(TARGET_GENE)

target_mask = (
    adata.obs[ANNOTATION_COLUMN]
    .astype(str)
    .isin(TARGET_POPULATIONS)
    .to_numpy()
)

if target_mask.sum() == 0:
    raise ValueError(
        "No mutant cells matched TARGET_POPULATIONS. "
        f"Available values: "
        f"{sorted(adata.obs[ANNOTATION_COLUMN].astype(str).unique())}"
    )

print(f"Using only {SAMPLE_COLUMN} = {MUTANT_VALUE}")
print(f"Mutant cells: {adata.n_obs}")
print(f"Target gene: {TARGET_GENE}")
print(f"Target populations: {TARGET_POPULATIONS}")
print(f"Directly perturbed cells: {target_mask.sum()}")
print(f"Device: {DEVICE}")


# =============================================================================
# RUN SIMCOMEN ON THE MUTANT SAMPLE
# =============================================================================

n_genes = adata.n_vars

torch.manual_seed(SEED)
np.random.seed(SEED)

if sparse.issparse(adata.X):
    x_raw = adata.X.toarray()
else:
    x_raw = np.asarray(adata.X)

x_raw = torch.tensor(
    x_raw,
    dtype=torch.float32,
)

if torch.any(
    torch.linalg.vector_norm(
        x_raw,
        dim=1,
        keepdim=True,
    ) == 0
):
    raise ValueError(
        "At least one mutant cell has zero expression "
        "across all model genes."
    )


# =============================================================================
# STATE 1: BEFORE PERTURBATION, WITH NORMAL RELB
# =============================================================================

x_normal = x_raw.clone()

normal_norm_factor = torch.sqrt(
    torch.pow(
        x_normal,
        2,
    ).sum(
        dim=1,
        keepdim=True,
    )
)

x_normal = x_normal / normal_norm_factor

pre_normal_sphex = calc_sphex(
    x_normal
)

pre_normal_expression = (
    calc_gex(
        pre_normal_sphex
    )
    .detach()
    .cpu()
    .numpy()
)

print(
    "Normal-Relb round-trip maximum difference:",
    torch.abs(
        calc_gex(pre_normal_sphex)
        - x_normal
    ).max().item(),
)


# =============================================================================
# STATE 2: RELB KO BEFORE SIMCOMEN OPTIMIZATION
# =============================================================================

x_ko = x_raw.clone()

target_mask_tensor = torch.from_numpy(
    target_mask
).bool()

x_ko[
    target_mask_tensor,
    gene_index,
] = 0.0

ko_norm_factor = torch.sqrt(
    torch.pow(
        x_ko,
        2,
    ).sum(
        dim=1,
        keepdim=True,
    )
)

if torch.any(ko_norm_factor == 0):
    raise ValueError(
        "At least one mutant cell has zero expression "
        "after Relb was set to zero."
    )

x_ko = x_ko / ko_norm_factor

ko_sphex = calc_sphex(
    x_ko
).to(DEVICE)


# =============================================================================
# BUILD THE SPATIAL GRAPH
# =============================================================================

graph = kneighbors_graph(
    adata.obsm["spatial"],
    n_neighbors=N_NEIGHBORS,
    include_self=False,
)

edge_index = torch.tensor(
    np.array(
        np.where(
            graph.toarray() == 1
        )
    ),
    dtype=torch.long,
    device=DEVICE,
)


# =============================================================================
# CREATE THE SIMCOMEN MODEL
# =============================================================================

model = simcomen(
    input_dim=n_genes,
    output_dim=n_genes,
    n_neighbors=N_NEIGHBORS,
    seed=SEED,
)

model.set_g2g(
    parameters[
        "g2g_intercellular"
    ].clone()
)

model.set_g2g_intra(
    parameters[
        "g2g_intracellular"
    ].clone()
)

model.to(DEVICE)

model.set_sphex(
    ko_sphex
)


# =============================================================================
# EXTRACT THE KO STATE BEFORE OPTIMIZATION
# =============================================================================

with torch.no_grad():
    model(
        edge_index,
        1,
    )

pre_ko_expression = (
    model.gex
    .detach()
    .cpu()
    .numpy()
    .copy()
)

print(
    "Relb-KO round-trip maximum difference:",
    np.abs(
        pre_ko_expression
        - x_ko.cpu().numpy()
    ).max(),
)


# =============================================================================
# RUN SIMCOMEN OPTIMIZATION
# =============================================================================

print(
    f"Running SIMCOMEN on {adata.n_obs} mutant cells, "
    f"with {target_mask.sum()} directly perturbed cells."
)

losses = train_simcomen(
    EPOCHS,
    LEARNING_RATE,
    model,
    edge_index,
    zmft_scalar=ZMFT_SCALAR,
    seed=SEED,
    device=DEVICE,
    verbose=False,
)


# =============================================================================
# STATE 3:  KO AFTER SIMCOMEN OPTIMIZATION
# =============================================================================

post_ko_expression = (
    model.gex
    .detach()
    .cpu()
    .numpy()
    .copy()
)

print("SIMCOMEN simulation completed.")


# =============================================================================
# CREATE THE THREE CELL-BY-GENE DIFFERENCE MATRICES
# =============================================================================

delta_ko_minus_normal = (
    pre_ko_expression
    - pre_normal_expression
)

delta_post_minus_ko = (
    post_ko_expression
    - pre_ko_expression
)

delta_post_minus_normal = (
    post_ko_expression
    - pre_normal_expression
)


# =============================================================================
# STORE THE TRAINING LOSS
# =============================================================================

loss_values = []

for loss in losses:
    if torch.is_tensor(loss):
        loss_values.append(
            loss.detach().cpu().item()
        )
    else:
        loss_values.append(
            float(loss)
        )

loss_table = pd.DataFrame(
    {
        "epoch": np.arange(
            1,
            len(loss_values) + 1,
        ),
        "loss": loss_values,
    }
)


# =============================================================================
# SAVE MUTANT-ONLY RESULTS
# =============================================================================

adata.obs["simcomen_targeted"] = target_mask

adata.uns["simcomen_target_gene"] = TARGET_GENE
adata.uns["simcomen_target_populations"] = TARGET_POPULATIONS
adata.uns["simcomen_sample_filter"] = MUTANT_VALUE
adata.uns["simcomen_normalization"] = "unit_L2_per_cell"


adata.layers["simcomen_pre_normal"] = (
    pre_normal_expression
)

adata.layers["simcomen_pre_perturbed"] = (
    pre_ko_expression
)

adata.layers["simcomen_post_perturbed"] = (
    post_ko_expression
)


adata.layers["simcomen_delta_perturbed_minus_normal"] = (
    delta_ko_minus_normal
)

adata.layers["simcomen_delta_post_minus_perturbed"] = (
    delta_post_minus_ko
)

adata.layers["simcomen_delta_post_minus_normal"] = (
    delta_post_minus_normal
)


# =============================================================================
# SAVE FILES
# =============================================================================

output_h5ad = OUTPUT_DIR / MUT_RESULTS_H5AD_NAME

loss_path = OUTPUT_DIR / MUT_LOSS_CSV_NAME

adata.write_h5ad(
    output_h5ad,
    compression="gzip",
)

loss_table.to_csv(
    loss_path,
    index=False,
)

print("SIMCOMEN completed.")
print(f"Saved: {output_h5ad}")
print(f"Saved: {loss_path}")


###########################################################


# =============================================================================
# EXTRA: PREPARE WT + MUTANT CELLS FOR A COMMON PCA/UMAP REFERENCE
# =============================================================================

ALL_CELLS_OUTPUT_DIR = CELCOMEN_DIR / REFERENCE_SUBDIR

ALL_CELLS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LOAD ALL CELLS: WT + MUTANT
# =============================================================================

adata_all = ad.read_h5ad(
    TRAINING_H5AD
)

if adata_all.var_names.tolist() != genes:
    raise ValueError(
        "The gene order in the all-cell H5AD does not match "
        "the CELCOMEN parameters."
    )

if SAMPLE_COLUMN not in adata_all.obs.columns:
    raise ValueError(
        f"{SAMPLE_COLUMN!r} was not found in adata_all.obs."
    )

if ANNOTATION_COLUMN not in adata_all.obs.columns:
    raise ValueError(
        f"{ANNOTATION_COLUMN!r} was not found in adata_all.obs."
    )

all_target_mask = (
    adata_all.obs[SAMPLE_COLUMN]
    .astype(str)
    .eq(MUTANT_VALUE)
    &
    adata_all.obs[ANNOTATION_COLUMN]
    .astype(str)
    .isin(TARGET_POPULATIONS)
).to_numpy()

if all_target_mask.sum() == 0:
    raise ValueError(
        "No mutant cells matched TARGET_POPULATIONS."
    )

print("Preparing WT + mutant cells for the common PCA/UMAP reference.")
print(f"All cells: {adata_all.n_obs}")
print(f"Directly perturbed mutant cells: {all_target_mask.sum()}")


# =============================================================================
# CONVERT THE COMPLETE EXPRESSION MATRIX TO A DENSE TENSOR
# =============================================================================

if sparse.issparse(adata_all.X):
    x_all_raw = adata_all.X.toarray()
else:
    x_all_raw = np.asarray(
        adata_all.X
    )

x_all_raw = torch.tensor(
    x_all_raw,
    dtype=torch.float32,
)

if torch.any(
    torch.linalg.vector_norm(
        x_all_raw,
        dim=1,
        keepdim=True,
    ) == 0
):
    raise ValueError(
        "At least one WT or mutant cell has zero expression "
        "across all model genes."
    )


# =============================================================================
# STATE 1: ALL CELLS WITH NORMAL RELB
# =============================================================================

x_all_normal = x_all_raw.clone()

all_normal_norm_factor = torch.sqrt(
    torch.pow(
        x_all_normal,
        2,
    ).sum(
        dim=1,
        keepdim=True,
    )
)

x_all_normal = (
    x_all_normal
    / all_normal_norm_factor
)

all_normal_sphex = calc_sphex(
    x_all_normal
)

all_pre_normal_expression = (
    calc_gex(
        all_normal_sphex
    )
    .detach()
    .cpu()
    .numpy()
)

print(
    "All-cell normal-Relb round-trip maximum difference:",
    torch.abs(
        calc_gex(all_normal_sphex)
        - x_all_normal
    ).max().item(),
)


# =============================================================================
# STATE 2: RELB KO ONLY IN THE SELECTED MUTANT POPULATIONS
# =============================================================================

x_all_ko = x_all_raw.clone()

all_target_mask_tensor = torch.from_numpy(
    all_target_mask
).bool()

x_all_ko[
    all_target_mask_tensor,
    gene_index,
] = 0.0

all_ko_norm_factor = torch.sqrt(
    torch.pow(
        x_all_ko,
        2,
    ).sum(
        dim=1,
        keepdim=True,
    )
)

if torch.any(all_ko_norm_factor == 0):
    raise ValueError(
        "At least one WT or mutant cell has zero expression "
        "after Relb was set to zero."
    )

x_all_ko = (
    x_all_ko
    / all_ko_norm_factor
)

all_ko_sphex = calc_sphex(
    x_all_ko
)

all_pre_ko_expression = (
    calc_gex(
        all_ko_sphex
    )
    .detach()
    .cpu()
    .numpy()
)

print(
    "All-cell Relb-KO round-trip maximum difference:",
    torch.abs(
        calc_gex(all_ko_sphex)
        - x_all_ko
    ).max().item(),
)


# =============================================================================
# CHECK THAT NON-TARGETED CELLS DID NOT CHANGE
# =============================================================================

non_target_max_difference = np.abs(
    all_pre_ko_expression[~all_target_mask]
    - all_pre_normal_expression[~all_target_mask]
).max()

print(
    "Maximum difference in non-targeted cells:",
    non_target_max_difference,
)


# =============================================================================
# SAVE THE TWO ALL-CELL PRE-SIMCOMEN STATES
# =============================================================================

adata_all.obs["simcomen_targeted"] = (
    all_target_mask
)

adata_all.uns["simcomen_target_gene"] = (
    TARGET_GENE
)
adata_all.uns["simcomen_target_populations"] = (
    TARGET_POPULATIONS
)
adata_all.uns["simcomen_directly_perturbed_sample"] = (
    MUTANT_VALUE
)
adata_all.uns["simcomen_normalization"] = (
    "unit_L2_per_cell"
)

adata_all.layers["simcomen_pre_normal"] = (
    all_pre_normal_expression
)

adata_all.layers["simcomen_pre_perturbed"] = (
    all_pre_ko_expression
)

all_cells_output_h5ad = ALL_CELLS_OUTPUT_DIR / REFERENCE_H5AD_NAME

adata_all.write_h5ad(
    all_cells_output_h5ad,
    compression="gzip",
)

print("All-cell reference preparation completed.")
print(f"Saved: {all_cells_output_h5ad}")
