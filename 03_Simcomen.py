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


# =============================================================================
# STABLE HYPERSPHERICAL TRANSFORMATIONS
# =============================================================================


def calc_gex(
    sphex: torch.Tensor,
) -> torch.Tensor:
    """
    Convert hyperspherical coordinates to Cartesian gene-expression
    coordinates.
    """

    if sphex.ndim != 2:
        raise ValueError(
            "sphex must be a two-dimensional cells-by-angles tensor."
        )

    if sphex.shape[1] < 1:
        raise ValueError(
            "sphex must contain at least one hyperspherical angle."
        )

    if not torch.isfinite(sphex).all().item():
        nan_count = torch.isnan(sphex).sum().item()
        inf_count = torch.isinf(sphex).sum().item()

        raise ValueError(
            "sphex contains non-finite values. "
            f"NaNs: {nan_count}; infinities: {inf_count}."
        )

    sin_angles = torch.sin(
        sphex
    )

    cos_angles = torch.cos(
        sphex
    )

    # prefix_sines[:, i] =
    # sin(theta_0) * sin(theta_1) * ... * sin(theta_i)
    prefix_sines = torch.cumprod(
        sin_angles,
        dim=1,
    )

    # x_0 = cos(theta_0)
    first_coordinate = cos_angles[
        :,
        :1,
    ]

    if sphex.shape[1] == 1:
        # Two-gene special case:
        # x_0 = cos(theta_0)
        # x_1 = sin(theta_0)
        gex = torch.cat(
            [
                first_coordinate,
                prefix_sines[:, -1:],
            ],
            dim=1,
        )

    else:
        # x_i = product(previous sines) * cos(theta_i)
        intermediate_coordinates = (
            prefix_sines[:, :-1]
            *
            cos_angles[:, 1:]
        )

        # x_last = product(all sines)
        final_coordinate = prefix_sines[
            :,
            -1:,
        ]

        gex = torch.cat(
            [
                first_coordinate,
                intermediate_coordinates,
                final_coordinate,
            ],
            dim=1,
        )

    if not torch.isfinite(gex).all().item():
        nan_count = torch.isnan(gex).sum().item()
        inf_count = torch.isinf(gex).sum().item()

        raise RuntimeError(
            "calc_gex produced non-finite expression values. "
            f"NaNs: {nan_count}; infinities: {inf_count}."
        )

    return gex



def calc_sphex(
    gex: torch.Tensor,
    *,
    normalize: bool = False,
    unit_norm_tolerance: float = 1e-5,
) -> torch.Tensor:
    """
    Convert Cartesian gene-expression coordinates to hyperspherical
    coordinates using tail norms and atan2.
    """

    if gex.ndim != 2:
        raise ValueError(
            "gex must be a two-dimensional cells-by-genes tensor."
        )

    if gex.shape[1] < 2:
        raise ValueError(
            "At least two genes are required."
        )

    if not torch.isfinite(gex).all().item():
        nan_count = torch.isnan(gex).sum().item()
        inf_count = torch.isinf(gex).sum().item()

        raise ValueError(
            "gex contains non-finite values. "
            f"NaNs: {nan_count}; infinities: {inf_count}."
        )

    output_dtype = (
        gex.dtype
        if gex.is_floating_point()
        else torch.float32
    )

    x = gex.to(
        dtype=torch.float64
    )

    norms = torch.linalg.vector_norm(
        x,
        dim=1,
        keepdim=True,
    )

    zero_norm_mask = norms.squeeze(1) == 0

    if zero_norm_mask.any().item():
        raise ValueError(
            "At least one cell has zero L2 norm. "
            f"Zero-norm cells: {zero_norm_mask.sum().item()}."
        )

    if normalize:
        x = x / norms

    else:
        maximum_norm_error = torch.max(
            torch.abs(
                norms - 1.0
            )
        ).item()

        if maximum_norm_error > unit_norm_tolerance:
            raise ValueError(
                "Expression vectors are not unit-L2 normalized. "
                f"Maximum norm deviation: {maximum_norm_error:.10g}. "
                "Normalize them first or call calc_sphex(..., normalize=True)."
            )

    n_genes = x.shape[1]

    if n_genes == 2:
        angles = torch.atan2(
            x[:, 1],
            x[:, 0],
        ).unsqueeze(
            dim=1
        )

    else:

        tail_squared = torch.flip(
            torch.cumsum(
                torch.flip(
                    x.square(),
                    dims=[1],
                ),
                dim=1,
            ),
            dims=[1],
        )

        polar_angles = torch.atan2(
            torch.sqrt(
                torch.clamp(
                    tail_squared[:, 1:-1],
                    min=0.0,
                )
            ),
            x[:, :-2],
        )

        final_angle = torch.atan2(
            x[:, -1],
            x[:, -2],
        ).unsqueeze(
            dim=1
        )

        angles = torch.cat(
            [
                polar_angles,
                final_angle,
            ],
            dim=1,
        )

    if not torch.isfinite(angles).all().item():
        nan_count = torch.isnan(angles).sum().item()
        inf_count = torch.isinf(angles).sum().item()

        raise RuntimeError(
            "calc_sphex produced non-finite angles. "
            f"NaNs: {nan_count}; infinities: {inf_count}."
        )

    return angles.to(
        dtype=output_dtype,
        device=gex.device,
    )


# =============================================================================
# PATCHING THE ORIGINAL SIMCOMEN CLASS
# =============================================================================

def _simcomen_calc_gex(
    self,
    sphex: torch.Tensor,
) -> torch.Tensor:
    return calc_gex(
        sphex
    )



def _simcomen_calc_sphex(
    self,
    gex: torch.Tensor,
) -> torch.Tensor:
    return calc_sphex(
        gex
    )


simcomen.calc_gex = _simcomen_calc_gex
simcomen.calc_sphex = _simcomen_calc_sphex


# =============================================================================
# ROUND-TRIP VALIDATION
# =============================================================================

ROUNDTRIP_MAXIMUM_ALLOWED_ERROR = 1e-5
roundtrip_diagnostics = []



def validate_roundtrip(
    expression: torch.Tensor,
    *,
    label: str,
    maximum_allowed_error: float = ROUNDTRIP_MAXIMUM_ALLOWED_ERROR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converting expression to spherical coordinates and back, printing comprehensive
    diagnostics, and aborting if the reconstruction is not accurate.
    """

    if not torch.isfinite(expression).all().item():
        raise ValueError(
            f"{label}: the starting expression contains NaN or infinity."
        )

    input_norms = torch.linalg.vector_norm(
        expression,
        dim=1,
    )

    maximum_input_norm_deviation = torch.max(
        torch.abs(
            input_norms - 1.0
        )
    ).item()

    sphex = calc_sphex(
        expression,
        normalize=False,
    )

    reconstructed = calc_gex(
        sphex
    )

    difference = torch.abs(
        reconstructed
        - expression
    )

    nan_angles = torch.isnan(sphex).sum().item()
    infinite_angles = torch.isinf(sphex).sum().item()
    nan_reconstructed = torch.isnan(reconstructed).sum().item()
    infinite_reconstructed = torch.isinf(reconstructed).sum().item()

    maximum_error = difference.max().item()
    mean_error = difference.mean().item()
    median_error = difference.median().item()

    fraction_above_1e6 = (
        difference > 1e-6
    ).float().mean().item()

    fraction_above_threshold = (
        difference > maximum_allowed_error
    ).float().mean().item()

    cells_above_threshold = (
        difference > maximum_allowed_error
    ).any(
        dim=1
    ).sum().item()

    reconstructed_norms = torch.linalg.vector_norm(
        reconstructed,
        dim=1,
    )

    maximum_reconstructed_norm_deviation = torch.max(
        torch.abs(
            reconstructed_norms - 1.0
        )
    ).item()

    diagnostics = {
        "label": label,
        "n_cells": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "nan_angles": int(nan_angles),
        "infinite_angles": int(infinite_angles),
        "nan_reconstructed_values": int(nan_reconstructed),
        "infinite_reconstructed_values": int(infinite_reconstructed),
        "maximum_absolute_error": float(maximum_error),
        "mean_absolute_error": float(mean_error),
        "median_absolute_error": float(median_error),
        "fraction_above_1e-6": float(fraction_above_1e6),
        "fraction_above_allowed_error": float(fraction_above_threshold),
        "cells_above_allowed_error": int(cells_above_threshold),
        "maximum_input_norm_deviation": float(
            maximum_input_norm_deviation
        ),
        "maximum_reconstructed_norm_deviation": float(
            maximum_reconstructed_norm_deviation
        ),
    }

    roundtrip_diagnostics.append(
        diagnostics
    )

    print("\n" + "=" * 79)
    print(f"ROUND-TRIP VALIDATION: {label}")
    print("=" * 79)
    print(f"Cells: {expression.shape[0]}")
    print(f"Genes: {expression.shape[1]}")
    print(f"NaN spherical angles: {nan_angles}")
    print(f"Infinite spherical angles: {infinite_angles}")
    print(f"NaN reconstructed values: {nan_reconstructed}")
    print(f"Infinite reconstructed values: {infinite_reconstructed}")
    print(f"Maximum absolute reconstruction error: {maximum_error:.12g}")
    print(f"Mean absolute reconstruction error: {mean_error:.12g}")
    print(f"Median absolute reconstruction error: {median_error:.12g}")
    print(f"Fraction of entries above 1e-6: {fraction_above_1e6:.12g}")
    print(
        f"Fraction of entries above {maximum_allowed_error}: "
        f"{fraction_above_threshold:.12g}"
    )
    print(
        f"Cells with at least one error above {maximum_allowed_error}: "
        f"{cells_above_threshold}"
    )
    print(
        "Maximum input L2-norm deviation from 1: "
        f"{maximum_input_norm_deviation:.12g}"
    )
    print(
        "Maximum reconstructed L2-norm deviation from 1: "
        f"{maximum_reconstructed_norm_deviation:.12g}"
    )

    if nan_angles != 0 or infinite_angles != 0:
        raise RuntimeError(
            f"{label}: the spherical coordinates contain non-finite values."
        )

    if nan_reconstructed != 0 or infinite_reconstructed != 0:
        raise RuntimeError(
            f"{label}: reconstructed expression contains non-finite values."
        )

    if maximum_error > maximum_allowed_error:
        raise RuntimeError(
            f"{label}: maximum round-trip error {maximum_error:.12g} "
            f"exceeds the allowed threshold {maximum_allowed_error}."
        )

    print("ROUND-TRIP VALIDATION PASSED.")
    print("=" * 79 + "\n")

    return sphex, reconstructed


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
CELCOMEN_DIR = INPUT_DIR / "celcomen_output_rna_top_genes"

INPUT_H5AD = INPUT_DIR / "celcomen_input_rna_top_genes.h5ad"
PARAMETER_FILE = CELCOMEN_DIR / "simcomen_interaction_parameters_rna.pt"

SAMPLE_COLUMN = SETTINGS["cohort"]["sample_column"]
ANNOTATION_COLUMN = SETTINGS["cohort"]["annotation_column"]
TARGET_GENE = SETTINGS["cohort"]["target_gene"]

_KO = SETTINGS["simcomen_ko"]
MUTANT_VALUE = _KO["group"]
TARGET_POPULATIONS = list(_KO["target_populations"])

N_NEIGHBORS = _KO["n_neighbors"]
EPOCHS = _KO["epochs"]
LEARNING_RATE = _KO["learning_rate"]
ZMFT_SCALAR = _KO["zmft_scalar"]
SEED = _KO["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = CELCOMEN_DIR / "simcomen_perturbed"


# =============================================================================
# LOAD DATA AND KEEP ONLY MUTANT CELLS
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(INPUT_H5AD)

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
).item():
    raise ValueError(
        "At least one mutant cell has zero expression "
        "across all model genes."
    )


# =============================================================================
# STATE 1: BEFORE PERTURBATION, WITH NORMAL RELB
# =============================================================================

x_normal = x_raw.clone()

normal_norm_factor = torch.linalg.vector_norm(
    x_normal,
    dim=1,
    keepdim=True,
)

x_normal = x_normal / normal_norm_factor

pre_normal_sphex, pre_normal_tensor = validate_roundtrip(
    x_normal,
    label="Mutant normal-Relb starting state",
)

pre_normal_expression = (
    pre_normal_tensor
    .detach()
    .cpu()
    .numpy()
    .copy()
)


# =============================================================================
# STATE 2: RELB KO BEFORE SIMCOMEN OPTIMIZATION
# =============================================================================

x_ko = x_raw.clone()

target_mask_tensor_cpu = torch.from_numpy(
    target_mask
).bool()

x_ko[
    target_mask_tensor_cpu,
    gene_index,
] = 0.0

ko_norm_factor = torch.linalg.vector_norm(
    x_ko,
    dim=1,
    keepdim=True,
)

if torch.any(ko_norm_factor == 0).item():
    raise ValueError(
        "At least one mutant cell has zero expression "
        "after Relb was set to zero."
    )

x_ko = x_ko / ko_norm_factor

ko_sphex_cpu, pre_ko_tensor = validate_roundtrip(
    x_ko,
    label="Mutant Relb-KO starting state",
)

pre_ko_expression_direct = (
    pre_ko_tensor
    .detach()
    .cpu()
    .numpy()
    .copy()
)

pre_ko_targeted_relb_maximum = np.max(
    np.abs(
        pre_ko_expression_direct[
            target_mask,
            gene_index,
        ]
    )
)

print(
    "Maximum absolute targeted Relb after the corrected KO round trip:",
    pre_ko_targeted_relb_maximum,
)

ko_sphex = ko_sphex_cpu.to(
    DEVICE
)

target_mask_tensor = target_mask_tensor_cpu.to(
    DEVICE
)


# =============================================================================
# BUILD THE SPATIAL GRAPH
# =============================================================================

if "spatial" not in adata.obsm:
    raise ValueError(
        'adata.obsm["spatial"] was not found.'
    )

graph = kneighbors_graph(
    adata.obsm["spatial"],
    n_neighbors=N_NEIGHBORS,
    include_self=False,
)

graph_row, graph_col = graph.nonzero()

edge_index = torch.tensor(
    np.vstack(
        [
            graph_row,
            graph_col,
        ]
    ),
    dtype=torch.long,
    device=DEVICE,
)


# =============================================================================
# CREATE THE ORIGINAL SIMCOMEN MODEL WITH PATCHED METHODS
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
    ].detach().clone()
)

model.set_g2g_intra(
    parameters[
        "g2g_intracellular"
    ].detach().clone()
)

model.to(
    DEVICE
)

model.set_sphex(
    ko_sphex.detach().clone()
)


# =============================================================================
# EXTRACT AND VERIFY THE RELB-KO STATE BEFORE OPTIMIZATION
# =============================================================================

model.eval()

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

model_initialization_maximum_difference = np.abs(
    pre_ko_expression
    - pre_ko_expression_direct
).max()

print(
    "Maximum difference between direct corrected calc_gex output and "
    "the patched model forward output:",
    model_initialization_maximum_difference,
)

if model_initialization_maximum_difference > ROUNDTRIP_MAXIMUM_ALLOWED_ERROR:
    raise RuntimeError(
        "The patched model did not reproduce the validated Relb-KO "
        "starting expression."
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
# STATE 3: RELB KO AFTER SIMCOMEN OPTIMIZATION
# =============================================================================

model.eval()

with torch.no_grad():
    model(
        edge_index,
        1,
    )

if not torch.isfinite(model.sphex).all().item():
    raise RuntimeError(
        "Final SIMCOMEN spherical coordinates contain NaN or infinity."
    )

if not torch.isfinite(model.gex).all().item():
    raise RuntimeError(
        "Final SIMCOMEN expression contains NaN or infinity."
    )

post_ko_expression = (
    model.gex
    .detach()
    .cpu()
    .numpy()
    .copy()
)

post_norms = np.linalg.norm(
    post_ko_expression,
    axis=1,
)

post_targeted_relb = post_ko_expression[
    target_mask,
    gene_index,
]

print("SIMCOMEN simulation completed.")
print(
    "Final expression L2-norm range:",
    float(post_norms.min()),
    "to",
    float(post_norms.max()),
)
print(
    "Final expression value range:",
    float(post_ko_expression.min()),
    "to",
    float(post_ko_expression.max()),
)
print(
    "Maximum absolute targeted Relb after optimization:",
    float(np.max(np.abs(post_targeted_relb))),
)
print(
    "Median targeted Relb after optimization:",
    float(np.median(post_targeted_relb)),
)


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

if not np.isfinite(loss_table["loss"]).all():
    raise RuntimeError(
        "The SIMCOMEN loss contains NaN or infinity."
    )


# =============================================================================
# SAVE MUTANT-ONLY RESULTS
# =============================================================================

adata.obs["simcomen_targeted"] = target_mask

adata.uns["simcomen_target_gene"] = TARGET_GENE
adata.uns["simcomen_target_populations"] = TARGET_POPULATIONS
adata.uns["simcomen_sample_filter"] = MUTANT_VALUE
adata.uns["simcomen_normalization"] = "unit_L2_per_cell"
adata.uns["simcomen_spherical_conversion"] = (
    "stable_tail_norm_atan2_and_vectorized_forward"
)
adata.uns["simcomen_roundtrip_maximum_allowed_error"] = (
    ROUNDTRIP_MAXIMUM_ALLOWED_ERROR
)


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
# SAVE MUTANT FILES
# =============================================================================

output_h5ad = (
    OUTPUT_DIR
    / "simcomen_perturbed_results.h5ad"
)

loss_path = (
    OUTPUT_DIR
    / "simcomen_training_loss.csv"
)

roundtrip_path = (
    OUTPUT_DIR
    / "simcomen_roundtrip_diagnostics.csv"
)

adata.write_h5ad(
    output_h5ad,
    compression="gzip",
)

loss_table.to_csv(
    loss_path,
    index=False,
)

pd.DataFrame(
    roundtrip_diagnostics
).to_csv(
    roundtrip_path,
    index=False,
)

print("SIMCOMEN mutant-only analysis completed.")
print(f"Saved: {output_h5ad}")
print(f"Saved: {loss_path}")
print(f"Saved: {roundtrip_path}")

