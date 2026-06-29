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

    prefix_sines = torch.cumprod(
        sin_angles,
        dim=1,
    )

    first_coordinate = cos_angles[
        :,
        :1,
    ]

    if sphex.shape[1] == 1:
        gex = torch.cat(
            [
                first_coordinate,
                prefix_sines[:, -1:],
            ],
            dim=1,
        )

    else:
        intermediate_coordinates = (
            prefix_sines[:, :-1]
            *
            cos_angles[:, 1:]
        )

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
    Convert expression to spherical coordinates and back, print comprehensive
    diagnostics, and abort if the reconstruction is not accurate.
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

_SHAM = SETTINGS["simcomen_sham"]
_group = _SHAM["group"]
SAMPLE_VALUES = [_group] if isinstance(_group, str) else list(_group)
TARGET_POPULATIONS_SHAM = list(_SHAM.get("target_populations") or [])

N_NEIGHBORS = _SHAM["n_neighbors"]
EPOCHS = _SHAM["epochs"]
LEARNING_RATE = _SHAM["learning_rate"]
ZMFT_SCALAR = _SHAM["zmft_scalar"]
SEED = _SHAM["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = CELCOMEN_DIR / "simcomen_sham"


# =============================================================================
# SHAM-SIMULATION FUNCTION
# =============================================================================


def run_sham_simcomen(
    adata_sample: ad.AnnData,
    *,
    sample_value: str,
    parameters: dict,
    genes: list[str],
) -> dict:
    """
    Run SIMCOMEN without changing any gene in one spatial sample.
    """

    print("\n" + "#" * 79)
    print(f"SHAM SIMCOMEN SAMPLE: {sample_value}")
    print("#" * 79)

    if adata_sample.n_obs <= N_NEIGHBORS:
        raise ValueError(
            f"Sample {sample_value!r} contains {adata_sample.n_obs} cells, "
            f"which is not enough for N_NEIGHBORS = {N_NEIGHBORS}."
        )

    if adata_sample.var_names.tolist() != genes:
        raise ValueError(
            f"The gene order for sample {sample_value!r} does not match "
            "the CELCOMEN parameters."
        )

    if "spatial" not in adata_sample.obsm:
        raise ValueError(
            f'adata.obsm["spatial"] was not found for sample {sample_value!r}.'
        )

    spatial_coordinates = np.asarray(
        adata_sample.obsm["spatial"]
    )

    if spatial_coordinates.ndim != 2:
        raise ValueError(
            f"Spatial coordinates for sample {sample_value!r} must be a "
            "two-dimensional array."
        )

    if not np.isfinite(spatial_coordinates).all():
        raise ValueError(
            f"Spatial coordinates for sample {sample_value!r} contain "
            "NaN or infinity."
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)


    if sparse.issparse(adata_sample.X):
        x_raw_array = adata_sample.X.toarray()
    else:
        x_raw_array = np.asarray(
            adata_sample.X
        )

    x_raw = torch.tensor(
        x_raw_array,
        dtype=torch.float32,
    )

    raw_norms = torch.linalg.vector_norm(
        x_raw,
        dim=1,
        keepdim=True,
    )

    if torch.any(raw_norms == 0).item():
        raise ValueError(
            f"At least one {sample_value} cell has zero expression across "
            "all model genes."
        )

    x_pre_sham = x_raw / raw_norms

    pre_sham_sphex_cpu, pre_sham_tensor = validate_roundtrip(
        x_pre_sham,
        label=f"{sample_value} sham starting state",
    )

    pre_sham_expression_direct = (
        pre_sham_tensor
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    graph = kneighbors_graph(
        spatial_coordinates,
        n_neighbors=N_NEIGHBORS,
        mode="connectivity",
        include_self=False,
    )

    distance_graph = kneighbors_graph(
        spatial_coordinates,
        n_neighbors=N_NEIGHBORS,
        mode="distance",
        include_self=False,
    )

    graph_row, graph_col = graph.nonzero()

    expected_edge_count = (
        adata_sample.n_obs
        * N_NEIGHBORS
    )

    if len(graph_row) != expected_edge_count:
        raise RuntimeError(
            f"Sample {sample_value!r} produced {len(graph_row)} directed "
            f"k-nearest-neighbor edges; expected {expected_edge_count}."
        )

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

    neighbor_distances = np.asarray(
        distance_graph.data,
        dtype=float,
    )

    if neighbor_distances.size != expected_edge_count:
        raise RuntimeError(
            f"Sample {sample_value!r} produced an unexpected number of "
            "neighbor-distance values."
        )

    graph_diagnostics = {
        "sample": sample_value,
        "n_cells": int(adata_sample.n_obs),
        "n_neighbors_per_cell": int(N_NEIGHBORS),
        "n_directed_edges": int(len(graph_row)),
        "minimum_neighbor_distance": float(neighbor_distances.min()),
        "median_neighbor_distance": float(np.median(neighbor_distances)),
        "mean_neighbor_distance": float(neighbor_distances.mean()),
        "maximum_neighbor_distance": float(neighbor_distances.max()),
        "cross_sample_edges": 0,
    }

    print(f"Cells: {adata_sample.n_obs}")
    print(f"Genes: {adata_sample.n_vars}")
    print(f"Directed spatial edges: {len(graph_row)}")
    print(
        "Neighbor-distance range:",
        graph_diagnostics["minimum_neighbor_distance"],
        "to",
        graph_diagnostics["maximum_neighbor_distance"],
    )
    print("Cross-sample edges: 0")
    print(f"Device: {DEVICE}")


    model = simcomen(
        input_dim=adata_sample.n_vars,
        output_dim=adata_sample.n_vars,
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
        pre_sham_sphex_cpu
        .to(DEVICE)
        .detach()
        .clone()
    )


    model.eval()

    with torch.no_grad():
        model(
            edge_index,
            1,
        )

    pre_sham_expression = (
        model.gex
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    model_initialization_maximum_difference = np.abs(
        pre_sham_expression
        - pre_sham_expression_direct
    ).max()

    print(
        "Maximum difference between the validated sham state and the "
        "patched model forward output:",
        model_initialization_maximum_difference,
    )

    if (
        model_initialization_maximum_difference
        > ROUNDTRIP_MAXIMUM_ALLOWED_ERROR
    ):
        raise RuntimeError(
            f"The patched model did not reproduce the validated "
            f"{sample_value} sham starting expression."
        )


    print(
        f"Running sham SIMCOMEN on {adata_sample.n_obs} {sample_value} cells "
        "with no gene deletion and no directly perturbed cells."
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

    model.eval()

    with torch.no_grad():
        model(
            edge_index,
            1,
        )

    if not torch.isfinite(model.sphex).all().item():
        raise RuntimeError(
            f"Final {sample_value} sham spherical coordinates contain "
            "NaN or infinity."
        )

    if not torch.isfinite(model.gex).all().item():
        raise RuntimeError(
            f"Final {sample_value} sham expression contains NaN or infinity."
        )

    post_sham_expression = (
        model.gex
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    post_norms = np.linalg.norm(
        post_sham_expression,
        axis=1,
    )

    delta_post_minus_pre = (
        post_sham_expression
        - pre_sham_expression
    )

    maximum_absolute_delta = float(
        np.max(
            np.abs(
                delta_post_minus_pre
            )
        )
    )

    mean_absolute_delta = float(
        np.mean(
            np.abs(
                delta_post_minus_pre
            )
        )
    )

    print(f"{sample_value} sham simulation completed.")
    print(
        "Final expression L2-norm range:",
        float(post_norms.min()),
        "to",
        float(post_norms.max()),
    )
    print(
        "Final expression value range:",
        float(post_sham_expression.min()),
        "to",
        float(post_sham_expression.max()),
    )
    print(
        "Maximum absolute post-minus-pre sham difference:",
        maximum_absolute_delta,
    )
    print(
        "Mean absolute post-minus-pre sham difference:",
        mean_absolute_delta,
    )


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
            "sample": sample_value,
            "epoch": np.arange(
                1,
                len(loss_values) + 1,
            ),
            "loss": loss_values,
        }
    )

    if len(loss_table) != EPOCHS:
        raise RuntimeError(
            f"The {sample_value} sham loss table contains "
            f"{len(loss_table)} rows instead of {EPOCHS}."
        )

    if not np.isfinite(loss_table["loss"]).all():
        raise RuntimeError(
            f"The {sample_value} sham loss contains NaN or infinity."
        )

    if loss_table["loss"].nunique() <= 1:
        raise RuntimeError(
            f"The {sample_value} sham loss did not change during training."
        )

    loss_array = loss_table["loss"].to_numpy()

    if len(loss_array) >= 25:
        recent_relative_improvement = (
            loss_array[-25]
            - loss_array[-1]
        ) / max(
            abs(loss_array[-25]),
            np.finfo(float).eps,
        )

        print(
            "Relative loss improvement over the final 25 epochs:",
            recent_relative_improvement,
        )

    if len(loss_array) >= 50:
        relative_improvement = (
            loss_array[-50]
            - loss_array[-1]
        ) / max(
            abs(loss_array[-50]),
            np.finfo(float).eps,
        )

        print(
            "Relative loss improvement over the final 50 epochs:",
            relative_improvement,
        )


    adata_sample.obs["simcomen_sham_run"] = True

    adata_sample.uns["simcomen_analysis"] = (
        "sham_no_gene_perturbation"
    )
    adata_sample.uns["simcomen_sample"] = sample_value
    adata_sample.uns["simcomen_sample_training"] = (
        "separate_per_spatial_sample"
    )
    adata_sample.uns["simcomen_directly_perturbed_cells"] = 0
    adata_sample.uns["simcomen_cross_sample_edges"] = False
    adata_sample.uns["simcomen_graph"] = (
        f"directed_{N_NEIGHBORS}_nearest_neighbors_within_sample"
    )
    adata_sample.uns["simcomen_graph_edge_weighting"] = (
        "unweighted_connectivity"
    )
    adata_sample.uns["simcomen_normalization"] = "unit_L2_per_cell"
    adata_sample.uns["simcomen_spherical_conversion"] = (
        "stable_tail_norm_atan2_and_vectorized_forward"
    )
    adata_sample.uns["simcomen_roundtrip_maximum_allowed_error"] = (
        ROUNDTRIP_MAXIMUM_ALLOWED_ERROR
    )
    adata_sample.uns["simcomen_n_neighbors"] = int(N_NEIGHBORS)
    adata_sample.uns["simcomen_epochs"] = int(EPOCHS)
    adata_sample.uns["simcomen_learning_rate"] = float(LEARNING_RATE)
    adata_sample.uns["simcomen_zmft_scalar"] = float(ZMFT_SCALAR)
    adata_sample.uns["simcomen_seed"] = int(SEED)

    adata_sample.layers["simcomen_pre_sham"] = (
        pre_sham_expression
    )

    adata_sample.layers["simcomen_post_sham"] = (
        post_sham_expression
    )

    adata_sample.layers["simcomen_delta_post_minus_pre_sham"] = (
        delta_post_minus_pre
    )

    output_h5ad = (
        OUTPUT_DIR
        / f"simcomen_sham_{sample_value}_results.h5ad"
    )

    loss_path = (
        OUTPUT_DIR
        / f"simcomen_training_loss_sham_{sample_value}.csv"
    )

    adata_sample.write_h5ad(
        output_h5ad,
        compression="gzip",
    )

    loss_table.to_csv(
        loss_path,
        index=False,
    )

    print(f"Saved: {output_h5ad}")
    print(f"Saved: {loss_path}")

    return {
        "pre_sham_expression": pre_sham_expression,
        "post_sham_expression": post_sham_expression,
        "delta_post_minus_pre": delta_post_minus_pre,
        "loss_table": loss_table,
        "graph_diagnostics": graph_diagnostics,
        "output_h5ad": output_h5ad,
        "loss_path": loss_path,
    }


# =============================================================================
# LOAD THE COMPLETE WT + MUT DATASET AND CELCOMEN PARAMETERS
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

adata_all = ad.read_h5ad(
    INPUT_H5AD
)

parameters = torch.load(
    PARAMETER_FILE,
    map_location="cpu",
    weights_only=False,
)

genes = list(
    parameters["genes"]
)

if adata_all.var_names.tolist() != genes:
    raise ValueError(
        "The gene order in the H5AD does not match the CELCOMEN parameters."
    )

if SAMPLE_COLUMN not in adata_all.obs.columns:
    raise ValueError(
        f"{SAMPLE_COLUMN!r} was not found in adata.obs."
    )

if "spatial" not in adata_all.obsm:
    raise ValueError(
        'adata.obsm["spatial"] was not found.'
    )

sample_labels = adata_all.obs[
    SAMPLE_COLUMN
].astype(str)

available_sample_values = set(
    sample_labels.unique()
)

missing_sample_values = [
    sample_value
    for sample_value in SAMPLE_VALUES
    if sample_value not in available_sample_values
]

if missing_sample_values:
    raise ValueError(
        f"The requested sample values {missing_sample_values} were not found "
        f"in {SAMPLE_COLUMN!r}. Available values: "
        f"{sorted(available_sample_values)}"
    )

keep_mask = sample_labels.isin(SAMPLE_VALUES).to_numpy()

if TARGET_POPULATIONS_SHAM:
    if ANNOTATION_COLUMN not in adata_all.obs.columns:
        raise ValueError(
            f"{ANNOTATION_COLUMN!r} was not found in adata.obs, but "
            "simcomen_sham.target_populations was set."
        )
    population_mask = (
        adata_all.obs[ANNOTATION_COLUMN]
        .astype(str)
        .isin(TARGET_POPULATIONS_SHAM)
        .to_numpy()
    )
    keep_mask = keep_mask & population_mask

if not keep_mask.any():
    raise ValueError(
        "No cells matched the configured sham groups/populations."
    )

adata_all = adata_all[keep_mask].copy()

sample_labels = adata_all.obs[SAMPLE_COLUMN].astype(str)

print("Starting WT + MUT sham SIMCOMEN analysis.")
print(f"Samples: {SAMPLE_VALUES}")
print(f"Total cells: {adata_all.n_obs}")
print(f"Genes: {adata_all.n_vars}")
print(f"Neighbors per cell: {N_NEIGHBORS}")
print(f"Epochs per sample: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"ZMFT scalar: {ZMFT_SCALAR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Device: {DEVICE}")

combined_pre_sham = np.empty(
    (adata_all.n_obs, adata_all.n_vars),
    dtype=np.float32,
)

combined_post_sham = np.empty(
    (adata_all.n_obs, adata_all.n_vars),
    dtype=np.float32,
)

combined_delta = np.empty(
    (adata_all.n_obs, adata_all.n_vars),
    dtype=np.float32,
)

completed_cells = np.zeros(
    adata_all.n_obs,
    dtype=bool,
)

loss_tables = []
graph_diagnostics_records = []

for sample_value in SAMPLE_VALUES:
    sample_mask = (
        sample_labels
        .eq(sample_value)
        .to_numpy()
    )

    sample_indices = np.flatnonzero(
        sample_mask
    )

    adata_sample = adata_all[
        sample_mask
    ].copy()

    result = run_sham_simcomen(
        adata_sample,
        sample_value=sample_value,
        parameters=parameters,
        genes=genes,
    )

    combined_pre_sham[
        sample_indices,
        :,
    ] = result[
        "pre_sham_expression"
    ]

    combined_post_sham[
        sample_indices,
        :,
    ] = result[
        "post_sham_expression"
    ]

    combined_delta[
        sample_indices,
        :,
    ] = result[
        "delta_post_minus_pre"
    ]

    completed_cells[
        sample_indices
    ] = True

    loss_tables.append(
        result["loss_table"]
    )

    graph_diagnostics_records.append(
        result["graph_diagnostics"]
    )

if not completed_cells.all():
    raise RuntimeError(
        "At least one cell was not included in the WT or MUT sham run."
    )

if not np.isfinite(combined_pre_sham).all():
    raise RuntimeError(
        "The combined pre-sham expression contains NaN or infinity."
    )

if not np.isfinite(combined_post_sham).all():
    raise RuntimeError(
        "The combined post-sham expression contains NaN or infinity."
    )

if not np.isfinite(combined_delta).all():
    raise RuntimeError(
        "The combined sham difference matrix contains NaN or infinity."
    )


# =============================================================================
# SAVE THE COMBINED WT + MUT RESULT
# =============================================================================

adata_all.obs["simcomen_sham_run"] = (
    completed_cells
)

adata_all.uns["simcomen_analysis"] = (
    "sham_no_gene_perturbation"
)
adata_all.uns["simcomen_samples"] = SAMPLE_VALUES
adata_all.uns["simcomen_sample_training"] = (
    "separate_per_spatial_sample_then_combined_for_output"
)
adata_all.uns["simcomen_directly_perturbed_cells"] = 0
adata_all.uns["simcomen_cross_sample_edges"] = False
adata_all.uns["simcomen_graph"] = (
    f"directed_{N_NEIGHBORS}_nearest_neighbors_within_each_sample"
)
adata_all.uns["simcomen_graph_edge_weighting"] = (
    "unweighted_connectivity"
)
adata_all.uns["simcomen_normalization"] = "unit_L2_per_cell"
adata_all.uns["simcomen_spherical_conversion"] = (
    "stable_tail_norm_atan2_and_vectorized_forward"
)
adata_all.uns["simcomen_roundtrip_maximum_allowed_error"] = (
    ROUNDTRIP_MAXIMUM_ALLOWED_ERROR
)
adata_all.uns["simcomen_n_neighbors"] = int(N_NEIGHBORS)
adata_all.uns["simcomen_epochs_per_sample"] = int(EPOCHS)
adata_all.uns["simcomen_learning_rate"] = float(LEARNING_RATE)
adata_all.uns["simcomen_zmft_scalar"] = float(ZMFT_SCALAR)
adata_all.uns["simcomen_seed"] = int(SEED)

adata_all.layers["simcomen_pre_sham"] = (
    combined_pre_sham
)

adata_all.layers["simcomen_post_sham"] = (
    combined_post_sham
)

adata_all.layers["simcomen_delta_post_minus_pre_sham"] = (
    combined_delta
)

combined_output_h5ad = (
    OUTPUT_DIR
    / "simcomen_sham_results.h5ad"
)

combined_loss_path = (
    OUTPUT_DIR
    / "simcomen_training_loss_sham.csv"
)

roundtrip_path = (
    OUTPUT_DIR
    / "simcomen_roundtrip_diagnostics_sham.csv"
)

graph_diagnostics_path = (
    OUTPUT_DIR
    / "simcomen_graph_diagnostics_sham.csv"
)

adata_all.write_h5ad(
    combined_output_h5ad,
    compression="gzip",
)

pd.concat(
    loss_tables,
    axis=0,
    ignore_index=True,
).to_csv(
    combined_loss_path,
    index=False,
)

pd.DataFrame(
    roundtrip_diagnostics
).to_csv(
    roundtrip_path,
    index=False,
)

pd.DataFrame(
    graph_diagnostics_records
).to_csv(
    graph_diagnostics_path,
    index=False,
)

print("\n" + "=" * 79)
print("WT + MUT SHAM SIMCOMEN ANALYSIS COMPLETED.")
print("=" * 79)
print(f"Saved combined result: {combined_output_h5ad}")
print(f"Saved combined loss: {combined_loss_path}")
print(f"Saved round-trip diagnostics: {roundtrip_path}")
print(f"Saved graph diagnostics: {graph_diagnostics_path}")
print("No gene was deleted or otherwise directly perturbed.")
print("WT and MUT were optimized separately with identical hyperparameters.")
print("The combined H5AD preserves the original all-cell ordering.")
print("=" * 79)
