library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
library(SeuratObject)

# =============================================================================
# CONFIGURATION
# =============================================================================
command_line_args <- commandArgs(trailingOnly = TRUE)
settings_flag <- match("--settings", command_line_args)

if (!is.na(settings_flag)) {
  if (settings_flag == length(command_line_args)) {
    stop("--settings must be followed by the path to settings.yaml.")
  }
  settings_path <- command_line_args[settings_flag + 1L]
} else {
  settings_path <- "settings.yaml"
}

settings_path <- normalizePath(
  path.expand(settings_path),
  mustWork = TRUE
)

settings <- yaml::read_yaml(settings_path)
project_dir <- path.expand(settings$project$project_dir)

python_env <- settings$export$python_env
reticulate::use_condaenv(python_env, required = TRUE)

celcomen_dir <- file.path(project_dir, "celcomen_input_rna_top_genes", "celcomen_output_rna_top_genes")

results_subdir   <- "simcomen_perturbed"
reference_subdir <- "simcomen_reference"

mut_h5ad_name <- "simcomen_perturbed_results.h5ad"
ref_h5ad_name <- "simcomen_reference_pre_states.h5ad"

h5ad_file_mut <- file.path(celcomen_dir, results_subdir, mut_h5ad_name)
h5ad_file_ref <- file.path(celcomen_dir, reference_subdir, ref_h5ad_name)

output_ref_rds <- file.path(project_dir, "simcomen_seurat_reference.rds")
output_mut_rds <- file.path(project_dir, "simcomen_seurat_perturbed.rds")


source_assay   <- "originalexp"             
mut_assay      <- "perturbed"
ref_assay      <- "reference"
mut_data_layer <- "simcomen_post_perturbed"
ref_data_layer <- "simcomen_pre_normal"
mut_project    <- "SIMCOMEN_perturbed"
ref_project    <- "SIMCOMEN_reference"

# --- Populations -------------------------------------------------------------
annotation_column      <- settings$cohort$annotation_column
mut_subset_populations <- unlist(settings$cohort$target_populations)
ref_subset_populations <- unlist(settings$cohort$reference_populations)

# --- Annotation display labels + colors --------------------------------------
anno_label_map <- unlist(settings$display$labels)
anno_color_map <- unlist(settings$display$colors)
anno_order_raw <- unlist(settings$display$order_raw)

anno_order_display <- unname(anno_label_map[anno_order_raw])
anno_display_color_map <- anno_color_map[anno_order_raw]
names(anno_display_color_map) <- anno_order_display

reference_cell_size <- settings$display$reference_cell_size

# --- Cloud-label styling -----------------------------------------------------
label_fontsize <- settings$plots$label_fontsize
label_stroke_linewidth <- settings$plots$label_stroke_linewidth

# --- PNG outputs -------------------
output_png_dir <- file.path(project_dir, "Simcomen_diff_results", "graphs")
dir.create(output_png_dir, recursive = TRUE, showWarnings = FALSE)
combined_preview_name <- "streamplot_with_background.png"
combined_transparent_name <- "streamplot_with_background_transparent.png"

# --- Fixed clustering / projection defaults ----------------------------------
elbow_ndims <- 50
pca_dims <- 1:25
ref_cluster_resolution <- 0.5

##########################################################################

sce_mut <- readH5AD(h5ad_file_mut)

simcomen_seurat_mut <- as.Seurat(
  x = sce_mut,
  counts = NULL,
  data = mut_data_layer,
  project = mut_project
)

simcomen_seurat_mut <- RenameAssays(
  simcomen_seurat_mut,
  assay.name = source_assay,
  new.assay.name = mut_assay)

mut_keep_cells <- colnames(simcomen_seurat_mut)[
  as.character(simcomen_seurat_mut@meta.data[[annotation_column]]) %in% mut_subset_populations
]
simcomen_seurat_mut <- subset(simcomen_seurat_mut, cells = mut_keep_cells)
#################

sce_ref <- readH5AD(h5ad_file_ref)

simcomen_seurat_ref <- as.Seurat(
  x = sce_ref,
  counts = NULL,
  data = ref_data_layer,
  project = ref_project
)

simcomen_seurat_ref <- RenameAssays(
  simcomen_seurat_ref,
  assay.name = source_assay,
  new.assay.name = ref_assay)

ref_keep_cells <- colnames(simcomen_seurat_ref)[
  as.character(simcomen_seurat_ref@meta.data[[annotation_column]]) %in% ref_subset_populations
]
simcomen_seurat_ref <- subset(simcomen_seurat_ref, cells = ref_keep_cells)

#############################################################################

#############################################################################

apply_anno_mapping <- function(
  seu,
  anno_col = annotation_column,
  new_col = "AnnoGroup_display"
) {

  if (!anno_col %in% colnames(seu@meta.data)) {
    stop(
      paste0(
        "Metadata column '",
        anno_col,
        "' was not found in the Seurat object."
      )
    )
  }

  raw_anno <- as.character(
    seu@meta.data[[anno_col]]
  )

  display_anno <- ifelse(
    raw_anno %in% names(anno_label_map),
    unname(anno_label_map[raw_anno]),
    raw_anno
  )

  display_levels <- c(
    anno_order_display,
    setdiff(unique(display_anno), anno_order_display)
  )

  seu@meta.data[[new_col]] <- factor(
    display_anno,
    levels = display_levels
  )

  return(seu)
}

simcomen_seurat_ref <- apply_anno_mapping(
  simcomen_seurat_ref
)

simcomen_seurat_mut <- apply_anno_mapping(
  simcomen_seurat_mut
)

stopifnot(
  "AnnoGroup_display" %in% colnames(simcomen_seurat_ref@meta.data),
  "AnnoGroup_display" %in% colnames(simcomen_seurat_mut@meta.data)
)

#############################################################################


VariableFeatures(
    simcomen_seurat_ref[[ref_assay]]
) <- rownames(
    simcomen_seurat_ref[[ref_assay]]
)

simcomen_seurat_ref <- ScaleData(
    simcomen_seurat_ref,
    assay = ref_assay,
    features = VariableFeatures(
        simcomen_seurat_ref[[ref_assay]]
    ),
    do.center = TRUE,
    do.scale = TRUE
)

simcomen_seurat_ref <- RunPCA(simcomen_seurat_ref) 
ElbowPlot(simcomen_seurat_ref, ndims = elbow_ndims)
simcomen_seurat_ref <- FindNeighbors(simcomen_seurat_ref, dims = pca_dims) 
simcomen_seurat_ref <- FindClusters(simcomen_seurat_ref, resolution = ref_cluster_resolution) 
simcomen_seurat_ref <- RunUMAP(simcomen_seurat_ref, reduction = "pca", dims = pca_dims, reduction.name = "umap", return.model = TRUE)

DimPlot(simcomen_seurat_ref, group.by = annotation_column, reduction = "umap")
saveRDS(simcomen_seurat_ref, file = output_ref_rds)


#############################################################################
# PROJECT THE POST-SIMCOMEN MUTANT CELLS INTO THE REFERENCE PCA AND UMAP
#############################################################################

DefaultAssay(simcomen_seurat_mut) <- mut_assay

reference_pca_loadings <- Loadings(
  simcomen_seurat_ref[["pca"]]
)

features_use <- rownames(reference_pca_loadings)

reference_data <- as.matrix(
  LayerData(
    simcomen_seurat_ref,
    assay = ref_assay,
    layer = "data"
  )[features_use, , drop = FALSE]
)

mutant_data <- as.matrix(
  LayerData(
    simcomen_seurat_mut,
    assay = mut_assay,
    layer = "data"
  )[features_use, , drop = FALSE]
)

stopifnot(
  identical(
    rownames(reference_data),
    rownames(mutant_data)
  )
)

#############################################################################
# SCALE THE MUTANT CELLS USING THE REFERENCE MEAN AND SD
#############################################################################

reference_gene_mean <- rowMeans(reference_data)

reference_gene_sd <- apply(
  reference_data,
  1,
  sd
)

if (any(!is.finite(reference_gene_sd) | reference_gene_sd == 0)) {
  stop(
    "At least one reference PCA gene has zero or non-finite SD."
  )
}

mutant_scaled <- sweep(
  mutant_data,
  MARGIN = 1,
  STATS = reference_gene_mean,
  FUN = "-"
)

mutant_scaled <- sweep(
  mutant_scaled,
  MARGIN = 1,
  STATS = reference_gene_sd,
  FUN = "/"
)

mutant_scaled[
  mutant_scaled > 10
] <- 10

mutant_scaled[
  mutant_scaled < -10
] <- -10

LayerData(
  simcomen_seurat_mut,
  assay = mut_assay,
  layer = "scale.data"
) <- mutant_scaled


VariableFeatures(
    simcomen_seurat_mut[[mut_assay]]
) <- rownames(
    simcomen_seurat_ref[[ref_assay]]
)
#############################################################################
# PROJECT MUTANT CELLS INTO THE EXISTING REFERENCE PCA
#############################################################################

simcomen_seurat_mut <- ProjectDimReduc(
  query = simcomen_seurat_mut,
  reference = simcomen_seurat_ref,

  mode = "pcaproject",

  query.assay = mut_assay,
  reference.assay = ref_assay,

  reference.reduction = "pca",

  do.scale = TRUE,

  reduction.name = "pca.ref",
  reduction.key = "PCref_",

  verbose = TRUE
)
#############################################################################
# PROJECT THE MUTANT PCA COORDINATES INTO THE REFERENCE UMAP
#############################################################################

simcomen_seurat_mut <- ProjectUMAP(
  query = simcomen_seurat_mut,
  query.reduction = "pca.ref",
  query.dims = pca_dims,

  reference = simcomen_seurat_ref,
  reference.reduction = "pca",
  reference.dims = pca_dims,

  reduction.model = "umap",

  reduction.name = "ref.umap",
  reduction.key = "refUMAP_"
)

#############################################################################
# PROJECT THE REFERENCE-PROJECTED PCA INTO THE EXISTING REFERENCE UMAP
#############################################################################

DimPlot(simcomen_seurat_mut, group.by = annotation_column, reduction = "ref.umap")
simcomen_seurat_mut[["query_ref.nn"]] <- NULL
saveRDS(simcomen_seurat_mut, file = output_mut_rds)

#############################################################################
# SCVELO-LIKE RelB-KO DISPLACEMENT FIELD ON THE REFERENCE UMAP
#############################################################################

reference_umap <- Embeddings(
  simcomen_seurat_ref,
  reduction = "umap"
)[, 1:2, drop = FALSE]

mutant_umap <- Embeddings(
  simcomen_seurat_mut,
  reduction = "ref.umap"
)[, 1:2, drop = FALSE]

common_cells <- rownames(mutant_umap)[
  rownames(mutant_umap) %in% rownames(reference_umap)
]


stopifnot(length(common_cells) == nrow(mutant_umap))

X <- reference_umap[common_cells, , drop = FALSE]

V <- mutant_umap[common_cells, , drop = FALSE] -
  reference_umap[common_cells, , drop = FALSE]

valid_cells <- apply(X, 1, function(x) all(is.finite(x))) &
  apply(V, 1, function(x) all(is.finite(x)))

X <- X[valid_cells, , drop = FALSE]
V <- V[valid_cells, , drop = FALSE]

#############################################################################
# SCVELO-LIKE GRID SMOOTHING
#############################################################################

density <- 0.60       # lower values create a coarser vector grid
smooth <- 0.50        # higher values produce stronger local smoothing
min_mass <- 1.00      # higher values remove more weakly supported arrows

n_grid <- max(
  10L,
  as.integer(50 * density)
)

n_neighbors <- max(
  1L,
  min(
    nrow(X),
    as.integer(nrow(X) / 50)
  )
)

make_grid_axis <- function(x, n) {

  limits <- range(x, finite = TRUE)

  if (diff(limits) == 0) {
    stop("One UMAP dimension has zero range.")
  }

  padding <- 0.01 * diff(limits)

  seq(
    limits[1] - padding,
    limits[2] + padding,
    length.out = n
  )
}

grid_x <- make_grid_axis(X[, 1], n_grid)
grid_y <- make_grid_axis(X[, 2], n_grid)

grid_xy <- expand.grid(
  UMAP_1 = grid_x,
  UMAP_2 = grid_y
)

nn <- RANN::nn2(
  data = X,
  query = as.matrix(grid_xy),
  k = n_neighbors
)

nn_index <- as.matrix(nn$nn.idx)
nn_distance <- as.matrix(nn$nn.dists)

kernel_sd <- mean(
  c(
    diff(grid_x)[1],
    diff(grid_y)[1]
  )
) * smooth

weights <- dnorm(
  nn_distance,
  mean = 0,
  sd = kernel_sd
)

probability_mass <- rowSums(weights)

neighbor_v1 <- matrix(
  V[, 1][as.vector(nn_index)],
  nrow = nrow(nn_index),
  ncol = ncol(nn_index)
)

neighbor_v2 <- matrix(
  V[, 2][as.vector(nn_index)],
  nrow = nrow(nn_index),
  ncol = ncol(nn_index)
)

V_grid <- cbind(
  rowSums(neighbor_v1 * weights),
  rowSums(neighbor_v2 * weights)
)

V_grid <- sweep(
  V_grid,
  MARGIN = 1,
  STATS = pmax(1, probability_mass),
  FUN = "/"
)

mass_threshold <- min_mass *
  as.numeric(
    quantile(
      probability_mass,
      probs = 0.99,
      names = FALSE
    )
  ) / 100

#############################################################################
# SMALL GRID ARROWS OVER THE REFERENCE CELLS
#############################################################################

arrow_length <- 1.00

vector_size <- sqrt(
  rowSums(V_grid^2)
)

keep_arrow <- probability_mass > mass_threshold &
  is.finite(vector_size) &
  vector_size > 0

if (!any(keep_arrow)) {
  stop(
    "No arrows passed the filtering. Reduce min_mass or increase smooth."
  )
}

reference_size <- as.numeric(
  quantile(
    vector_size[keep_arrow],
    probs = 0.95,
    names = FALSE
  )
)

target_size <- 0.80 * mean(
  c(
    diff(sort(unique(grid_xy$UMAP_1)))[1],
    diff(sort(unique(grid_xy$UMAP_2)))[1]
  )
)

visual_scale <- arrow_length *
  target_size /
  reference_size

arrow_data <- data.frame(
  x = grid_xy$UMAP_1[keep_arrow],
  y = grid_xy$UMAP_2[keep_arrow],
  dx = V_grid[keep_arrow, 1] * visual_scale,
  dy = V_grid[keep_arrow, 2] * visual_scale
)

arrow_data$xend <- arrow_data$x + arrow_data$dx
arrow_data$yend <- arrow_data$y + arrow_data$dy

p_reference <- DimPlot(
  simcomen_seurat_ref,
  group.by = "AnnoGroup_display",
  reduction = "umap",
  cols = anno_display_color_map,
  pt.size = reference_cell_size,
  order = anno_order_display
)

p_reference_arrows <- p_reference +
  ggplot2::geom_segment(
    data = arrow_data,
    mapping = ggplot2::aes(
      x = x,
      y = y,
      xend = xend,
      yend = yend
    ),
    inherit.aes = FALSE,
    colour = "grey20",
    linewidth = 0.35,
    alpha = 0.85,
    lineend = "round",
    arrow = grid::arrow(
      type = "closed",
      length = grid::unit(
        0.10,
        "inches"
      )
    )
  ) +
  ggplot2::ggtitle(
    "Reference to simulated perturbation displacement field"
  )

p_reference_arrows

#############################################################################
# CONTINUOUS POPULATION FIELD WITH AUTOMATED LABELS + COLORS
#############################################################################

reference_umap_all <- Embeddings(
  simcomen_seurat_ref,
  reduction = "umap"
)[, 1:2, drop = FALSE]

ref_plot_df <- data.frame(
  cell = rownames(reference_umap_all),
  UMAP_1 = reference_umap_all[, 1],
  UMAP_2 = reference_umap_all[, 2],
  AnnoGroup_raw = as.character(
    simcomen_seurat_ref$AnnoGroup[
      rownames(reference_umap_all)
    ]
  ),
  AnnoGroup_display = as.character(
    simcomen_seurat_ref$AnnoGroup_display[
      rownames(reference_umap_all)
    ]
  ),
  stringsAsFactors = FALSE
)

groups_use_raw <- intersect(
  anno_order_raw,
  unique(ref_plot_df$AnnoGroup_raw)
)

ref_plot_df <- ref_plot_df[
  ref_plot_df$AnnoGroup_raw %in% groups_use_raw &
    is.finite(ref_plot_df$UMAP_1) &
    is.finite(ref_plot_df$UMAP_2),
  ,
  drop = FALSE
]

groups_use_display <- unname(
  anno_label_map[groups_use_raw]
)

ref_plot_df$AnnoGroup_display <- factor(
  ref_plot_df$AnnoGroup_display,
  levels = groups_use_display
)

group_cols <- anno_display_color_map[
  groups_use_display
]

X_background <- as.matrix(
  ref_plot_df[, c("UMAP_1", "UMAP_2")]
)

group_index <- as.integer(
  ref_plot_df$AnnoGroup_display
)

stopifnot(
  nrow(X_background) > 10,
  !anyNA(group_index)
)

#############################################################################
# PARAMETERS
#############################################################################

grid_nx <- 420L
k_colour <- min(60L, nrow(X_background))
colour_bandwidth_factor <- 1.35
fill_radius_factor <- 1.20
colour_sharpness <- 1.60
maximum_alpha <- 0.78
white_mixing <- 0.10

#############################################################################
# ESTIMATE TYPICAL CELL SPACING
#############################################################################

k_spacing <- min(
  10L,
  nrow(X_background) - 1L
)

cell_nn <- RANN::nn2(
  data = X_background,
  query = X_background,
  k = k_spacing + 1L
)

cell_spacing <- cell_nn$nn.dists[
  ,
  k_spacing + 1L
]

cell_spacing <- cell_spacing[
  is.finite(cell_spacing) &
    cell_spacing > 0
]

if (length(cell_spacing) == 0) {
  stop("Could not estimate the local UMAP cell spacing.")
}

typical_spacing <- median(
  cell_spacing
)

colour_bandwidth <- typical_spacing *
  colour_bandwidth_factor

fill_radius <- as.numeric(
  quantile(
    cell_spacing,
    probs = 0.90,
    names = FALSE
  )
) * fill_radius_factor

#############################################################################
# CREATE DENSE GRID
#############################################################################

x_range <- range(
  X_background[, 1],
  finite = TRUE
)

y_range <- range(
  X_background[, 2],
  finite = TRUE
)

x_padding <- 0.03 * diff(x_range)
y_padding <- 0.03 * diff(y_range)

x_limits <- x_range + c(-x_padding, x_padding)
y_limits <- y_range + c(-y_padding, y_padding)

grid_ny <- max(
  100L,
  as.integer(
    round(
      grid_nx *
        diff(y_limits) /
        diff(x_limits)
    )
  )
)

grid_x <- seq(
  x_limits[1],
  x_limits[2],
  length.out = grid_nx
)

grid_y <- seq(
  y_limits[1],
  y_limits[2],
  length.out = grid_ny
)

background_grid <- expand.grid(
  UMAP_1 = grid_x,
  UMAP_2 = grid_y
)

#############################################################################
# LOCAL GROUP MIXING ON THE GRID
#############################################################################

grid_nn <- RANN::nn2(
  data = X_background,
  query = as.matrix(background_grid),
  k = k_colour
)

nn_index <- as.matrix(
  grid_nn$nn.idx
)

nn_distance <- as.matrix(
  grid_nn$nn.dists
)

neighbor_groups <- matrix(
  group_index[as.vector(nn_index)],
  nrow = nrow(nn_index),
  ncol = ncol(nn_index)
)

weights <- exp(
  -0.5 * (nn_distance / colour_bandwidth)^2
)

population_scores <- matrix(
  0,
  nrow = nrow(background_grid),
  ncol = length(groups_use_display)
)

colnames(population_scores) <- groups_use_display

for (g in seq_along(groups_use_display)) {
  population_scores[, g] <- rowSums(
    weights * (neighbor_groups == g)
  )
}

score_sum <- rowSums(
  population_scores
)

population_probability <- population_scores /
  pmax(score_sum, .Machine$double.eps)

population_probability <- population_probability^colour_sharpness

population_probability <- population_probability /
  pmax(
    rowSums(population_probability),
    .Machine$double.eps
  )

#############################################################################
# CONVERT POPULATION PROBABILITIES TO RGB
#############################################################################

group_rgb <- t(
  grDevices::col2rgb(
    group_cols[groups_use_display]
  )
) / 255

mixed_rgb <- population_probability %*%
  group_rgb

mixed_rgb <- (1 - white_mixing) * mixed_rgb +
  white_mixing

#############################################################################
# ONE SHARED UMAP-SHAPED MASK
#############################################################################

distance_to_nearest_cell <- nn_distance[, 1]

fade_start <- 0.60 * fill_radius

edge_position <- (
  fill_radius - distance_to_nearest_cell
) / (
  fill_radius - fade_start
)

edge_position <- pmax(
  0,
  pmin(1, edge_position)
)

edge_alpha <- edge_position^2 *
  (3 - 2 * edge_position)

pixel_alpha <- maximum_alpha * edge_alpha

background_grid$pixel_colour <- grDevices::rgb(
  red = mixed_rgb[, 1],
  green = mixed_rgb[, 2],
  blue = mixed_rgb[, 3],
  alpha = pixel_alpha
)

#############################################################################
# AUTOMATED LABEL POSITIONS
#############################################################################

label_df <- aggregate(
  ref_plot_df[, c("UMAP_1", "UMAP_2")],
  by = list(label = ref_plot_df$AnnoGroup_display),
  FUN = median
)

colnames(label_df)[2:3] <- c("x", "y")

#############################################################################
# MATPLOTLIB STREAMPLOT WITH THE SMOOTH UMAP BACKGROUND
#############################################################################

stopifnot(
  exists("background_grid"),
  exists("grid_xy"),
  exists("V_grid"),
  exists("probability_mass"),
  exists("mass_threshold")
)

stream_x <- sort(unique(grid_xy$UMAP_1))
stream_y <- sort(unique(grid_xy$UMAP_2))

if (
  nrow(grid_xy) !=
    length(stream_x) * length(stream_y)
) {
  stop("grid_xy is not a complete rectangular vector-field grid.")
}

stream_order <- order(
  grid_xy$UMAP_2,
  grid_xy$UMAP_1
)

U_values <- V_grid[stream_order, 1]
V_values <- V_grid[stream_order, 2]

keep_stream <- probability_mass[stream_order] > mass_threshold &
  is.finite(U_values) &
  is.finite(V_values) &
  sqrt(U_values^2 + V_values^2) > 0

if (!any(keep_stream)) {
  stop(
    paste0(
      "No stream vectors passed the filtering. ",
      "Reduce min_mass or increase smooth."
    )
  )
}

U_values[!keep_stream] <- NA_real_
V_values[!keep_stream] <- NA_real_

U_mat <- matrix(
  U_values,
  nrow = length(stream_y),
  ncol = length(stream_x),
  byrow = TRUE
)

V_mat <- matrix(
  V_values,
  nrow = length(stream_y),
  ncol = length(stream_x),
  byrow = TRUE
)

required_background_columns <- c(
  "UMAP_1",
  "UMAP_2",
  "pixel_colour"
)

if (!all(required_background_columns %in% colnames(background_grid))) {
  stop(
    "background_grid must contain UMAP_1, UMAP_2 and pixel_colour."
  )
}

#############################################################################
# 1. RECONSTRUCT THE BACKGROUND RGBA IMAGE
#############################################################################

background_x <- sort(
  unique(background_grid$UMAP_1)
)

background_y <- sort(
  unique(background_grid$UMAP_2)
)

nx_background <- length(background_x)
ny_background <- length(background_y)

if (
  nrow(background_grid) !=
    nx_background * ny_background
) {
  stop("background_grid is not a complete rectangular raster.")
}

background_ordered <- background_grid[
  order(
    background_grid$UMAP_2,
    background_grid$UMAP_1
  ),
  ,
  drop = FALSE
]

background_rgba_values <- grDevices::col2rgb(
  background_ordered$pixel_colour,
  alpha = TRUE
) / 255

background_rgba <- array(
  0,
  dim = c(
    ny_background,
    nx_background,
    4L
  )
)

for (channel in 1:4) {

  background_rgba[, , channel] <- matrix(
    background_rgba_values[channel, ],
    nrow = ny_background,
    ncol = nx_background,
    byrow = TRUE
  )
}

#############################################################################
# 2. LABELS
#############################################################################

if (!exists("label_df")) {

  label_df <- aggregate(
    ref_plot_df[, c("UMAP_1", "UMAP_2")],
    by = list(
      label = ref_plot_df$AnnoGroup_display
    ),
    FUN = median
  )

  colnames(label_df)[2:3] <- c(
    "x",
    "y"
  )
}

#############################################################################
# 3. VISUAL SETTINGS
#############################################################################

stream_density <- 1.25
stream_linewidth <- 1.35
stream_arrowsize <- 1.45
stream_maxlength <- 4.0
stream_minlength <- 0.12

#############################################################################
# 4. OUTPUT FILES
#############################################################################

combined_preview_file <- file.path(
  output_png_dir,
  combined_preview_name
)

combined_transparent_file <- file.path(
  output_png_dir,
  combined_transparent_name
)

#############################################################################
# 5. TRANSFER EVERYTHING TO PYTHON
#############################################################################

python_main <- reticulate::import_main(
  convert = TRUE
)

python_main$x <- as.numeric(stream_x)
python_main$y <- as.numeric(stream_y)
python_main$u <- U_mat
python_main$v <- V_mat

python_main$bg_rgba <- background_rgba

python_main$bg_xmin <- min(background_x)
python_main$bg_xmax <- max(background_x)
python_main$bg_ymin <- min(background_y)
python_main$bg_ymax <- max(background_y)

# Labels
python_main$label_x <- as.numeric(label_df$x)
python_main$label_y <- as.numeric(label_df$y)
python_main$label_text <- as.character(label_df$label)
python_main$label_fontsize <- label_fontsize
python_main$label_stroke <- label_stroke_linewidth

# Streamline settings
python_main$stream_density <- stream_density
python_main$stream_linewidth <- stream_linewidth
python_main$stream_arrowsize <- stream_arrowsize
python_main$stream_maxlength <- stream_maxlength
python_main$stream_minlength <- stream_minlength

# Output
python_main$combined_preview_path <- combined_preview_file
python_main$combined_transparent_path <- combined_transparent_file

#############################################################################
# 6. DRAW BACKGROUND + MATPLOTLIB STREAMLINES
#############################################################################

reticulate::py_run_string(
"
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)

u = np.asarray(u, dtype=float)
v = np.asarray(v, dtype=float)

bg_rgba = np.asarray(bg_rgba, dtype=float)

expected_shape = (len(y), len(x))

if u.shape != expected_shape:
    raise ValueError(
        f'U has shape {u.shape}; expected {expected_shape}.'
    )

if v.shape != expected_shape:
    raise ValueError(
        f'V has shape {v.shape}; expected {expected_shape}.'
    )

if bg_rgba.ndim != 3 or bg_rgba.shape[2] != 4:
    raise ValueError(
        f'Background must have shape (height, width, 4); '
        f'found {bg_rgba.shape}.'
    )

invalid = (~np.isfinite(u)) | (~np.isfinite(v))

u_masked = np.ma.masked_where(
    invalid,
    u
)

v_masked = np.ma.masked_where(
    invalid,
    v
)


def draw_combined_plot(save_path, transparent):

    fig, ax = plt.subplots(
        figsize=(8, 8),
        dpi=300
    )

    ax.imshow(
        bg_rgba,
        origin='lower',
        extent=[
            float(bg_xmin),
            float(bg_xmax),
            float(bg_ymin),
            float(bg_ymax)
        ],
        interpolation='bilinear',
        aspect='auto',
        zorder=1
    )

    ax.streamplot(
        x,
        y,
        u_masked,
        v_masked,

        density=float(stream_density),

        color='black',
        linewidth=float(stream_linewidth),

        arrowsize=float(stream_arrowsize),
        arrowstyle='-|>',

        minlength=float(stream_minlength),
        maxlength=float(stream_maxlength),

        integration_direction='both',
        broken_streamlines=True,

        zorder=5
    )

    for xi, yi, lab in zip(
        label_x,
        label_y,
        label_text
    ):
        txt = ax.text(
            float(xi),
            float(yi),
            str(lab),

            horizontalalignment='center',
            verticalalignment='center',

            fontsize=int(label_fontsize),
            fontweight='bold',
            color='black',

            zorder=10
        )
        
        txt.set_path_effects([
	    pe.Stroke(linewidth=float(label_stroke), foreground='white'),
	    pe.Normal()
	])
	    
    ax.set_xlim(
        float(bg_xmin),
        float(bg_xmax)
    )

    ax.set_ylim(
        float(bg_ymin),
        float(bg_ymax)
    )

    ax.set_aspect(
        'equal',
        adjustable='box'
    )

    ax.axis('off')

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        facecolor = 'none'
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        facecolor = 'white'

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.02,
        transparent=transparent,
        facecolor=facecolor
    )

    plt.close(fig)


draw_combined_plot(
    combined_preview_path,
    transparent=False
)

draw_combined_plot(
    combined_transparent_path,
    transparent=True
)
"
)

#############################################################################
# 7. SHOW THE COMPLETE FIGURE IN RSTUDIO
#############################################################################

if (!requireNamespace("png", quietly = TRUE)) {
  install.packages("png")
}

combined_preview_image <- png::readPNG(
  combined_preview_file
)

grid::grid.newpage()

grid::grid.raster(
  combined_preview_image
)

cat(
  "\nComplete plot:\n",
  normalizePath(
    combined_preview_file,
    mustWork = TRUE
  ),
  "\n\nTransparent version:\n",
  normalizePath(
    combined_transparent_file,
    mustWork = TRUE
  ),
  "\n"
)
