# =============================================================================
library(Seurat)
library(Matrix)

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

# --- Input / output -----------------------------------------------------------
input_object_path <- file.path(
  project_dir,
  settings$project$input_object
)

output_dir <- file.path(
  project_dir,
  "celcomen_input_rna_top_genes"
)

output_h5ad <- file.path(
  output_dir,
  "celcomen_input_rna_top_genes.h5ad"
)

# --- Input Seurat expression --------------------------------------------------
assay <- settings$input_data$spatial_assay
diet_layer <- settings$input_data$diet_layer

if (!diet_layer %in% c("counts", "data")) {
  stop(
    "input_data.diet_layer must be either 'counts' or 'data'."
  )
}

# --- Analysis genes -----------------------------------------------------------
included_genes <- unique(
  unlist(settings$cohort$included_genes)
)

n_model_genes <- as.integer(
  settings$model_genes$n_model_genes
)

if (n_model_genes < length(included_genes)) {
  stop(
    "model_genes.n_model_genes must be at least as large as the number ",
    "of cohort.included_genes."
  )
}

python_executable <- NULL
python_env <- NULL

if (!is.null(settings$python$executable)) {
  python_executable <- normalizePath(
    path.expand(settings$python$executable),
    mustWork = TRUE
  )
} else if (!is.null(settings$export$python_env)) {
  python_env <- settings$export$python_env
} else {
  stop(
    "Specify either python.executable or export.python_env in settings.yaml."
  )
}

coordinate_scale <- "hires"

# =============================================================================
# SELECT PYTHON BEFORE USING THE R ANNDATA PACKAGE
# =============================================================================

if (!is.null(python_executable)) {
  reticulate::use_python(
    python_executable,
    required = TRUE
  )
} else {
  reticulate::use_condaenv(
    python_env,
    required = TRUE
  )
}

# =============================================================================
# LOAD AND VALIDATE THE SEURAT OBJECT
# =============================================================================

seurat_obj <- readRDS(input_object_path)

if (!assay %in% Assays(seurat_obj)) {
  stop(
    "Assay '",
    assay,
    "' was not found. Available assays: ",
    paste(Assays(seurat_obj), collapse = ", ")
  )
}

DefaultAssay(seurat_obj) <- assay

available_layers_before_diet <- SeuratObject::Layers(
  seurat_obj[[assay]]
)

if (!diet_layer %in% available_layers_before_diet) {
  stop(
    "Requested layer '",
    diet_layer,
    "' was not found in assay '",
    assay,
    "'. Available layers: ",
    paste(available_layers_before_diet, collapse = ", ")
  )
}

missing_from_assay <- setdiff(
  included_genes,
  rownames(seurat_obj[[assay]])
)

if (length(missing_from_assay) > 0) {
  stop(
    "Included analysis genes absent from assay: ",
    paste(missing_from_assay, collapse = ", ")
  )
}


# =============================================================================
# SELECT THE CELCOMEN ANALYSIS GENE SET
# =============================================================================
VariableFeatures(seurat_obj[[assay]]) <- character(0)

seurat_obj <- FindVariableFeatures(
  object = seurat_obj,
  assay = assay,
  nfeatures = n_model_genes
)

# Ordered from most variable to least variable.
top_variable <- VariableFeatures(
  seurat_obj[[assay]]
)

if (length(top_variable) < n_model_genes) {
  stop(
    "FindVariableFeatures returned only ",
    length(top_variable),
    " genes, fewer than the requested ",
    n_model_genes,
    "."
  )
}

included_to_add <- setdiff(
  included_genes,
  top_variable
)

# Replace the same number of lowest-ranked genes outside included_genes so the
# final model contains exactly n_model_genes genes.
if (length(included_to_add) > 0) {

  removable_genes <- rev(
    top_variable[!top_variable %in% included_genes]
  )

  if (length(removable_genes) < length(included_to_add)) {
    stop(
      "There are not enough non-included variable genes to preserve the ",
      "requested model-gene count."
    )
  }

  genes_to_remove <- removable_genes[
    seq_len(length(included_to_add))
  ]

  model_genes <- c(
    top_variable[!top_variable %in% genes_to_remove],
    included_to_add
  )

} else {

  model_genes <- top_variable

}

model_genes <- unique(model_genes)

stopifnot(
  length(model_genes) == n_model_genes,
  all(included_genes %in% model_genes)
)


# =============================================================================
# RETAIN ONLY THE REQUESTED ASSAY AND INPUT LAYER
# =============================================================================

seurat_obj <- DietSeurat(
  object = seurat_obj,
  assays = assay,
  layers = diet_layer,
  dimreducs = NULL,
  graphs = NULL,
  misc = FALSE
)

DefaultAssay(seurat_obj) <- assay

VariableFeatures(
  seurat_obj[[assay]]
) <- model_genes


# =============================================================================
# ENSURE THAT THE LOG-NORMALIZED DATA LAYER EXISTS
# =============================================================================

if (diet_layer == "counts") {

  seurat_obj <- NormalizeData(
    object = seurat_obj,
    assay = assay
  )

} else {

  available_layers_after_diet <- SeuratObject::Layers(
    seurat_obj[[assay]]
  )

  if (!"data" %in% available_layers_after_diet) {
    stop(
      "diet_layer was set to 'data', but the retained assay has no data layer."
    )
  }
}


# =============================================================================
# BUILD THE ANNDATA EXPRESSION MATRIX
# =============================================================================

cells <- colnames(seurat_obj)

expression <- SeuratObject::LayerData(
  object = seurat_obj,
  assay = assay,
  layer = "data"
)[model_genes, cells, drop = FALSE]

expression <- as(
  expression,
  "CsparseMatrix"
)

stopifnot(
  identical(rownames(expression), model_genes),
  identical(colnames(expression), cells)
)

X <- Matrix::t(expression)
rownames(X) <- cells
colnames(X) <- model_genes


# =============================================================================
# CELL METADATA: adata.obs
# =============================================================================

obs <- seurat_obj@meta.data[
  cells,
  ,
  drop = FALSE
]

obs$barcode <- rownames(obs)

obs <- obs[
  ,
  c("barcode", setdiff(colnames(obs), "barcode")),
  drop = FALSE
]


# =============================================================================
# SPATIAL COORDINATES: adata.obsm[["spatial"]]
# =============================================================================

image_names <- Images(seurat_obj)

if (length(image_names) == 0) {
  stop("No spatial image was found in the Seurat object.")
}

image_name <- image_names[1]

coordinates <- GetTissueCoordinates(
  object = seurat_obj[[image_name]],
  scale = coordinate_scale
)

if (!all(cells %in% rownames(coordinates))) {
  missing_coordinate_cells <- setdiff(
    cells,
    rownames(coordinates)
  )

  stop(
    "Spatial coordinates are missing for ",
    length(missing_coordinate_cells),
    " cells."
  )
}

coordinates <- coordinates[
  cells,
  ,
  drop = FALSE
]

if (all(c("imagecol", "imagerow") %in% colnames(coordinates))) {

  spatial <- as.matrix(
    data.frame(
      imagecol = coordinates$imagecol,
      imagerow = coordinates$imagerow,
      row.names = cells
    )
  )

} else if (all(c("x", "y") %in% colnames(coordinates))) {

  spatial <- as.matrix(
    data.frame(
      imagecol = coordinates$x,
      imagerow = coordinates$y,
      row.names = cells
    )
  )

} else {

  stop(
    "The spatial-coordinate table must contain either imagecol/imagerow ",
    "or x/y columns."
  )
}

storage.mode(spatial) <- "double"


# =============================================================================
# WRITE THE SINGLE CELCOMEN INPUT FILE
# =============================================================================

var <- data.frame(
  row.names = model_genes
)

adata <- anndata::AnnData(
  X = X,
  obs = obs,
  var = var,
  obsm = list(
    spatial = spatial
  )
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

adata$write_h5ad(
  output_h5ad,
  compression = "gzip"
)

cat("Preparation and export completed.\n")
cat("File:", output_h5ad, "\n")
cat("Cells:", length(cells), "\n")
cat("Genes:", length(model_genes), "\n")
cat("Matrix dimensions (cells x genes):", nrow(X), "x", ncol(X), "\n")
cat("Input assay:", assay, "\n")
cat("Retained input layer:", diet_layer, "\n")
cat("Exported expression layer: data\n")
cat("Included analysis genes:", paste(included_genes, collapse = ", "), "\n")
