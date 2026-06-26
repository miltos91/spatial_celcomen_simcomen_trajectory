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

input_ref_rds       <- file.path(project_dir, "simcomen_seurat_reference.rds")
input_mut_rds       <- file.path(project_dir, "simcomen_seurat_perturbed.rds")
output_combined_rds <- file.path(project_dir, "seurat_simcomen_combined.rds")

mutant_cell_prefix  <- settings$cohort$mutant_value
perturbation_before <- "Before"
perturbation_after  <- "After"

ref_source_assay <- "reference"
mut_source_assay <- "perturbed"
combined_assay   <- "SIMCOMEN"
combined_project <- "SIMCOMEN_combined"

pca_reduction     <- "pca"
mut_pca_reduction <- "pca.ref"

#############################################################################
# 1. LOAD OBJECTS
#############################################################################

simcomen_seurat_ref <- readRDS(
    input_ref_rds
)

simcomen_seurat_mut <- readRDS(
    input_mut_rds
)

#############################################################################
# 2. PREFIX MUTANT/POST-PERTURBATION CELL NAMES
#############################################################################

simcomen_seurat_mut <- RenameCells(
    simcomen_seurat_mut,
    add.cell.id = mutant_cell_prefix
)

#############################################################################
# 3. ADD PERTURBATION METADATA
#############################################################################

simcomen_seurat_ref$Perturbation <- perturbation_before
simcomen_seurat_mut$Perturbation <- perturbation_after

#############################################################################
# 4. GIVE BOTH OBJECTS THE SAME ASSAY NAME
#############################################################################

simcomen_seurat_ref <- RenameAssays(
    simcomen_seurat_ref,
    assay.name = ref_source_assay,
    new.assay.name = combined_assay
)

simcomen_seurat_mut <- RenameAssays(
    simcomen_seurat_mut,
    assay.name = mut_source_assay,
    new.assay.name = combined_assay
)

DefaultAssay(simcomen_seurat_ref) <- combined_assay
DefaultAssay(simcomen_seurat_mut) <- combined_assay

#############################################################################
# 5. VERIFY THAT THE TWO PCA MATRICES ARE COMPATIBLE
#############################################################################

reference_pca <- Embeddings(
    simcomen_seurat_ref,
    reduction = pca_reduction
)

mutant_pca <- Embeddings(
    simcomen_seurat_mut,
    reduction = mut_pca_reduction
)

stopifnot(
    ncol(reference_pca) == ncol(mutant_pca)
)

colnames(mutant_pca) <- colnames(reference_pca)

#############################################################################
# 6. SAVE THE SCALE.DATA MATRICES BEFORE MERGING
#############################################################################

reference_scaled <- LayerData(
    simcomen_seurat_ref,
    assay = combined_assay,
    layer = "scale.data"
)

mutant_scaled <- LayerData(
    simcomen_seurat_mut,
    assay = combined_assay,
    layer = "scale.data"
)

stopifnot(
    identical(
        rownames(reference_scaled),
        rownames(mutant_scaled)
    )
)

combined_scaled <- cbind(
    reference_scaled,
    mutant_scaled
)

#############################################################################
# 7. MERGE EXPRESSION DATA AND METADATA
#############################################################################

simcomen_seurat_combined <- merge(
    x = simcomen_seurat_ref,
    y = simcomen_seurat_mut,
    collapse = TRUE,
    merge.data = TRUE,
    merge.dr = FALSE,
    project = combined_project
)

DefaultAssay(simcomen_seurat_combined) <- combined_assay

#############################################################################
# 8. RESTORE THE COMBINED SCALE.DATA LAYER
#############################################################################

stopifnot(
    setequal(
        colnames(combined_scaled),
        Cells(simcomen_seurat_combined)
    )
)

combined_scaled <- combined_scaled[
    ,
    Cells(simcomen_seurat_combined),
    drop = FALSE
]

LayerData(
    simcomen_seurat_combined,
    assay = combined_assay,
    layer = "scale.data"
) <- combined_scaled

#############################################################################
# 9. COMBINE THE REFERENCE AND PROJECTED PCA COORDINATES
#############################################################################

combined_pca <- rbind(
    reference_pca,
    mutant_pca
)

stopifnot(
    setequal(
        rownames(combined_pca),
        Cells(simcomen_seurat_combined)
    )
)

combined_pca <- combined_pca[
    Cells(simcomen_seurat_combined),
    ,
    drop = FALSE
]

#############################################################################
# 10. CREATE ONE COMMON PCA REDUCTION
#############################################################################

reference_loadings <- Loadings(
    simcomen_seurat_ref[[pca_reduction]]
)

reference_stdev <- Stdev(
    simcomen_seurat_ref,
    reduction = pca_reduction
)

simcomen_seurat_combined[[pca_reduction]] <- CreateDimReducObject(
    embeddings = combined_pca,
    loadings = reference_loadings,
    stdev = reference_stdev,
    assay = combined_assay,
    key = "PC_"
)

#############################################################################
# 11. SET PERTURBATION ORDER
#############################################################################

simcomen_seurat_combined$Perturbation <- factor(
    simcomen_seurat_combined$Perturbation,
    levels = c(
        perturbation_before,
        perturbation_after
    )
)

#############################################################################
# 12. VALIDATE THE RESULT
#############################################################################

stopifnot(
    nrow(Embeddings(simcomen_seurat_combined, pca_reduction)) ==
        ncol(simcomen_seurat_combined),

    identical(
        rownames(Embeddings(simcomen_seurat_combined, pca_reduction)),
        Cells(simcomen_seurat_combined)
    ),

    identical(
        colnames(LayerData(
            simcomen_seurat_combined,
            assay = combined_assay,
            layer = "scale.data"
        )),
        Cells(simcomen_seurat_combined)
    )
)

print(simcomen_seurat_combined)

dim(
    Embeddings(
        simcomen_seurat_combined,
        reduction = pca_reduction
    )
)

Layers(
    simcomen_seurat_combined[[combined_assay]]
)

#############################################################################
# 13. SAVE
#############################################################################

saveRDS(
    simcomen_seurat_combined,
    file = output_combined_rds
)
