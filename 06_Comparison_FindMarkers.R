library(stringr)
library(Seurat)
library(biomaRt)

# ============================================================================
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

combined_rds_path <- file.path(project_dir, "seurat_simcomen_combined.rds")

default_assay       <- "SIMCOMEN"
annotation_column   <- settings$cohort$annotation_column
perturbation_column <- "Perturbation"
pert_pop_column     <- "Pert_Pop"
Ident_to_use        <- pert_pop_column

comparison_pairs <- settings$comparisons
for (i in seq_along(comparison_pairs)) {
  assign(paste0("Ident_pair_", i), unlist(comparison_pairs[[i]]))
}

min_cells_per_ident <- as.integer(settings$differential_expression$min_cells_per_ident)
p_val_adj_threshold <- 0.05
Check_only_protein_coding_genes <- settings$differential_expression$check_only_protein_coding_genes
species_study <- settings$differential_expression$species_study

main_dir <- file.path(project_dir, "Simcomen_diff_results", "DEGs")
protein_gene_dir <- path.expand(settings$project$protein_gene_dir)

deg_subdir_all <- "All_genes/"
deg_subdir_rnk <- "rnks/"
deg_subdir_sig <- "Significant_DEGs/"

seurat_obj <- readRDS(combined_rds_path)

seurat_obj[[pert_pop_column]] <- paste(
  seurat_obj[[annotation_column]][, 1],
  seurat_obj[[perturbation_column]][, 1],
  sep = "_"
)
saveRDS(seurat_obj, file = combined_rds_path)


DefaultAssay(seurat_obj) <- default_assay
Idents(seurat_obj) <- Ident_to_use


################################################################################################
### Read protein coding genes
################################################################################################
if (Check_only_protein_coding_genes) {
  file_name  <- paste0(species_study, "_protein_coding_symbols_uniprot.txt")
  file_path  <- file.path(protein_gene_dir, file_name)
  prot_genes <- readLines(file_path)
} else {
  prot_genes <- character(0)   # not used unless Check_only_protein_coding_genes is TRUE
}
################################################################################################
### Folder preparation
################################################################################################

dir_save_total_markers          <- file.path(main_dir, deg_subdir_all)
dir_save_rnk_files 		<- file.path(main_dir, deg_subdir_rnk)
dir_save_DEG_markers 		<- file.path(main_dir, deg_subdir_sig)

dir_vars <- c("dir_save_total_markers", "dir_save_rnk_files", "dir_save_DEG_markers")

for (dv in dir_vars) {
  dir_path <- path.expand( get(dv) )

  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    message("Created directory: ", dir_path)
  } else {
    message("Directory already exists: ", dir_path)
  }
}

################################################################################################
### FindMarkers and save csv
################################################################################################

Idents(seurat_obj) <- Ident_to_use
Ident_pair_objects <- ls(pattern = "^Ident_pair_\\d+$")

seurat_features <- rownames(seurat_obj)
int_seu_prot_genes <- intersect(prot_genes , seurat_features)

for (pair_name in Ident_pair_objects) {
  idents <- get(pair_name)
  ident_1st <- idents[1]
  ident_2nd <- idents[2]
  markers_name <- paste0("markers_", ident_1st, "_vs_", ident_2nd)

  cells1 <- WhichCells(seurat_obj, idents = ident_1st)
  cells2 <- WhichCells(seurat_obj, idents = ident_2nd)

  if (length(cells1) < min_cells_per_ident || length(cells2) < min_cells_per_ident) {
    message(sprintf("Skipping %s vs %s: %d vs %d cells (< %d).",
                    ident_1st, ident_2nd, length(cells1), length(cells2), min_cells_per_ident))
    next
  }

  message("Finding markers: ", ident_1st, " vs ", ident_2nd)
  markers <- FindMarkers(
    object = seurat_obj,
    ident.1 = ident_1st,
    ident.2 = ident_2nd,
    features = if (Check_only_protein_coding_genes) int_seu_prot_genes else NULL,
    min.cells.group = min_cells_per_ident,
    logfc.threshold = 0, recorrect_umi=FALSE
  )

  assign(markers_name, markers, envir = .GlobalEnv)
  rm("markers", "ident_1st", "ident_2nd")
}

rm("markers_name", "markers", "ident_1st", "ident_2nd", "markers_reg_objects")

markers_reg_objects <- ls(pattern = "markers_.*$")
if (!dir.exists(dir_save_total_markers)) {dir.create(dir_save_total_markers, recursive = TRUE)}

for (obj_name in markers_reg_objects) {
  df <- get(obj_name, envir = .GlobalEnv)
  write.csv(df, file = paste0(dir_save_total_markers, obj_name, ".csv"))}


################################################################################################
### Make and save rnk files
################################################################################################

if (!dir.exists(dir_save_rnk_files)) {dir.create(dir_save_rnk_files, recursive = TRUE)}

for (df_markers in markers_reg_objects) {

  df <- get(df_markers)

  GSEA_table <- df[, c("avg_log2FC", "p_val")]

  GSEA_table$sign <- sign(GSEA_table$avg_log2FC)

  GSEA_table <- GSEA_table[, c("p_val", "sign")]

  GSEA_table$preranked <- -10 * log10(GSEA_table[, "p_val"]) * GSEA_table[, "sign"]

  smallest_value <- min(GSEA_table$preranked[is.finite(GSEA_table$preranked)], na.rm = TRUE)
  highest_value  <- max(GSEA_table$preranked[is.finite(GSEA_table$preranked)], na.rm = TRUE)

  GSEA_table$preranked[GSEA_table$preranked == Inf] <- highest_value + 0.000000000000000001
  GSEA_table$preranked[GSEA_table$preranked == -Inf] <- smallest_value - 0.000000000000000001

  GSEA_table <- GSEA_table[order(GSEA_table$preranked, decreasing = TRUE), ]

  GSEA_table <- GSEA_table[, "preranked", drop = FALSE]

  csv_path <- paste0(dir_save_rnk_files, df_markers, ".rnk")
  write.table(GSEA_table, file = csv_path, sep = "\t", quote = FALSE, col.names = FALSE)
  
  cat("Wrote", df_markers, "to", csv_path, "\n")
}


################################################################################################
### Make and FindMarkers DEG csv
################################################################################################

if (!dir.exists(dir_save_DEG_markers)) {dir.create(dir_save_DEG_markers, recursive = TRUE)}

for (m in markers_reg_objects) {
  df <- get(m, envir = .GlobalEnv)
  
  up_name   <- sub("_vs_", "_up_vs_",   m, fixed = TRUE)
  down_name <- sub("_vs_", "_down_vs_", m, fixed = TRUE)
  
  assign(up_name,
         subset(df, avg_log2FC > 0  & p_val_adj <= p_val_adj_threshold),
         envir = .GlobalEnv)
  assign(down_name,
         subset(df, avg_log2FC < 0  & p_val_adj <= p_val_adj_threshold),
         envir = .GlobalEnv)
  
  rm(list = m, envir = .GlobalEnv)
}

markers_diff_reg_objects <- ls(pattern = "_(up|down)_vs_.*$")

for (obj_dif_name in markers_diff_reg_objects) {
  dif <- get(obj_dif_name, envir = .GlobalEnv)
  write.csv(dif, file = paste0(dir_save_DEG_markers, obj_dif_name, ".csv"))}


