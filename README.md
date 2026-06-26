# Seurat v5 - Spatial in-silico perturbation with CELCOMEN / SIMCOMEN and trajectory inference

## Main goal of the pipeline
This pipeline takes a Seurat v5 spatial object as input and runs six scripts, organized by a Snakefile, that:
1. Prepare the object and export it to a single AnnData `.h5ad` (select the model gene set, normalize, keep the spatial coordinates).
2. Train CELCOMEN to learn the gene-gene interaction parameters from the spatial neighbour graph.
3. Run SIMCOMEN to simulate an in-silico perturbation of a target gene in selected populations, producing perturbed and reference cell states.
4. Return the results to Seurat, project the perturbed cells into the reference PCA and UMAP, and draw the displacement and stream plots (trajectory inference).
5. Combine the reference and perturbed objects into one object labelled Before and After.
6. Run FindMarkers between the Before and After states for each population and export the differential-expression tables and ranked lists.

Every parameter is read from a single `settings.yaml`. Each script self-computes its paths from `project_dir`, so the only thing you set is that one folder plus the biological choices.

## Essential packages and versions
The versions below are a known-good reference. Pin to whatever your environment was validated against.

### R environment (tested on 4.5.1)
Required
- Seurat >= 5.0.0
- reticulate (bridge to the Python env for the AnnData export and the stream-plot rendering)
- anndata (R package; writes the `.h5ad` through reticulate)
- zellkonverter
- SingleCellExperiment
- yaml
- Matrix
- ggplot2 3.5.2
- stringr 1.5.1
- biomaRt 2.60.0
- grid (ships with base R)

### Python environment (tested on 3.12)
Required
- celcomen (provides both CELCOMEN and SIMCOMEN, and pulls in its own dependencies)
- torch 2.8.0
- anndata 0.10.8
- numpy 1.26.4
- pandas 2.3.2
- pyyaml 6.0.2
- scipy 1.14.1
- scikit-learn 1.7.2
- matplotlib 3.9.2 (used by the R stream-plot step through reticulate)

The Python environment is shared. The `.py` scripts use it directly, and the R scripts (1 and 4) reach into the same environment through reticulate, so it must also contain `anndata` and `matplotlib`. Its name is set in `settings.yaml` under `export.python_env`.

## Installation

### 1) Clone the repository
On Ubuntu:
```bash
sudo apt update
sudo apt install -y git r-base r-base-dev   # if needed

git clone git@github.com:miltos91/spatial_celcomen_simcomen_trajectory.git
cd spatial_celcomen_simcomen_trajectory
```

### 2) Build the Python environment
The environment name must match `export.python_env` in `settings.yaml`:
```bash
conda create -n celcomen_env python=3.12 -y
conda activate celcomen_env

pip install \
  torch==2.8.0 \
  anndata==0.10.8 \
  numpy==1.26.4 \
  pandas==2.3.2 \
  pyyaml==6.0.2 \
  scipy==1.14.1 \
  scikit-learn==1.7.2 \
  matplotlib==3.9.2

pip install celcomen
```

### 3) Install the R libraries
```bash
Rscript -e 'pkgs <- c("Seurat","reticulate","anndata","zellkonverter","SingleCellExperiment","yaml","Matrix","ggplot2","stringr","biomaRt"); \
miss <- setdiff(pkgs, rownames(installed.packages())); \
if(length(miss)) install.packages(miss, repos="https://cloud.r-project.org")'
```

```bash
Rscript -e 'if(!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); \
BiocManager::install(c("zellkonverter","SingleCellExperiment","biomaRt"))'
```

## How to use

### 1) Configure settings.yaml
Edit `settings.yaml`. Every script reads it, so it is the single place you set things:
- project: `project_dir` (the one main folder; everything else is created inside it), `input_object` (the raw Seurat `.rds` name), `protein_gene_dir` (only needed when `check_only_protein_coding_genes` is true).
- input_data: `spatial_assay`, `diet_layer`.
- cohort: `sample_column`, `mutant_value`, `annotation_column`, `target_gene`, `included_genes`, `target_populations`, `reference_populations`.
- display: labels, colours, order, and reference cell size for the plots.
- model_genes: `n_model_genes`.
- celcomen and simcomen: neighbours, epochs, learning rate, scalar, seed for each step.
- export: `python_env` (the conda env used by the Python scripts and by reticulate).
- plots: `label_fontsize`, `label_stroke_linewidth`.
- differential_expression: `min_cells_per_ident`, `species_study`, `check_only_protein_coding_genes`.
- comparisons: the Before and After pairs passed to FindMarkers.

Structural names (sub-folders, intermediate object names, output locations) are fixed defaults inside the scripts and created automatically, so they are not in `settings.yaml`.

### 2) Run the full pipeline
From the workflow directory (where the Snakefile and settings.yaml are):
```bash
snakemake -j1
```
The R steps run with the system `Rscript`; the Python steps run inside the conda env named in `export.python_env` (the Snakefile calls `conda run -n <env>`). The same env is used by reticulate inside scripts 1 and 4.

Outputs (created under `project_dir`):
- celcomen_input_rna_top_genes/ - the AnnData `.h5ad` and the CELCOMEN model, interaction parameters, and training loss
- celcomen_input_rna_top_genes/celcomen_output_rna_top_genes/simcomen_perturbed/ and .../simcomen_reference/ - the SIMCOMEN result and reference `.h5ad` files
- simcomen_seurat_reference.rds, simcomen_seurat_perturbed.rds, seurat_simcomen_combined.rds - the Seurat objects
- Simcomen_diff_results/graphs/ - the stream plots
- Simcomen_diff_results/DEGs/ - the FindMarkers tables and ranked lists (All_genes/, rnks/, Significant_DEGs/)

## Complete steps of the pipeline

### 1) 1_R_prep_export.R - prepare and export to AnnData
- Load packages and read settings
- Read the Seurat object and set the spatial assay
- Select the model gene set, forcing the included genes in
- Normalize and keep the chosen layer
- Build an AnnData with expression, metadata, and spatial coordinates
- Write a single `.h5ad` to the CELCOMEN input folder

### 2) 2_Celcomen.py - train CELCOMEN
- Initialize packages and read settings
- Load and validate the AnnData (sample column and spatial coordinates present)
- Build the spatial neighbour graph and the CELCOMEN model
- Train and learn the gene-gene interaction parameters
- Save the model, the interaction parameters, and the training loss

### 3) 03_Simcomen.py - run SIMCOMEN (in-silico perturbation)
- Initialize packages and read settings
- Keep the perturbed sample and the selected target populations
- Set the target gene to zero and run SIMCOMEN to the new steady state
- Store the pre, post, and delta expression layers
- Save the perturbed results and a reference (all-cells) `.h5ad`

### 4) 04_Trajectory_Inference.R - return to Seurat and stream plots
- Load packages and read settings
- Read the SIMCOMEN `.h5ad` files back into Seurat
- Build the reference PCA and UMAP and project the perturbed cells into them
- Compute the displacement field between reference and perturbed positions
- Draw the stream and background plots through reticulate and matplotlib
- Save the reference and perturbed Seurat objects and the plots

### 5) 05_Combine_objects.R - combine the objects
- Load packages and read settings
- Read the reference and perturbed objects
- Prefix the perturbed cell names and label Before and After
- Rename to a common assay and merge into one object
- Carry the shared PCA across and save the combined object

### 6) 06_Comparison_FindMarkers.R - differential expression
- Load packages and read settings
- Read the combined object and build the comparison identity
- Run FindMarkers for each Before and After pair
- Keep the genes passing the adjusted p-value threshold
- Save the up- and down-regulated tables and the ranked lists
