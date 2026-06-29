from pathlib import Path


# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

WORKFLOW_DIR = Path(workflow.basedir).resolve()
SETTINGS_FILE = WORKFLOW_DIR / "settings.yaml"

configfile: str(SETTINGS_FILE)

PROJECT_DIR = Path(config["project"]["project_dir"]).expanduser().resolve()
INPUT_OBJECT = PROJECT_DIR / config["project"]["input_object"]
PYTHON_ENV = config["export"]["python_env"]

INPUT_DIR = PROJECT_DIR / "celcomen_input_rna_top_genes"
CELCOMEN_DIR = INPUT_DIR / "celcomen_output_rna_top_genes"
SIMCOMEN_RESULTS_DIR = CELCOMEN_DIR / "simcomen_perturbed"
SHAM_RESULTS_DIR = CELCOMEN_DIR / "simcomen_sham"
GRAPH_DIR = PROJECT_DIR / "Simcomen_diff_results" / "graphs"
DEG_DIR = PROJECT_DIR / "Simcomen_diff_results" / "DEGs"

INPUT_H5AD = INPUT_DIR / "celcomen_input_rna_top_genes.h5ad"
CELCOMEN_MODEL = CELCOMEN_DIR / "celcomen_model_rna_top_genes.pt"
CELCOMEN_PARAMETERS = CELCOMEN_DIR / "simcomen_interaction_parameters_rna.pt"
CELCOMEN_LOSS = CELCOMEN_DIR / "training_loss.csv"

MUTANT_H5AD = SIMCOMEN_RESULTS_DIR / "simcomen_perturbed_results.h5ad"
SIMCOMEN_LOSS = SIMCOMEN_RESULTS_DIR / "simcomen_training_loss.csv"

SHAM_H5AD = SHAM_RESULTS_DIR / "simcomen_sham_results.h5ad"
SHAM_LOSS = SHAM_RESULTS_DIR / "simcomen_training_loss_sham.csv"

REFERENCE_RDS = PROJECT_DIR / "simcomen_seurat_reference.rds"
PERTURBED_RDS = PROJECT_DIR / "simcomen_seurat_perturbed.rds"
COMBINED_RDS = PROJECT_DIR / "seurat_simcomen_combined.rds"

PREVIEW_PNG = GRAPH_DIR / "streamplot_with_background.png"
TRANSPARENT_PNG = GRAPH_DIR / "streamplot_with_background_transparent.png"
DEG_COMPLETE = DEG_DIR / ".findmarkers_complete"


# =============================================================================
# FINAL TARGETS
# =============================================================================

rule all:
    input:
        str(COMBINED_RDS),
        str(PREVIEW_PNG),
        str(TRANSPARENT_PNG),
        str(DEG_COMPLETE),


# =============================================================================
# PIPELINE
# =============================================================================

rule prepare_and_export_anndata:
    input:
        settings=str(SETTINGS_FILE),
        object=str(INPUT_OBJECT),
        script=str(WORKFLOW_DIR / "01_R_prep_export.R"),
    output:
        str(INPUT_H5AD),
    shell:
        'Rscript "{input.script}" --settings "{input.settings}"'


rule train_celcomen:
    input:
        settings=str(SETTINGS_FILE),
        h5ad=str(INPUT_H5AD),
        script=str(WORKFLOW_DIR / "02_Celcomen.py"),
    output:
        model=str(CELCOMEN_MODEL),
        parameters=str(CELCOMEN_PARAMETERS),
        loss=str(CELCOMEN_LOSS),
    params:
        env=PYTHON_ENV,
    shell:
        'conda run --no-capture-output -n "{params.env}" '
        'python "{input.script}" --settings "{input.settings}"'


rule run_simcomen:
    input:
        settings=str(SETTINGS_FILE),
        h5ad=str(INPUT_H5AD),
        parameters=str(CELCOMEN_PARAMETERS),
        script=str(WORKFLOW_DIR / "03_Simcomen.py"),
    output:
        mutant_h5ad=str(MUTANT_H5AD),
        loss=str(SIMCOMEN_LOSS),
    params:
        env=PYTHON_ENV,
    shell:
        'conda run --no-capture-output -n "{params.env}" '
        'python "{input.script}" --settings "{input.settings}"'


rule run_sham:
    input:
        settings=str(SETTINGS_FILE),
        h5ad=str(INPUT_H5AD),
        parameters=str(CELCOMEN_PARAMETERS),
        script=str(WORKFLOW_DIR / "03B_Simcomen.py"),
    output:
        sham_h5ad=str(SHAM_H5AD),
        loss=str(SHAM_LOSS),
    params:
        env=PYTHON_ENV,
    shell:
        'conda run --no-capture-output -n "{params.env}" '
        'python "{input.script}" --settings "{input.settings}"'


rule return_to_seurat:
    input:
        settings=str(SETTINGS_FILE),
        mutant_h5ad=str(MUTANT_H5AD),
        reference_h5ad=str(SHAM_H5AD),
        script=str(WORKFLOW_DIR / "04_Trajectory_Inference.R"),
    output:
        reference_rds=str(REFERENCE_RDS),
        perturbed_rds=str(PERTURBED_RDS),
        preview_png=str(PREVIEW_PNG),
        transparent_png=str(TRANSPARENT_PNG),
    shell:
        'Rscript "{input.script}" --settings "{input.settings}"'


rule combine_seurat:
    input:
        settings=str(SETTINGS_FILE),
        reference_rds=str(REFERENCE_RDS),
        perturbed_rds=str(PERTURBED_RDS),
        script=str(WORKFLOW_DIR / "05_Combine_objects.R"),
    output:
        str(COMBINED_RDS),
    shell:
        'Rscript "{input.script}" --settings "{input.settings}"'


rule differential_expression:
    input:
        settings=str(SETTINGS_FILE),
        combined_rds=str(COMBINED_RDS),
        script=str(WORKFLOW_DIR / "06_Comparison_FindMarkers.R"),
    output:
        touch(str(DEG_COMPLETE)),
    shell:
        'Rscript "{input.script}" --settings "{input.settings}"'
