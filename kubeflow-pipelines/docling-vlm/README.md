# Docling VLM Pipeline - Kubeflow Pipelines Demo

## Running the demo pipeline directly

Assuming you have a working Kubeflow Pipelines installation, download [vlm_pipeline.yaml](vlm_pipeline.yaml) and then upload it to your Kubeflow Pipelines UI or via the CLI. Then, start a run of that pipeline to see the conversion process happen.

## Modifying and running from source locally

### Clone repository, create venv, install dependencies

```bash
git clone https://github.com/opendatahub-io/odh-data-processing.git
cd odh-data-processing/kubeflow-pipelines/docling-vlm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Compile the kubeflow pipeline

```bash
python vlm_convert_pipeline.py
```

This generates a `vlm_pipeline.yaml` that you can now run in any Kubeflow Pipeline. This is a simple demo pipeline that downloads some PDFs, splits them into a handful of batches, and converts the batches across a cluster using the Docling VLM pipeline.

### Accelerator Device Selection

The pipeline now supports selecting between CPU and accelerator devices:

- **CPU only**: Set `docling_accelerator_device: "cpu"`
- **CUDA GPU**: Set `docling_accelerator_device: "cuda"`
- **Apple MPS**: Set `docling_accelerator_device: "mps"`
- **Auto-detection**: Set `docling_accelerator_device: "auto"` (default)

### Remote Model Configuration

The VLM pipeline also supports remote model endpoints. When `docling_remote_model_enabled: true`, the pipeline will use a remote model service instead of local VLM models.
