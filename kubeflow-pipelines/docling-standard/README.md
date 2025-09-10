# Docling and Kubeflow Pipelines Demo

## Running the demo pipeline directly

Assuming you have a working Kubeflow Pipelines installation, download [docling_convert_pipeline_compiled.yaml](docling_convert_pipeline_compiled.yaml) and the upload it to your Kubeflow Pipelines UI or via the CLI. Then, start a run of that pipeline to see the conversion process happen.

## Modifying and running from source locally

### Clone repository, create venv, install dependencies

```
git clone https://github.com/fabianofranz/docling-kfp-demo
cd docling-kfp-demo
source venv/bin/activate
pip install -r requirements.txt
```

### Compile the kubeflow pipeline

```
python docling_convert_pipeline.py
```

This generates a `docling_convert_pipeline_compiled.yaml` that you can now run in any Kubeflow Pipeline. This is a simple demo pipeline that downloads some PDFs, splits them into a handful of batches, and converts the batches across a cluster.
