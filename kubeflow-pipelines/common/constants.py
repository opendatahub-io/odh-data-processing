# Base container images used across all Docling Kubeflow Pipelines
PYTHON_BASE_IMAGE = "registry.access.redhat.com/ubi9/python-311:9.6-1755074620"
DOCLING_BASE_IMAGE = "quay.io/fabianofranz/docling-ubi9:2.54.0"

# Model types for download_docling_models component
MODEL_TYPE_STANDARD = "standard"
MODEL_TYPE_VLM = "vlm"
MODEL_TYPE_VLM_REMOTE = "vlm-remote"
