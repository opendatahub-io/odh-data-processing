"""
Common components for Docling Kubeflow Pipelines.

This module contains shared components that are used across different
Docling pipeline implementations (standard and VLM).
"""

# Import all common components to make them easily accessible
from .components import (
    import_pdfs,
    create_pdf_splits,
    download_docling_models,
)

from .constants import (
    PYTHON_BASE_IMAGE,
    DOCLING_BASE_IMAGE,
    MODEL_TYPE_STANDARD,
    MODEL_TYPE_VLM,
)

__all__ = [
    "import_pdfs",
    "create_pdf_splits", 
    "download_docling_models",
    "PYTHON_BASE_IMAGE",
    "DOCLING_BASE_IMAGE", 
    "MODEL_TYPE_STANDARD",
    "MODEL_TYPE_VLM",
]