"""Docling example for PDF conversion with VLM pipeline"""

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode

source = "https://raw.githubusercontent.com/py-pdf/sample-files/refs/heads/main/026-latex-multicolumn/multicolumn.pdf"  # Path or URL to PDF

pipeline_options = VlmPipelineOptions()
pipeline_options.vlm_options = smoldocling_vlm_conversion_options

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
    }
)

result = converter.convert(source)
md = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

print(md)
