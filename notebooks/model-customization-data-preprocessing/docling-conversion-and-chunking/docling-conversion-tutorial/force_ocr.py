"""Docling example for PDF conversion with OCR"""

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling_core.types.doc import ImageRefMode

source = "https://raw.githubusercontent.com/py-pdf/sample-files/refs/heads/main/026-latex-multicolumn/multicolumn.pdf"  # Path or URL to PDF

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions()
pipeline_options.ocr_options.force_full_page_ocr = True
pipeline_options.generate_picture_images = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.AUTO
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=DoclingParseV4DocumentBackend,
        )
    }
)

result = converter.convert(source)
md = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

print(md)
