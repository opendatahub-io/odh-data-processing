"""Docling example for PDF conversion with image description and classification"""

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
from docling_core.types.doc import (
    ImageRefMode,
    PictureClassificationData,
)

source = "https://raw.githubusercontent.com//docling-project/docling/refs/heads/main/tests/data/pdf/picture_classification.pdf"  # Path or URL to PDF

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions()
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.generate_page_images = True
pipeline_options.do_picture_classification = True
pipeline_options.do_picture_description = True
pipeline_options.do_formula_enrichment = False
pipeline_options.do_code_enrichment = False
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

for picture in result.document.pictures:
    for annotation in picture.annotations:
        print(annotation.provenance)
        if isinstance(annotation, PictureClassificationData):
            for predicted_class in annotation.predicted_classes:
                print(
                    f"{predicted_class.class_name} with {predicted_class.confidence} confidence"
                )
