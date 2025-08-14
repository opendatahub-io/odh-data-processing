from typing import List

from kfp import dsl

PYTHON_BASE_IMAGE = "python:3.11"


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["gitpython"],
)
def import_test_pdfs(
    output_path: dsl.Output[dsl.Artifact],
):
    """
    Import test PDF files from the Docling GitHub repository.

    Args:
        output_path: Path to the output directory for the PDF files.
    """
    from pathlib import Path
    import shutil
    from git import Repo  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402

    output_path_p = Path(output_path.path)

    docling_github_repo = "https://github.com/docling-project/docling/"
    full_repo_path = output_path_p / "docling"
    Repo.clone_from(docling_github_repo, full_repo_path, branch="v2.43.0")

    pdfs_path = full_repo_path / "tests" / "data" / "pdf"
    shutil.copytree(pdfs_path, output_path_p, dirs_exist_ok=True)

    shutil.rmtree(full_repo_path)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
)
def create_pdf_splits(
    input_path: dsl.Input[dsl.Artifact],
    num_splits: int,
) -> List[List[str]]:
    """
    Create a list of PDF splits.

    Args:
        input_path: Path to the input directory containing PDF files.
        num_splits: Number of splits to create.
    """
    from pathlib import Path

    input_path_p = Path(input_path.path)

    all_pdfs = [path.name for path in input_path_p.glob("*.pdf")]
    return [all_pdfs[i::num_splits] for i in range(num_splits)]


@dsl.component(
    base_image="quay.io/fabianofranz/docling:v2.43.0",
)
def download_docling_models(
    output_path: dsl.Output[dsl.Artifact],
):
    """
    Download Docling models.

    Args:
        output_path: Path to the output directory for Docling models.
    """
    from pathlib import Path
    from docling.utils.model_downloader import download_models

    output_path_p = Path(output_path.path)

    output_path_p.mkdir(parents=True, exist_ok=True)

    download_models(
        output_dir=output_path_p,
        progress=True,
        with_layout=True,
        with_tableformer=True,
        with_easyocr=True,
    )


@dsl.component(
    base_image="quay.io/fabianofranz/docling:v2.43.0",
)
def docling_convert(
    input_path: dsl.Input[dsl.Artifact],
    pdf_split: List[str],
    pdf_backend: str,
    artifacts_path: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
):
    """
    Convert a list of PDF files to JSON and Markdown using Docling.

    Args:
        input_path: Path to the input directory containing PDF files.
        pdf_split: List of PDF file names to process.
        pdf_backend: Backend to use for PDF processing.
        artifacts_path: Path to the directory containing Docling models.
        output_path: Path to the output directory for JSON and Markdown files.
    """
    import os
    from importlib import import_module
    from pathlib import Path

    from docling_core.types.doc.base import ImageRefMode  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
    from docling.datamodel.base_models import InputFormat  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
    from docling.datamodel.pipeline_options import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
        PdfPipelineOptions,
        PdfBackend,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402

    allowed_backends = {e.value for e in PdfBackend}
    if pdf_backend not in allowed_backends:
        raise ValueError(
            f"Invalid pdf_backend: {pdf_backend}. Must be one of {sorted(allowed_backends)}"
        )

    input_path_p = Path(input_path.path)
    artifacts_path_p = Path(artifacts_path.path)
    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    input_pdfs = [input_path_p / name for name in pdf_split]
    print(f"docling-convert: starting with backend='{pdf_backend}', files={len(input_pdfs)}", flush=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = artifacts_path_p
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_page_images = True

    backend_to_impl = {
        PdfBackend.PYPDFIUM2.value: (
            "docling.backend.pypdfium2_backend",
            "PyPdfiumDocumentBackend",
        ),
        PdfBackend.DLPARSE_V1.value: (
            "docling.backend.docling_parse_backend",
            "DoclingParseDocumentBackend",
        ),
        PdfBackend.DLPARSE_V2.value: (
            "docling.backend.docling_parse_v2_backend",
            "DoclingParseV2DocumentBackend",
        ),
        PdfBackend.DLPARSE_V4.value: (
            "docling.backend.docling_parse_v4_backend",
            "DoclingParseV4DocumentBackend",
        ),
    }

    module_name, class_name = backend_to_impl[pdf_backend]
    backend_class = getattr(import_module(module_name), class_name)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=backend_class,
            )
        }
    )

    easyocr_path_p = artifacts_path_p / "EasyOcr"
    os.environ["MODULE_PATH"] = str(easyocr_path_p)
    os.environ["EASYOCR_MODULE_PATH"] = str(easyocr_path_p)

    results = doc_converter.convert_all(input_pdfs, raises_on_error=True)

    for result in results:
        doc_filename = result.input.file.stem

        output_json_path = output_path_p / f"{doc_filename}.json"
        print(f"docling-convert: saving {output_json_path}", flush=True)
        result.document.save_as_json(output_json_path, image_mode=ImageRefMode.PLACEHOLDER)

        output_md_path = output_path_p / f"{doc_filename}.md"
        print(f"docling-convert: saving {output_md_path}", flush=True)
        result.document.save_as_markdown(output_md_path, image_mode=ImageRefMode.PLACEHOLDER)

    print("docling-convert: done", flush=True)
