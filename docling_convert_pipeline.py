# SPDX-License-Identifier: Apache-2.0

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
    from pathlib import Path
    import shutil
    from git import Repo

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
    import os
    from importlib import import_module
    from pathlib import Path

    from docling_core.types.doc.base import ImageRefMode
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PdfBackend,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

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


@dsl.pipeline()
def convert_pipeline(num_splits: int = 3, pdf_backend: str = "dlparse_v4"):
    importer = import_test_pdfs()

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=num_splits,
    )

    artifacts = download_docling_models()

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        docling_convert(
            input_path=importer.outputs["output_path"],
            pdf_split=pdf_split,
            pdf_backend=pdf_backend,
            artifacts_path=artifacts.outputs["output_path"],
        )


if __name__ == "__main__":
    import kfp

    output_yaml = "docling_pipeline.yaml"
    kfp.compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"Docling pipeline compiled to {output_yaml}")
