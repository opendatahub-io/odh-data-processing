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
    import os
    import shutil
    from git import Repo

    output_dir = output_path.path

    docling_github_repo = "https://github.com/docling-project/docling/"
    full_repo_path = os.path.join(output_dir, "docling")
    Repo.clone_from(docling_github_repo, full_repo_path, branch="v2.43.0")

    pdfs_path = os.path.join(full_repo_path, "tests", "data", "pdf")
    shutil.copytree(pdfs_path, output_dir, dirs_exist_ok=True)

    shutil.rmtree(full_repo_path)

@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
)
def create_pdf_splits(
    input_path: dsl.Input[dsl.Artifact],
    num_splits: int,
) -> List[List[str]]:
    import pathlib

    all_pdfs = [path.name for path in pathlib.Path(input_path.path).glob("*.pdf")]
    return [all_pdfs[i::num_splits] for i in range(num_splits)]

@dsl.component(
    base_image="quay.io/fabianofranz/docling:v2.43.0",
)
def docling_convert(
    input_path: dsl.Input[dsl.Artifact],
    pdf_split: List[str],
    pdf_backend: str,
    output_path: dsl.Output[dsl.Artifact],
):
    import pathlib
    import os
    from importlib import import_module

    from docling_core.types.doc.base import ImageRefMode
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend
    from docling.document_converter import DocumentConverter, PdfFormatOption

    os.environ["DOCLING_ARTIFACTS_PATH"] = "/opt/app-root/src/.cache/docling/models"

    input_dir_p = pathlib.Path(input_path.path)
    output_dir_p = pathlib.Path(output_path.path)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    input_pdfs = [input_dir_p / name for name in pdf_split]

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    allowed_backends = {e.value for e in PdfBackend}
    if pdf_backend not in allowed_backends:
        raise ValueError(
            f"Invalid pdf_backend: {pdf_backend}. Must be one of {sorted(allowed_backends)}"
        )

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

    conv_results = doc_converter.convert_all(input_pdfs, raises_on_error=True)
    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem
        output_json_path = output_dir_p / f"{doc_filename}.json"
        conv_res.document.save_as_json(output_json_path, image_mode=ImageRefMode.PLACEHOLDER)

@dsl.pipeline()
def convert_pipeline(num_splits: int = 3, pdf_backend: str = "dlparse_v4"):
    importer = import_test_pdfs()

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=num_splits,
    )

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        docling_convert(
            input_path=importer.outputs["output_path"],
            pdf_split=pdf_split,
            pdf_backend=pdf_backend,
        )

if __name__ == '__main__':
    import kfp
    output_yaml = "docling_pipeline.yaml"
    kfp.compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"\nDocling pipeline compiled to {output_yaml}")
