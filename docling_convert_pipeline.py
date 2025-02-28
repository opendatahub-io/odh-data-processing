# SPDX-License-Identifier: Apache-2.0

from typing import List

from kfp import dsl

PYTHON_BASE_IMAGE = "python:3.10"

@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["gitpython"],
)
def import_test_pdfs(
    output_path: dsl.OutputPath("Directory"),
):
    import os
    import shutil
    from git import Repo

    docling_github_repo = "https://github.com/DS4SD/docling/"
    full_repo_path = os.path.join(output_path, "docling")
    Repo.clone_from(docling_github_repo, full_repo_path, branch="v2.25.0")

    # Copy some tests pdf up to the root of our output folder
    pdfs_path = os.path.join(full_repo_path, "tests", "data", "pdf")
    shutil.copytree(pdfs_path, output_path, dirs_exist_ok=True)

    # Delete the rest of the docling repo, leaving only the PDFs
    shutil.rmtree(full_repo_path)

@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
)
def create_pdf_splits(
    input_path: dsl.InputPath("Directory"),
    num_splits: int,
) -> List[List[str]]:
    import pathlib

    # Split our entire directory of pdfs into n batches, where n == num_splits
    all_pdfs = [path.name for path in pathlib.Path(input_path).glob("*.pdf")]
    splits = [all_pdfs[i::num_splits] for i in range(num_splits)]
    return splits

# A Docling container built from
# https://github.com/DS4SD/docling/blob/v2.25.0/Dockerfile
@dsl.component(
    base_image="quay.io/bbrowning/docling-kfp:v2.25.0",
)
def docling_convert(
    input_path: dsl.InputPath("Directory"),
    pdf_split: List[str],
    output_path: dsl.OutputPath("Directory"),
):
    import pathlib
    import os

    from docling_core.types.doc import ImageRefMode
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    input_pdfs = [input_path / name for name in pdf_split]

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_results = doc_converter.convert_all(
        input_pdfs,
        raises_on_error=True,
    )

    for conv_res in conv_results:
        # TODO: handle errors, record success/failure somewhere - via
        # calling some API, writing to some shared storage, or
        # something else that each parallel task can do independently
        doc_filename = conv_res.input.file.stem
        output_json_path = pathlib.Path(output_path) / f"{doc_filename}.json"
        conv_res.document.save_as_json(
            output_json_path,
            image_mode=ImageRefMode.PLACEHOLDER,
        )

@dsl.pipeline()
def convert_pipeline():
    importer = import_test_pdfs()

    pdf_splits = create_pdf_splits(
        input_path=importer.output,
        num_splits=3,
    )

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        docling_convert(
            input_path=importer.output,
            pdf_split=pdf_split,
        )

if __name__ == '__main__':
    import kfp
    output_yaml = "docling_pipeline.yaml"
    kfp.compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"\nDocling pipeline compiled to {output_yaml}")
