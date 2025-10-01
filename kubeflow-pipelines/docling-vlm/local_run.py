from typing import List

from kfp import dsl, local

from docling_convert_components import (
    create_pdf_splits,
    docling_convert,
    download_docling_models,
    import_pdfs,
)


@dsl.component(base_image="python:3.11")
def take_first_split(splits: List[List[str]]) -> List[str]:
    return splits[0] if splits else []


@dsl.pipeline()
def convert_pipeline_local():
    importer = import_pdfs(
        filenames="2305.03393v1-pg9.pdf",
        base_url="https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    )

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=1,
    )

    artifacts = download_docling_models(remote_model_endpoint_enabled=False)

    first_split = take_first_split(splits=pdf_splits.output)

    docling_convert(
        input_path=importer.outputs["output_path"],
        artifacts_path=artifacts.outputs["output_path"],
        pdf_filenames=first_split.output,
    )


def main() -> None:
    # Requires: pip install docker; and a Docker-compatible daemon (Docker or Podman socket)
    local.init(runner=local.DockerRunner())
    convert_pipeline_local()


if __name__ == "__main__":
    main()
