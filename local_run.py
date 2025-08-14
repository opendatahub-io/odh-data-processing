from typing import List

from kfp import dsl, local

from docling_convert_components import (
    import_test_pdfs,
    create_pdf_splits,
    docling_convert,
    download_docling_models,
)


@dsl.component(base_image="python:3.11")
def take_first_split(splits: List[List[str]]) -> List[str]:
    return splits[0] if splits else []


@dsl.pipeline()
def convert_pipeline_local(num_splits: int = 1, pdf_backend: str = "dlparse_v4"):
    importer = import_test_pdfs()

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=num_splits,
    )

    artifacts = download_docling_models()

    first_split = take_first_split(splits=pdf_splits.output)

    docling_convert(
        input_path=importer.outputs["output_path"],
        pdf_split=first_split.output,
        pdf_backend=pdf_backend,
        artifacts_path=artifacts.outputs["output_path"],
    )


def main() -> None:
    # Requires: pip install docker; and a Docker-compatible daemon (Docker or Podman socket)
    local.init(runner=local.DockerRunner())
    convert_pipeline_local(num_splits=1, pdf_backend="dlparse_v4")


if __name__ == "__main__":
    main()
