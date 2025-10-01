import sys
from pathlib import Path

# Add the parent directory to Python path to find common
sys.path.insert(0, str(Path(__file__).parent.parent))

from kfp import dsl, compiler

# Import common components from the shared module
from common import (
    import_pdfs,
    create_pdf_splits,
    download_docling_models,
    MODEL_TYPE_VLM,
)

from vlm_components import docling_convert_vlm

@dsl.pipeline(
    name= "data-processing-docling-vlm-pipeline",
    description= "Docling VLM convert pipeline by the Data Processing Team",
)
def convert_pipeline(
    num_splits: int = 3,
    pdf_from_s3: bool = False,
    pdf_filenames: str = "2203.01017v2.pdf",
    # URL source params
    pdf_base_url: str = "https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    # Docling params
    docling_num_threads: int = 4,
    docling_timeout_per_document: int = 300,
    docling_image_export_mode: str = "embedded",
    docling_remote_model_enabled: bool = False,
    docling_accelerator_device: str = "auto",  # parameter for accelerator device
):
    from kfp import kubernetes # pylint: disable=import-outside-toplevel

    s3_secret_mount_path = "/mnt/secrets"
    importer = import_pdfs(
        filenames=pdf_filenames,
        base_url=pdf_base_url,
        from_s3=pdf_from_s3,
        s3_secret_mount_path=s3_secret_mount_path,
    )
    importer.set_caching_options(False)
    kubernetes.use_secret_as_volume(importer,
            secret_name="data-processing-docling-pipeline",
            mount_path=s3_secret_mount_path,
            optional=True)

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=num_splits,
    )

    artifacts = download_docling_models(
        pipeline_type=MODEL_TYPE_VLM,
        remote_model_endpoint_enabled=docling_remote_model_enabled,
    )
    artifacts.set_caching_options(False)

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        remote_model_secret_mount_path = "/mnt/secrets"
        converter = docling_convert_vlm(
            input_path=importer.outputs["output_path"],
            artifacts_path=artifacts.outputs["output_path"],
            pdf_filenames=pdf_split,
            num_threads=docling_num_threads,
            timeout_per_document=docling_timeout_per_document,
            image_export_mode=docling_image_export_mode,
            remote_model_enabled=docling_remote_model_enabled,
            remote_model_secret_mount_path=remote_model_secret_mount_path,
            accelerator_device=docling_accelerator_device,  # parameter for accelerator device
        )
        converter.set_caching_options(False)
        converter.set_memory_request("1G")
        converter.set_memory_limit("6G")
        converter.set_cpu_request("500m")
        converter.set_cpu_limit("4")
        kubernetes.use_secret_as_volume(converter,
            secret_name="data-processing-docling-pipeline",
            mount_path=remote_model_secret_mount_path,
            optional=True)


if __name__ == "__main__":
    output_yaml = "vlm_convert_pipeline_compiled.yaml"
    compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"Docling vlm pipeline compiled to {output_yaml}")
