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
)

from standard_components import docling_convert_standard

@dsl.pipeline(
    name= "data-processing-docling-standard-pipeline",
    description= "Docling standard convert pipeline by the Data Processing Team",
)
def convert_pipeline(
    num_splits: int = 3,
    pdf_from_s3: bool = False,
    pdf_filenames: str = "2203.01017v2.pdf,2206.01062.pdf,2305.03393v1-pg9.pdf,2305.03393v1.pdf,amt_handbook_sample.pdf,code_and_formula.pdf,multi_page.pdf,redp5110_sampled.pdf",
    # URL source params
    pdf_base_url: str = "https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    # Docling params
    docling_pdf_backend: str = "dlparse_v4",
    docling_image_export_mode: str = "embedded",
    docling_table_mode: str = "accurate",
    docling_num_threads: int = 4,
    docling_timeout_per_document: int = 300,
    docling_ocr: bool = True,
    docling_force_ocr: bool = False,
    docling_ocr_engine: str = "tesseract",
    docling_allow_external_plugins: bool = False,
    docling_enrich_code: bool = False,
    docling_enrich_formula: bool = False,
    docling_enrich_picture_classes: bool = False,
    docling_enrich_picture_description: bool = False,
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
        pipeline_type="standard",
        remote_model_endpoint_enabled=False,
    )
    artifacts.set_caching_options(False)

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        converter = docling_convert_standard(
            input_path=importer.outputs["output_path"],
            artifacts_path=artifacts.outputs["output_path"],
            pdf_filenames=pdf_split,
            pdf_backend=docling_pdf_backend,
            image_export_mode=docling_image_export_mode,
            table_mode=docling_table_mode,
            num_threads=docling_num_threads,
            timeout_per_document=docling_timeout_per_document,
            ocr=docling_ocr,
            force_ocr=docling_force_ocr,
            ocr_engine=docling_ocr_engine,
            allow_external_plugins=docling_allow_external_plugins,
            enrich_code=docling_enrich_code,
            enrich_formula=docling_enrich_formula,
            enrich_picture_classes=docling_enrich_picture_classes,
            enrich_picture_description=docling_enrich_picture_description,
        )
        converter.set_caching_options(False)
        converter.set_memory_request("1G")
        converter.set_memory_limit("6G")
        converter.set_cpu_request("500m")
        converter.set_cpu_limit("4")


if __name__ == "__main__":
    output_yaml = "standard_convert_pipeline_compiled.yaml"
    compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"Docling standard pipeline compiled to {output_yaml}")
