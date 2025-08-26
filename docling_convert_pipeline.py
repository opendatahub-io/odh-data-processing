from kfp import dsl, compiler

from docling_convert_components import (
    import_pdfs, 
    create_pdf_splits, 
    download_docling_models, 
    docling_convert,
)

@dsl.pipeline(
    name= "data-processing-docling-pipeline",
    description= "Docling convert pipeline by the Data Processing Team",
)
def convert_pipeline(
    num_splits: int = 3, 
    pdf_base_url: str = "https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    pdf_filenames: str = "2203.01017v2.pdf,2206.01062.pdf,2305.03393v1-pg9.pdf,2305.03393v1.pdf,amt_handbook_sample.pdf,code_and_formula.pdf,multi_page.pdf,redp5110_sampled.pdf",
    docling_pdf_backend: str = "dlparse_v4",
    docling_image_export_mode: str = "embedded",
    docling_table_mode: str = "accurate",
    docling_num_threads: int = 4,
    docling_timeout_per_document: int = 0,
    docling_remote_model_enabled: bool = False,
    docling_remote_model_endpoint_url: str = "",
    docling_remote_model_api_key: str = "",
    docling_remote_model_name: str = "",
):

    importer = import_pdfs(
        pdf_base_url=pdf_base_url,
        pdf_filenames=pdf_filenames,
    )

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=num_splits,
    )

    artifacts = download_docling_models()

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        converter = docling_convert(
            input_path=importer.outputs["output_path"],
            artifacts_path=artifacts.outputs["output_path"],
            pdf_split=pdf_split,
            pdf_backend=docling_pdf_backend,
            image_export_mode=docling_image_export_mode,
            table_mode=docling_table_mode,
            num_threads=docling_num_threads,
            timeout_per_document=docling_timeout_per_document,
            remote_model_enabled=docling_remote_model_enabled,
            remote_model_endpoint_url=docling_remote_model_endpoint_url,
            remote_model_api_key=docling_remote_model_api_key,
            remote_model_name=docling_remote_model_name,
        )
        
        converter.set_caching_options(False)
        converter.set_memory_request("1G")
        converter.set_memory_limit("6G")
        converter.set_cpu_request("500m")
        converter.set_cpu_limit("4")


if __name__ == "__main__":
    output_yaml = "docling_convert_pipeline_compiled.yaml"
    compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"Docling pipeline compiled to {output_yaml}")
