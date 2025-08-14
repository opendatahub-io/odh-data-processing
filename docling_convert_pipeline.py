from kfp import dsl, compiler

from docling_convert_components import (
    import_test_pdfs, 
    create_pdf_splits, 
    download_docling_models, 
    docling_convert,
)

@dsl.pipeline(
    name= "data-processing-docling-pipeline",
    description= "Docling convert pipeline by the Data Processing Team",
)
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
    output_yaml = "docling_convert_pipeline_compiled.yaml"
    compiler.Compiler().compile(convert_pipeline, output_yaml)
    print(f"Docling pipeline compiled to {output_yaml}")
