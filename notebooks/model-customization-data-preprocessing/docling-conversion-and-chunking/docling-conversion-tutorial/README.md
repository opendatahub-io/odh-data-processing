# Docling Conversion Tutorials

A collection of tutorials and techniques for document processing and parsing using [Docling](https://docling-project.github.io/docling/) and related tools.

This repository provides a curated set of opinionated _conversion profiles_ that, based on our experience, effectively address some of the common problems users face when preparing and parsing documents for AI ingestion. Our goal is to help users achieve great results without needing to dive deep into all of Docling’s options and features.

Most examples focus on PDF to Markdown parsing, but can be easily adapted for other input and output formats.

## Installation

Please refer to the [documentation](https://docling-project.github.io/docling/installation/) for official installation instructions, but in most cases, the Docling CLI can be easily installed with:

```bash
$ pip install docling
```

If you prefer a web UI, [Docling Serve](https://github.com/docling-project/docling-serve) can be used with the same options available in the CLI. Install and run it with:

```bash
$ pip install "docling-serve[ui]"
$ docling-serve run --enable-ui
```

## Document parsing

### Standard settings

Most of the settings in the example below are already the **defaults** and will produce good and fast results for most documents. Images will be embedded in the output document as Base64 and OCR will be used only for bitmap content. A newer PDF backend (dlparse_v4) is being used.

```bash
$ docling /path/to/document.pdf \
    --to md \
    --pdf-backend dlparse_v4 \
    --image-export-mode embedded \
    --ocr \
    --ocr-engine easyocr \
    --table-mode accurate
```

A Python version of this conversion technique is available in [standard_settings.py](./standard_settings.py).

### Force OCR

Depending on how a PDF document was structured upon its creation, the backend might not be able to effectively parse its layers and contents. That may happen even in documents apparently containing pure text. In these cases, **forcing OCR** on the entire document usually produce better results.

```bash
$ docling /path/to/document.pdf \
    --to md \
    --pdf-backend dlparse_v4 \
    --image-export-mode embedded \
    --force-ocr \
    --ocr-engine easyocr \
    --table-mode accurate
```

A Python version of this conversion technique is available in [force_ocr.py](./force_ocr.py).

### Enrichment

Documents with many **code blocks**, **images**, or **formulas** can be parsed using _enriched_ conversion pipelines that add additional model executions tailored to handle these types of content. Note that these options may increase the processing time.

#### Code blocks and formulas

For documents heavy on **code blocks** and **formulas**, use:

```bash
$ docling /path/to/document.pdf \
    --to md \
    --pdf-backend dlparse_v4 \
    --image-export-mode embedded \
    --ocr \
    --ocr-engine easyocr \
    --table-mode accurate \
    --device auto \
    --enrich-code \
    --enrich-formula
```

#### Image classification and description

For documents heavy on **images**, the _picture classification_ step will understand the classes of pictures found in the document, like chart types, flow diagrams, logos, signatures, and so on; while the _picture description_ step will annotate (caption) pictures using a vision model.

```bash
$ docling /path/to/document.pdf \
    --to md \
    --pdf-backend dlparse_v4 \
    --image-export-mode embedded \
    --ocr \
    --ocr-engine easyocr \
    --table-mode accurate \
    --device auto \
    --enrich-picture-classes \
    --enrich-picture-description
```

A Python version of this conversion technique is available in [enrichment.py](./enrichment.py).

### VLM

Docling supports the use of VLMs (Visual Language Models), which can be a good choice in cases where the previous conversion profiles didn't produce good results. The Docling team provides [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview), a small and fast model specifically targeted at document conversion. Other models can also be used, like [Granite Vision](https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview).

```bash
$ docling /path/to/document.pdf \
    --to md \
    --pipeline vlm \
    --vlm-model smoldocling \
    --device auto \
    --table-mode accurate
```

A Python version of this conversion technique is available in [vlm.py](./vlm.py).

Note that using a VLM significantly increases processing time, so running it on GPU is strongly recommended. If you know the accelerator device of the machine where you're running the parsing, it might be a good idea to provide either `cuda` or `mps` in the `--device` option.
