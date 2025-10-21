# ODH Data Processing

![Status dev-preview](https://img.shields.io/badge/status-dev--preview-blue)
![GitHub License](https://img.shields.io/github/license/opendatahub-io/odh-data-processing)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/opendatahub-io/odh-data-processing)

This repository provides reference **data-processing pipelines and examples** for [Open Data Hub](https://github.com/opendatahub-io) / [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai). It focuses on **document conversion** and **chunking** using the [Docling](https://docling-project.github.io/docling/) toolkit, packaged as [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) (KFP), example [Jupyter Notebooks](https://jupyter.org/), and helper scripts.

The workbenches directory also provides a guide on how to create a custom [workbench image](https://github.com/opendatahub-io-contrib/workbench-images) to run Docling and the example notebooks in this repository.

## 📦 Repository Structure

```bash
odh-data-processing
|
|- kubeflow-pipelines
|   |- docling-standard
|   |- docling-vlm
|
|- notebooks
    |- tutorials
    |- use-cases
|
|- custom-workbench-image
```

## ✨ Getting Started

### Kubeflow Pipelines

Refer to the [ODH Data Processing Kubeflow Pipelines](kubeflow-pipelines) documentation for instructions on how to install, run, and customize the [Standard](kubeflow-pipelines/docling-standard) and [VLM](kubeflow-pipelines/docling-vlm) pipelines.

## 🤝 Contributing

We welcome issues and pull requests. Please:
- Open an issue describing the change.
- Include testing instructions.
- For pipeline/component changes, recompile the pipeline and update generated YAML if applicable.
- Keep parameter names and docs consistent between code and README.

### Quality & CI

This repo enforces Python style and clean notebooks via **pre-commit** and a GitHub Actions workflow.

**What runs:**
- **Ruff** (lint, autofixes)
- **Black** (format)
- **isort** (import order, Black profile)
- **nbstripout** (removes Jupyter outputs)

**Where it runs:**
- On every **Pull Request**
- Once **post-merge** to `main` (final validation)

**Quick start (local):**
```bash
pip install pre-commit
pre-commit install               # installs the git hook
pre-commit run --all-files       # run all checks on the repo

## 📄 License

Apache License 2.0
