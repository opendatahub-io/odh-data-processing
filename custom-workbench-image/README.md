# Custom Workbench for Data Processing

Open Data Hub has the ability for users to add and run [custom workbench images](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.24/html/managing_openshift_ai/creating-custom-workbench-images).

Below are guidelines on how to create a custom workbench image that is started up with the jupyter notebooks in this repository.

## Base Images

Depending on hardware resources (ex: GPU access) it is recommended to start with a jupyter-minimal image on python version 3.12.

Examples include:
```
quay.io/modh/odh-workbench-jupyter-minimal-cuda-py312-ubi9
quay.io/modh/odh-workbench-jupyter-minimal-cpu-py312-ubi9
```

The following can be added in a `Containerfile`:

```
FROM quay.io/modh/odh-workbench-jupyter-minimal-cuda-py312-ubi9
```

## Starting up Workbench with Example Notebooks at Runtime

Specific Jupyter notebooks can be the starting point of users during the start up of a custom workbench.

To configure this in a custom work bench users must download the notebooks when they are building the work bench image and set `NOTEBOOK_ROOT_DIR` to the path of the notebooks under `/opt/app-root/src`.

Since `/opt/app-root/src` is mounted by the notebook controller upon the start of a workbench image and any content in that directory [will be cleared](https://github.com/opendatahub-io/notebooks/tree/main/examples#opendatahub-dashboard) a script is required to move the notebooks.

An example script that accomplishes this is in `odh-dp-entrypoint.sh` and below:

```
#!/bin/bash
set -e

# Define source and destination paths
SRC_REPO="/opt/app-root/tmp/odh-data-processing"
DEST_REPO="/opt/app-root/src/odh-data-processing"

# Copy the notebooks to the user's persistent home directory if they don't exist
# This ensures the notebooks are present on the first and subsequent launch
if [ ! -d "$DEST_REPO" ]; then
  echo "Notebooks not found in home directory. Copying from image..."
  cp -r $SRC_REPO /opt/app-root/src/
fi

# Execute the original command to start the Jupyter server
# This will use the NOTEBOOK_ROOT_DIR env var you set
exec /opt/app-root/bin/start-notebook.sh
```

This script is used within this snippet of a `Containerfile` to set a repository of notebooks at startup.

```
USER 1001

# Clone the repository to a temporary, non-mounted directory
RUN git clone https://github.com/opendatahub-io/odh-data-processing.git /opt/app-root/tmp/odh-data-processing

# Copy a custom entrypoint script into the container
COPY --chown=1001:1 odh-dp-entrypoint.sh /opt/app-root/bin/odh-dp-entrypoint.sh

# Check to make sure entry point script can be executed by random high number user
RUN ls -l /opt/app-root/bin/odh-dp-entrypoint.sh

# Set the NOTEBOOK_ROOT_DIR to the final destination in the user's home directory
ENV NOTEBOOK_ROOT_DIR="/opt/app-root/src/odh-data-processing/notebooks/use-cases"

# Set the custom script as the new entrypoint
ENTRYPOINT ["/opt/app-root/bin/odh-dp-entrypoint.sh"]
```

## Workbench Size Recommendations

When users select the container size for the Workbench, they are advised to at least use the Medium size container.
This will have 3-6 CPUs and at least 24 GB of memory.

## Consuming Data from S3

To consume data from S3 in your workbench please refer to this [tutorial](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/pdf/working_with_data_in_an_s3-compatible_object_store/Red_Hat_OpenShift_AI_Cloud_Service-1-Working_with_data_in_an_S3-compatible_object_store-en-US.pdf) on how to include files from S3 in a Jupyter notebook.
