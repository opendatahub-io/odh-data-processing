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
