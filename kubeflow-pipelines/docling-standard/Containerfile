FROM registry.access.redhat.com/ubi9/python-311:9.6-1755074620

USER 0

RUN cat /etc/redhat-release

RUN dnf -y update --setopt=tsflags=nodocs && \
    dnf clean all && \
    dnf -y install libglvnd-glx glib2 && \
    dnf -y clean all && \
    rm -rf /var/cache/dnf

RUN mkdir -p /opt/app-root/bin && \
    chown -R 1001:0 /opt/app-root

USER 1001

WORKDIR /opt/app-root/bin

ENV OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/opt/app-root/src \
    PATH=/opt/app-root/src:$PATH \
    DOCLING_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models

# This will install torch with *only* cpu support
# Remove the --extra-index-url part if you want to install all the gpu requirements
# For more details in the different torch distribution visit https://pytorch.org/.
RUN pip3 install --no-cache-dir docling --extra-index-url https://download.pytorch.org/whl/cpu

ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

RUN docling --version

RUN echo "Downloading models..." && \
    docling-tools models download -o "${DOCLING_ARTIFACTS_PATH}"

# Running with `DOCLING_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models` will use the
# models included in the container image.
