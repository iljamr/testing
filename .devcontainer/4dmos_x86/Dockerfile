################################################################################
ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS minkowski
################################################################################

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TZ=Europe/Stockholm
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install tzdata
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

################################################################################
FROM minkowski AS mos4d
################################################################################

ENV PROJECT=/mos4d
RUN mkdir -p $PROJECT
RUN mkdir -p /mos4d/logs

ENV DATA=/mos4d/data
RUN mkdir -p $DATA

# Install MinkowskiEngine Dependencies
RUN apt-get update || true && apt-get install --no-install-recommends -y \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install project related dependencies
WORKDIR $PROJECT
COPY . $PROJECT
RUN python3 -m pip install --editable .
RUN rm -rf $PROJECT

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir --upgrade \
    ruff \
    pre-commit \
    jupyter \
    directory-tree

################################################################################
FROM mos4d AS user
################################################################################

# Add vscode user with same UID and GID as your host system
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Set environment variable for user
ENV USER=$USERNAME
ENV DATA_PATH=/home/${USERNAME}/data
ENV DATA=/home/${USERNAME}/data/kitti/SemanticKITTI/symlinks/4dmos/sequences

################################################################################

