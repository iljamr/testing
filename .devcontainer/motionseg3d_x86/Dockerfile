################################################################################
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html
FROM nvcr.io/nvidia/pytorch:21.10-py3 AS base 
################################################################################

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    libegl1 \
    libgl1 \
    libgomp1 \
    libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TorchSparse
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    FORCE_CUDA=1 python3 -m pip install --no-cache-dir --upgrade \
    git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

RUN python3 -m pip install --no-cache-dir --upgrade \
    ruff \
    pre-commit \
    open3d \
    icecream \
    tensorboardX \
    directory-tree \
    Werkzeug==2.2.3

# Install SoftPool
WORKDIR /opt
RUN git clone https://github.com/alexandrosstergiou/SoftPool.git
WORKDIR /opt/SoftPool/pytorch
RUN git checkout 2d2ec6d
RUN make install

################################################################################
FROM base AS user
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
ENV CUDA_VISIBLE_DEVICES=0

################################################################################
