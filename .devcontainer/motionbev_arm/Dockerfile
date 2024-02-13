################################################################################
FROM dustynv/torch_tensorrt:r35.2.1 AS base
################################################################################

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    ruff \
    pre-commit \
    notebook \
    pytorch-lightning \
    torchmetrics \
    tqdm \
    matplotlib \
    pandas \
    numpy \
    Pillow \
    seaborn \
    scikit_learn \
    ipython

################################################################################
FROM base AS motionbev
################################################################################

RUN python3 -m pip install \
    open3d \
    pyyaml \
    strictyaml \
    icecream \
    scipy \
    numba \
    dropblock

# One of: cpu, cu117, cu118, or cu121. https://data.pyg.org/whl/
ARG CUDA=cu118

# https://github.com/rusty1s/pytorch_scatter
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html

################################################################################
FROM motionbev AS user
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

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git git-lfs

################################################################################
