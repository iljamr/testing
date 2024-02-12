################################################################################
FROM dustynv/torch_tensorrt:r35.2.1 AS base
################################################################################

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
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

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git git-lfs

################################################################################
