################################################################################
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html
# cuda 11.4 
FROM nvcr.io/nvidia/pytorch:21.10-py3 AS base 
################################################################################

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    libegl1 \
    libgl1 \
    libgomp1 \
    libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 && \
    python3 -m pip install --no-cache-dir --upgrade \
    open3d \
    Werkzeug==2.2.3

WORKDIR /opt
RUN git clone https://github.com/alexandrosstergiou/SoftPool.git
WORKDIR /opt/SoftPool/pytorch
RUN git checkout 2d2ec6d
RUN make install

################################################################################
FROM base AS conda
################################################################################

# Install SoftPool follow https://github.com/alexandrosstergiou/SoftPool
# git clone https://github.com/alexandrosstergiou/SoftPool.git
# cd SoftPool
# git checkout 2d2ec6d # rollback to 2d2ec6dca10b7683ffd41061a27910d67816bfa5
# cd pytorch

# make install
# --- (optional) ---
# make test

# # Install TorchSparse follow https://github.com/mit-han-lab/torchsparse
# sudo apt install libsparsehash-dev 
# pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# RUN python -m pip install \
#     icecream \
#     tqdm \
#     matplotlib \
#     directory-tree \
#     open3d


# open3d \
# pyyaml \
# tensorboard \
# tensorboardX \
# vispy \
# cython \
# easydict \
# nose \
# numpy \
# opencv-contrib-python \
# opencv-python \
# scikit-learn \
# sklearn \
# strictyaml \
# sparsehash \
# torchvision \
# ipython \
# numba \
# pyqt

# transforms3d
# numba
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh && \
#     echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
# ENV PATH /opt/conda/bin:$PATH

# RUN conda install -y python=3.8

# COPY test.yml /tmp/conda-tmp/
# COPY requirements.txt /tmp/conda-tmp/

# RUN conda init bash \
#     && . ~/.bashrc \
#     && conda env create -f /tmp/conda-tmp/test.yml \
#     && conda activate mos3d \
#     && pip install -r /tmp/conda-tmp/requirements.txt 
#&& pip install git+http://github.com/mit-han-lab/torchsparse.git@v1.4.0 

################################################################################
# FROM conda AS motionseg3d
################################################################################



################################################################################
FROM conda AS user
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
# RUN sudo apt update && sudo apt upgrade -y

# Install Git
# RUN sudo apt install -y git wget

################################################################################
