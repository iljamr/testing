################################################################################
FROM arm64v8/ros:noetic AS ros
################################################################################

# Install ROS packages
RUN apt-get update && apt-get install -y \
    clang-format \
    python3-pip \
    ros-noetic-rviz \
    ros-noetic-foxglove-bridge \
    ros-noetic-rosbridge-suite \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        ruff \
        pre-commit \
        jupyter

################################################################################
FROM ros AS dynablox
################################################################################

# Install dependencies
RUN apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    libpcl-dev \
    libspdlog-dev \
    libtool \
    protobuf-compiler \
    python3-catkin-tools \
    python3-vcstool \
    ros-noetic-cmake-modules \
    ros-noetic-cv-bridge \
    ros-noetic-eigen-conversions \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-eigen \
    rsync \
    wget \
    && rm -rf /var/lib/apt/lists/*

################################################################################
FROM dynablox AS user
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
RUN sudo apt install -y git

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

################################################################################
