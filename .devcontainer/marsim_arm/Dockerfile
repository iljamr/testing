################################################################################
FROM arm64v8/ros:noetic AS ros
################################################################################

# Install ros packages
RUN apt-get update && apt-get install -y \
    python3-catkin-tools \
    ros-noetic-cmake-modules \
    ros-noetic-cv-bridge \
    ros-noetic-eigen-conversions \
    ros-noetic-foxglove-bridge \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    ros-noetic-rosbridge-suite \
    ros-noetic-rviz \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-eigen \
    && rm -rf /var/lib/apt/lists/*

################################################################################
FROM ros AS base
################################################################################

# Install base packages
RUN apt-get update && apt-get install -y \
    black \
    build-essential \
    libtool \
    clang-format \
    isort \
    rsync \
    wget \
    && rm -rf /var/lib/apt/lists/*

################################################################################
FROM base AS marsim
################################################################################

# Install MARSIM dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libglew-dev \
    libglfw3-dev \
    libpcl-dev \
    libspdlog-dev \
    protobuf-compiler \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

################################################################################
FROM marsim AS user
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
RUN echo "source /home/${USERNAME}/code/simulators/marsim/marsim_ws/devel/setup.bash" >> ~/.bashrc

################################################################################
