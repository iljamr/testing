################################################################################
FROM arm64v8/ros:noetic AS ros
################################################################################

# Install ROS packages
RUN apt update && apt install -y \
    clang-format \
    python3-pip \
    ros-noetic-rviz \
    ros-noetic-foxglove-bridge \
    python3-catkin-tools \
    ros-noetic-rosbridge-suite \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    ruff \
    pre-commit \
    jupyter \
    directory-tree

################################################################################
FROM ros AS mdetector
################################################################################

# Install dependencies
RUN apt update && apt install -y \
    build-essential \
    libpcl-dev \
    libeigen3-dev \
    libspdlog-dev \
    libtbb-dev \
    libopencv-dev \
    ros-noetic-eigen-conversions \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Livox SDK. Based on the instructions from:
# https://github.com/Livox-SDK/Livox-SDK
ENV LIVOX_INSTALL_DIR=/opt/livox

RUN git clone https://github.com/Livox-SDK/Livox-SDK.git $LIVOX_INSTALL_DIR

WORKDIR ${LIVOX_INSTALL_DIR}/build
RUN  cmake ..
RUN make
RUN make install

# Install Livox ROS driver. Based on the instructions from:
# https://github.com/Livox-SDK/livox_ros_driver?tab=readme-ov-file
ENV LIVOX_WS_DIR=${LIVOX_INSTALL_DIR}/livox_ws
WORKDIR $LIVOX_INSTALL_DIR

RUN git clone https://github.com/Livox-SDK/livox_ros_driver.git ${LIVOX_WS_DIR}/src
WORKDIR $LIVOX_WS_DIR
RUN bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

################################################################################
FROM mdetector AS user
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
RUN echo "source ${LIVOX_WS_DIR}/devel/setup.bash" >> ~/.bashrc

################################################################################
