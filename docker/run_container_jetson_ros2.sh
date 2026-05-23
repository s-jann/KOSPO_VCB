#!/bin/bash
# Run FoundationPose ROS2 container on Jetson AGX Orin
# JetPack 6 / ROS2 Humble version

docker rm -f foundationpose-jetson-yolo >/dev/null 2>&1
DIR=$(cd "$(dirname "$0")/.." && pwd)

xhost + >/dev/null 2>&1

docker run \
    --runtime nvidia \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --privileged \
    --network=host \
    --name foundationpose-jetson-yolo \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /dev:/dev \
    -v "$DIR":"$DIR" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    foundationpose-jetson-yolo:ros2-humble \
    bash -c "cd $DIR && bash"