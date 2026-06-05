#!/bin/bash
# Run FoundationPose ROS2 container on Jetson AGX Orin

docker rm -f foundationpose-jetson-yolo >/dev/null 2>&1
DIR=$(cd "$(dirname "$0")/.." && pwd)

xhost + >/dev/null 2>&1

docker run \
    --runtime nvidia \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --network=host \
    --name foundationpose-jetson-yolo \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    --device /dev/video1:/dev/video1 \
    --device /dev/video2:/dev/video2 \
    --device /dev/video3:/dev/video3 \
    --device /dev/video4:/dev/video4 \
    --device /dev/video5:/dev/video5 \
    --device /dev/media1:/dev/media1 \
    --device /dev/media2:/dev/media2 \
    -v "$DIR":"$DIR" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    foundationpose-jetson-yolo:ros2-humble \
    bash -c "cd $DIR && source /opt/ros/humble/setup.bash && bash"