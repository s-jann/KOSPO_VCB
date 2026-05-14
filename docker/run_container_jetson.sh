#!/bin/bash
# Run FoundationPose container on Jetson AGX Orin
# Uses --runtime nvidia (L4T convention) instead of --gpus all

docker rm -f foundationpose-jetson >/dev/null 2>&1
DIR=$(cd "$(dirname "$0")/.." && pwd)

xhost + >/dev/null 2>&1
docker run \
    --runtime nvidia \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --network=host \
    --name foundationpose-jetson-yolo \
    --device=/dev/video0 \
    --device=/dev/video1 \
    --device=/dev/video2 \
    --device=/dev/video3 \
    --device=/dev/video4 \
    --device=/dev/video5 \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$DIR":"$DIR" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    foundationpose-jetson-yolo:latest \
    bash -c "cd $DIR && bash"
