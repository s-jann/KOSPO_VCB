#!/bin/bash
source /opt/ros/humble/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  pointcloud.enable:=false \
  enable_gyro:=false \
  enable_accel:=false \
  rgb_camera.color_profile:=1280x720x30 \
  depth_module.depth_profile:=640x480x30
