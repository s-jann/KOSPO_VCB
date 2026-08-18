#!/usr/bin/env bash

PROJECT_ROOT="/home/robot/KOSPO_VCB"
LOG_DIR="$PROJECT_ROOT/logs/headless"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

source /opt/ros/humble/setup.bash

if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
    source "$PROJECT_ROOT/install/setup.bash"
fi


CAMERA_PID=""
FP_PID=""
YOLO_PID=""
RESULT_PID=""


# ============================================================
# 종료 처리
# ============================================================

cleanup()
{
    echo ""
    echo "========================================"
    echo " Stopping KOSPO VCB system"
    echo "========================================"

    if [ -n "$RESULT_PID" ]; then
        kill "$RESULT_PID" 2>/dev/null || true
    fi

    if [ -n "$YOLO_PID" ]; then
        kill "$YOLO_PID" 2>/dev/null || true
    fi

    if [ -n "$FP_PID" ]; then
        kill "$FP_PID" 2>/dev/null || true
    fi

    if [ -n "$CAMERA_PID" ]; then
        kill "$CAMERA_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true

    echo "All processes stopped."
}

trap cleanup EXIT INT TERM


# ============================================================
# ROS topic 대기
# ============================================================

wait_for_topic()
{
    TOPIC="$1"
    MAX_WAIT="$2"

    COUNT=0

    echo "Waiting for topic: $TOPIC"

    while true
    do
        if ros2 topic list 2>/dev/null | grep -Fxq "$TOPIC"; then
            echo "Topic ready: $TOPIC"
            return 0
        fi

        sleep 1
        COUNT=$((COUNT + 1))

        if [ "$COUNT" -ge "$MAX_WAIT" ]; then
            echo "ERROR: timeout waiting for $TOPIC"
            return 1
        fi
    done
}


echo "========================================"
echo " KOSPO VCB HEADLESS SYSTEM"
echo "========================================"


# ============================================================
# 1. RealSense
# ============================================================

echo ""
echo "[1/5] Starting RealSense..."

ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    </dev/null \
    > "$LOG_DIR/realsense.log" 2>&1 &

CAMERA_PID=$!

echo "RealSense PID: $CAMERA_PID"


wait_for_topic \
    "/camera/camera/color/image_raw" \
    30 || exit 1

wait_for_topic \
    "/camera/camera/color/camera_info" \
    30 || exit 1


# ============================================================
# 2. FoundationPose HEADLESS
# ============================================================

echo ""
echo "[2/5] Starting FoundationPose headless..."

python3 -u camera/pose_trigger_ros2.py \
    --rotation 90_cw \
    --headless \
    </dev/null \
    > "$LOG_DIR/foundationpose.log" 2>&1 &

FP_PID=$!

echo "FoundationPose PID: $FP_PID"


wait_for_topic \
    "/foundation_pose/result" \
    30 || exit 1


# ============================================================
# 3. 6D Pose result monitor
#
# /foundation_pose/result를 받아서
# 사람이 보기 쉬운 형태로 터미널에 출력
# ============================================================

echo ""
echo "[3/5] Starting 6D pose result monitor..."

python3 -u - <<'PY' &
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class FoundationPoseResultMonitor(Node):

    def __init__(self):
        super().__init__("foundationpose_result_monitor")

        self.attempts = {}

        self.sub = self.create_subscription(
            String,
            "/foundation_pose/result",
            self.callback,
            10,
        )

    def callback(self, msg):

        try:
            data = json.loads(msg.data)

        except Exception as exc:
            print(
                "\n[FP RESULT] Invalid JSON:",
                exc,
                flush=True,
            )
            return

        command_id = data.get("command_id")
        target_label = data.get("target_label")
        current_state = data.get("current_state")
        desired_state = data.get("desired_state")

        object_found = bool(
            data.get("object_found", False)
        )

        # 같은 command_id에 대해 몇 번째 FP 결과인지
        if command_id not in self.attempts:
            self.attempts[command_id] = 0

        self.attempts[command_id] += 1

        attempt = self.attempts[command_id]

        print()
        print("=" * 60)
        print(" FOUNDATIONPOSE RESULT")
        print("=" * 60)

        print(f" command_id     : {command_id}")
        print(f" target         : {target_label}")
        print(f" current_state  : {current_state}")
        print(f" desired_state  : {desired_state}")
        print(f" attempt        : {attempt}/3")
        print(f" object_found   : {object_found}")

        if not object_found:
            print()
            print(" RESULT          : OBJECT NOT FOUND")

            if attempt < 3:
                print("                  -> retry expected")
            else:
                print("                  -> maximum attempts reached")

            print("=" * 60)
            print()
            return

        pose_6d = data.get("pose_6d") or {}

        translation = (
            pose_6d.get("translation") or {}
        )

        euler = (
            pose_6d.get("rotation_euler_deg") or {}
        )

        quat = (
            pose_6d.get("rotation_quaternion") or {}
        )

        confidence = data.get(
            "confidence",
            0.0,
        )

        print()
        print(" -------- 6D POSE --------")
        print()

        print(" Translation [m]")
        print(
            f"   x = {translation.get('x')}"
        )
        print(
            f"   y = {translation.get('y')}"
        )
        print(
            f"   z = {translation.get('z')}"
        )

        print()
        print(" Rotation Euler [deg]")
        print(
            f"   roll  = {euler.get('roll')}"
        )
        print(
            f"   pitch = {euler.get('pitch')}"
        )
        print(
            f"   yaw   = {euler.get('yaw')}"
        )

        print()
        print(" Quaternion [x y z w]")
        print(
            f"   x = {quat.get('x')}"
        )
        print(
            f"   y = {quat.get('y')}"
        )
        print(
            f"   z = {quat.get('z')}"
        )
        print(
            f"   w = {quat.get('w')}"
        )

        print()
        print(
            f" confidence = {confidence}"
        )

        print()
        print(" RESULT = SUCCESS")
        print("=" * 60)
        print()


def main():

    rclpy.init()

    node = FoundationPoseResultMonitor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

PY

RESULT_PID=$!

echo "Result monitor PID: $RESULT_PID"


# ============================================================
# 4. YOLO / OCR / HSV
# ============================================================

echo ""
echo "[4/5] Starting YOLO perception..."

python3 -u yolo/src/main_infer_ros2_semantic_fp.py \
    </dev/null \
    > "$LOG_DIR/yolo.log" 2>&1 &

YOLO_PID=$!

echo "YOLO PID: $YOLO_PID"

sleep 5


# ============================================================
# 5. Operator command
# ============================================================

echo ""
echo "[5/5] Starting operator command..."

echo ""
echo "============================================================"
echo " KOSPO VCB SYSTEM READY"
echo "============================================================"
echo ""
echo " Example:"
echo ""
echo "   4SW01-01B close"
echo "   4SW02-01B open"
echo ""
echo " FoundationPose 6D result will appear in this terminal."
echo ""
echo " Ctrl+C : stop all processes"
echo "============================================================"
echo ""


python3 -u operator_command.py