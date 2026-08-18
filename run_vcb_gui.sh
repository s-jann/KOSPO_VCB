#!/usr/bin/env bash

PROJECT_ROOT="/home/robot/KOSPO_VCB"
LOG_DIR="$PROJECT_ROOT/logs/gui"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

source /opt/ros/humble/setup.bash

if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
    source "$PROJECT_ROOT/install/setup.bash"
fi

CAMERA_PID=""
FP_PID=""
YOLO_PID=""

cleanup()
{
    echo ""
    echo "========================================"
    echo " Stopping KOSPO VCB system"
    echo "========================================"

    [ -n "$YOLO_PID" ] && kill "$YOLO_PID" 2>/dev/null || true
    [ -n "$FP_PID" ] && kill "$FP_PID" 2>/dev/null || true
    [ -n "$CAMERA_PID" ] && kill "$CAMERA_PID" 2>/dev/null || true

    wait 2>/dev/null || true

    echo "All processes stopped."
}

trap cleanup EXIT INT TERM


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
echo " KOSPO VCB GUI TEST SYSTEM"
echo "========================================"


# ============================================================
# 1. RealSense
# ============================================================

echo ""
echo "[1/4] Starting RealSense..."

ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    </dev/null \
    > "$LOG_DIR/realsense.log" 2>&1 &

CAMERA_PID=$!

wait_for_topic \
    "/camera/camera/color/image_raw" \
    30 || exit 1


# ============================================================
# 2. FoundationPose GUI
# ============================================================

echo ""
echo "[2/4] Starting FoundationPose GUI..."

python3 -u camera/pose_trigger_ros2.py \
    --rotation 90_cw \
    </dev/null \
    > "$LOG_DIR/foundationpose.log" 2>&1 &

FP_PID=$!

sleep 3


# ============================================================
# 3. YOLO GUI
# YAML: enable_gui: true
# ============================================================

echo ""
echo "[3/4] Starting YOLO GUI..."

python3 -u yolo/src/main_infer_ros2_semantic_fp.py \
    </dev/null \
    > "$LOG_DIR/yolo.log" 2>&1 &

YOLO_PID=$!

sleep 5


# ============================================================
# 4. Operator command
# ============================================================

echo ""
echo "========================================"
echo " VCB SYSTEM READY"
echo "========================================"
echo ""
echo "Example:"
echo "  4SW01-01B close"
echo ""
echo "Enter operator command below."
echo "Ctrl+C : stop all processes"
echo "========================================"
echo ""

# 이것만 현재 터미널의 stdin/stdout 사용
python3 -u operator_command.py