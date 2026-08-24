#!/usr/bin/env bash
#
# 명령 기반 VCB 6D Pose 시스템
#
#   operator 명령 (예: "4SW01-02B close") 입력
#     -> YOLO(semantic_fp)가 OCR 라벨 매칭 + 상태 판정
#     -> ACTION_REQUIRED 3프레임 연속 시 FoundationPose 요청
#     -> pose_trigger_ros2.py가 6D pose 추정 후 결과 발행
#
# 사용법:
#   ./run_vcb_command.sh          # headless (운영용)
#   ./run_vcb_command.sh --gui    # FP/YOLO GUI 표시 (실험실 확인용)
#
# 카메라 노드가 이미 실행 중이면 재사용한다 (중복 기동 금지).
# 종료: Ctrl+C

PROJECT_ROOT="/home/robot/KOSPO_VCB"
LOG_DIR="$PROJECT_ROOT/logs/vcb_command"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

source /opt/ros/humble/setup.bash

if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
    source "$PROJECT_ROOT/install/setup.bash"
fi

# GUI 여부
FP_ARGS="--headless"
YOLO_ARGS="--headless"
if [ "$1" = "--gui" ]; then
    FP_ARGS=""
    YOLO_ARGS="--gui"
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
    echo " Stopping VCB command system"
    echo "========================================"

    for PID in "$RESULT_PID" "$YOLO_PID" "$FP_PID" "$CAMERA_PID"; do
        if [ -n "$PID" ]; then
            kill "$PID" 2>/dev/null || true
        fi
    done

    wait 2>/dev/null || true

    echo "All processes stopped."
    echo "Logs: $LOG_DIR/"
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
echo " VCB COMMAND SYSTEM"
echo "========================================"


# ============================================================
# 1. RealSense (이미 실행 중이면 재사용)
# ============================================================

echo ""
echo "[1/4] Checking RealSense..."

if ros2 topic list 2>/dev/null | grep -Fxq "/camera/camera/color/image_raw"; then
    echo "기존 RealSense 노드 감지 -> 재사용합니다."
    CAMERA_PID=""
else
    echo "RealSense 노드 없음 -> 새로 시작합니다."
    ros2 launch realsense2_camera rs_launch.py \
        align_depth.enable:=true \
        initial_reset:=true \
        </dev/null \
        > "$LOG_DIR/realsense.log" 2>&1 &

    CAMERA_PID=$!
    echo "RealSense PID: $CAMERA_PID"

    wait_for_topic "/camera/camera/color/image_raw" 30 || exit 1
fi

wait_for_topic "/camera/camera/aligned_depth_to_color/image_raw" 30 || exit 1


# ============================================================
# 2. FoundationPose (요청 기반 트리거 노드)
# ============================================================

echo ""
echo "[2/4] Starting FoundationPose trigger node..."

python3 -u camera/pose_trigger_ros2.py \
    --rotation 90_cw \
    --input_mode rgbd \
    $FP_ARGS \
    </dev/null \
    > "$LOG_DIR/foundationpose.log" 2>&1 &

FP_PID=$!

echo "FoundationPose PID: $FP_PID"

wait_for_topic "/foundation_pose/result" 60 || exit 1


# ============================================================
# 3. 6D Pose 결과 모니터 (터미널 출력)
# ============================================================

echo ""
echo "[3/4] Starting topic monitor..."

# 공용 토픽 모니터: /vcb/perception 요약 + /foundation_pose/result 상세 출력
python3 -u scripts/vcb_topic_monitor.py &

RESULT_PID=$!

echo "Result monitor PID: $RESULT_PID"


# ============================================================
# 4. YOLO perception (semantic_fp)
# ============================================================

echo ""
echo "[4/4] Starting YOLO perception (semantic_fp)..."

python3 -u yolo/src/main_infer_ros2_semantic_fp.py \
    $YOLO_ARGS \
    </dev/null \
    > "$LOG_DIR/yolo.log" 2>&1 &

YOLO_PID=$!

echo "YOLO PID: $YOLO_PID"

sleep 5


# ============================================================
# 5. Operator command (foreground)
# ============================================================

echo ""
echo "============================================================"
echo " VCB COMMAND SYSTEM READY"
echo "============================================================"
echo ""
echo " 명령 예:"
echo "   4SW01-02B close"
echo "   4SW02-01B open"
echo ""
echo " 6D 결과는 이 터미널에 출력됩니다."
echo " 로그: tail -f $LOG_DIR/yolo.log / foundationpose.log"
echo ""
echo " Ctrl+C : 전체 종료"
echo "============================================================"
echo ""

python3 -u operator_command.py
