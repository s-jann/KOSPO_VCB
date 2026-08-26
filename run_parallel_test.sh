#!/usr/bin/env bash
#
# 테스트용: 사용자 명령 없이 YOLO 인식 + 6D Pose 추정 동시 실행
#
#   - YOLO: main_infer_ros2_semantic_fp.py
#     * 명령이 없으므로 target 판정(READY_FOR_WORK/BLOCKED_CLOSE 등)은 NO_COMMAND
#     * --diagnostic-hsv로 실행하여, 명령 없이도 검출된 모든 VCB의
#       status에 HSV를 수행 (판정에는 미사용, 파이프라인 동작 확인용)
#     * /vcb/perception 발행
#     * 입력 설정은 yolo/configs/infer_config.yaml (mode: ros, rotation: 90_cw)
#   - 6D  : pose_streamer_ros2.py (연속 자동 register, 트리거 없음)
#
# 목적: 두 모델이 같은 카메라/GPU에서 동시에 돌 때의
#       프레임레이트 저하, FP 소요시간 증가, 예외 발생 여부 확인
# 토픽: run_vcb_command.sh와 동일 세트 사용
#
# 사용법:
#   ./run_parallel_test.sh          # headless 모드 (토픽 요약이 터미널에 출력)
#   ./run_parallel_test.sh --gui    # GUI 모드 (디버깅 용도로 화면에 출력)
#
# 종료: Ctrl+C (모든 프로세스 정리)

PROJECT_ROOT="/home/robot/KOSPO_VCB"
LOG_DIR="$PROJECT_ROOT/logs/parallel_test"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

source /opt/ros/humble/setup.bash

if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
    source "$PROJECT_ROOT/install/setup.bash"
fi

# 기본 headless, --gui 옵션 시 두 GUI 모두 표시 (run_vcb_command.sh와 동일)
# --diagnostic-hsv: 사용자 명령이 없어도 검출된 모든 VCB의 status에 대해
#                    HSV를 수행한다 (판정에는 미사용, 파이프라인 검증 전용).
#                    run_vcb_command.sh에는 넣지 않으므로 명령 기반 동작에는 영향 없음.
STREAMER_EXTRA_ARGS="--headless"
YOLO_ARGS="--headless --diagnostic-hsv"
if [ "$1" = "--gui" ]; then
    STREAMER_EXTRA_ARGS=""
    YOLO_ARGS="--gui --diagnostic-hsv"
fi

CAMERA_PID=""
FP_PID=""
YOLO_PID=""


# ============================================================
# 종료 처리
# ============================================================

cleanup()
{
    echo ""
    echo "========================================"
    echo " Stopping parallel test"
    echo "========================================"

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
echo " YOLO + 6D Pose 병렬 테스트"
echo "========================================"


# ============================================================
# 1. RealSense
#
# 중요: RealSense 노드는 카메라 1대당 1개만 있어야 한다.
# 이미 다른 터미널에서 실행 중이면 (예: scripts/run_realsense_vcb_ros2.sh)
# 그 노드를 그대로 사용하고 여기서는 새로 띄우지 않는다.
# (중복 기동 시 initial_reset이 스트리밍 중인 카메라를 리셋시켜
#  기존 노드가 죽고 카메라가 미인식 상태에 빠진다)
# ============================================================

start_camera()
{
    ros2 launch realsense2_camera rs_launch.py \
        align_depth.enable:=true \
        initial_reset:=true \
        </dev/null \
        > "$LOG_DIR/realsense.log" 2>&1 &

    CAMERA_PID=$!
    echo "RealSense PID: $CAMERA_PID"
}

echo ""
echo "[1/3] Checking RealSense..."

if ros2 topic list 2>/dev/null | grep -Fxq "/camera/camera/color/image_raw"; then
    echo "기존 RealSense 노드 감지 -> 재사용합니다 (새로 띄우지 않음)."
    CAMERA_PID=""
else
    echo "RealSense 노드 없음 -> 새로 시작합니다."
    start_camera

    if ! wait_for_topic "/camera/camera/color/image_raw" 30; then
        echo "WARN: RealSense 기동 실패. 노드를 정리하고 재시도합니다..."
        kill "$CAMERA_PID" 2>/dev/null || true
        sleep 5
        start_camera
        wait_for_topic "/camera/camera/color/image_raw" 30 || {
            echo "ERROR: RealSense 재시도 실패."
            echo "  1) 다른 터미널/컨테이너에 realsense 노드가 남아있는지 확인:"
            echo "       ps aux | grep realsense"
            echo "  2) 카메라 USB 케이블을 뽑았다 다시 꽂아주세요"
            echo "  3) 로그 확인: $LOG_DIR/realsense.log"
            exit 1
        }
    fi
fi

wait_for_topic "/camera/camera/aligned_depth_to_color/image_raw" 30 || exit 1


# ============================================================
# 2. 6D Pose 스트리머 (연속 자동)
# ============================================================

echo ""
echo "[2/3] Starting 6D pose streamer..."

python3 -u camera/pose_streamer_ros2.py \
    --rotation 90_cw \
    --input_mode rgbd \
    $STREAMER_EXTRA_ARGS \
    </dev/null \
    > "$LOG_DIR/foundationpose.log" 2>&1 &

FP_PID=$!

echo "Pose streamer PID: $FP_PID"

wait_for_topic "/foundation_pose/result" 60 || exit 1


# ============================================================
# 3. YOLO (semantic_fp: 명령이 없으면 인식만 수행, NO_COMMAND)
#    run_vcb_command.sh와 동일한 노드/토픽을 사용한다.
#    (/vcb/perception 발행 포함)
# ============================================================

echo ""
echo "[3/3] Starting YOLO perception (semantic_fp)..."

python3 -u yolo/src/main_infer_ros2_semantic_fp.py \
    $YOLO_ARGS \
    </dev/null \
    > "$LOG_DIR/yolo.log" 2>&1 &

YOLO_PID=$!

echo "YOLO PID: $YOLO_PID"


echo ""
echo "============================================================"
echo " PARALLEL TEST RUNNING"
echo "============================================================"
echo ""
echo " 아래에 토픽 요약이 출력됩니다:"
echo "   [PERCEPTION]           : /vcb/perception (2초 간격 요약, diag_hsv 포함)"
echo "   FOUNDATIONPOSE RESULT  : /foundation_pose/result (추정마다)"
echo ""
echo " diagnostic HSV: 명령 없이도 검출된 모든 VCB status에 HSV 수행 (판정 미사용)"
echo ""
echo " 상세 로그:"
echo "   tail -f $LOG_DIR/yolo.log"
echo "   tail -f $LOG_DIR/foundationpose.log"
echo ""
echo " Ctrl+C : 전체 종료"
echo "============================================================"
echo ""

# 토픽 모니터를 포그라운드로 실행 (터미널에 요약 출력)
python3 -u scripts/vcb_topic_monitor.py
