import os
import sys
import cv2
import csv
import json
import time
import yaml
import rclpy
from rclpy.node import Node
from pathlib import Path
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rotation_utils import normalize_rotation, rotate_image
from easyocr_val_data_rule import init_easyocr_reader, run_easyocr_on_crop
from hsv_val_data import run_hsv_on_crop

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

BASE_DIR = PROJECT_ROOT / "yolo"
CLASS_NAMES = ["vcb", "label", "status"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v"}
COMMAND_TOPIC = "/vcb/operator_command"
PERCEPTION_TOPIC = "/vcb/perception"
FOUNDATIONPOSE_REQUEST_TOPIC = "/foundation_pose/request"
FOUNDATIONPOSE_RESULT_TOPIC = "/foundation_pose/result"
CLOSE_NOTICE_TOPIC = "/vcb/status_notice"

# 최초 시도 포함 총 FoundationPose 최대 실행 횟수
FP_MAX_ATTEMPTS = 3

# 실패 후 바로 연속 실행하지 않고
# 최신 카메라 frame이 갱신될 시간을 조금 준다.
FP_RETRY_DELAY_SEC = 1.0

# 작업 진행 가능으로 판단하는 기준 상태.
# 라벨만 입력받고 이 상태인지 아닌지로 분기한다 (OPEN이면 작업 진행,
# CLOSE면 작업을 진행하지 않고 CLOSE_NOTICE_TOPIC으로 알린다).
WORK_READY_STATE = "OPEN"

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_nested(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def is_image_file(path: str):
    if not isinstance(path, str):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTS


def is_video_file(path: str):
    if not isinstance(path, str):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTS



def parse_selected_frames(selected_cfg):
    """
    selected_cfg examples:
      null
      [0, 10, 30]
      "0,10,30"
    """
    if selected_cfg is None:
        return set()

    if isinstance(selected_cfg, list):
        out = set()
        for x in selected_cfg:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out

    if isinstance(selected_cfg, str):
        out = set()
        for token in selected_cfg.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.add(int(token))
            except Exception:
                pass
        return out

    return set()


def should_save_selected_frame(frame_idx, selected_frames, every_n):
    if frame_idx in selected_frames:
        return True

    if isinstance(every_n, int) and every_n > 0 and frame_idx % every_n == 0:
        return True

    return False


def get_boxes_by_class(results, model_names):
    """
    클래스별 모든 bbox 유지

    return:
        {
            "vcb": [
                (x1, y1, x2, y2, conf),
                ...
            ],
            "label": [
                (x1, y1, x2, y2, conf),
                ...
            ],
            "status": [
                (x1, y1, x2, y2, conf),
                ...
            ],
        }
    """

    boxes_by_class = {
        "vcb": [],
        "label": [],
        "status": [],
    }

    if results.boxes is None or len(results.boxes) == 0:
        return boxes_by_class

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        cls_name = model_names[cls_id]

        if cls_name not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = map(int, box)

        boxes_by_class[cls_name].append(
            (x1, y1, x2, y2, float(conf))
        )

    # 디버깅 시 출력 순서를 일정하게 하기 위한 정렬.
    # target 선택 기준으로 사용하는 것은 아님.
    for cls_name in CLASS_NAMES:
        boxes_by_class[cls_name].sort(
            key=lambda box: box[4],
            reverse=True,
        )

    return boxes_by_class


def crop_image(img, box):
    x1, y1, x2, y2, _ = box
    h, w = img.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2].copy()


def draw_result(frame, boxes_by_class, label_results, status_results):
    vis = frame.copy()

    # 모든 YOLO bbox 표시
    for cls_name, box_list in boxes_by_class.items():
        for box in box_list:
            x1, y1, x2, y2, conf = box

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                vis,
                f"{cls_name}:{conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # 각 label bbox의 OCR 결과 표시
    for item in label_results:
        x1, y1, x2, y2, _ = item["box"]
        ocr_text = item["ocr_text"]

        cv2.putText(
            vis,
            f"OCR:{ocr_text}",
            (x1, min(vis.shape[0] - 10, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # 각 status bbox의 HSV 결과 표시
    for item in status_results:
        x1, y1, x2, y2, _ = item["box"]
        hsv_label = item["hsv_label"]

        cv2.putText(
            vis,
            f"HSV:{hsv_label}",
            (x1, min(vis.shape[0] - 10, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # 현재 처리된 개수 확인용
    cv2.putText(
        vis,
        f"OCR labels:{len(label_results)}  HSV status:{len(status_results)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    return vis


# bbox 하나를 CSV 저장용 문자열로 변환
def box_to_str(box):
    if box is None:
        return ""

    x1, y1, x2, y2, conf = box

    return f"{x1},{y1},{x2},{y2},{conf:.4f}"


# bbox 여러 개를 CSV 저장용 문자열로 변환
def boxes_to_str(boxes):
    if not boxes:
        return ""

    return " | ".join(
        box_to_str(box)
        for box in boxes
    )

def get_box_center(box):
    """
    bbox 중심점 반환
    box: (x1, y1, x2, y2, conf)
    """
    x1, y1, x2, y2, _ = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return cx, cy


def point_in_box(point, box, margin_ratio=0.0):
    """
    point가 bbox 내부에 있는지 확인.

    margin_ratio:
        0.0  -> 원래 VCB bbox 그대로 사용
        0.05 -> bbox 폭/높이의 5% 만큼 바깥으로 확장
    """
    px, py = point
    x1, y1, x2, y2, _ = box

    w = x2 - x1
    h = y2 - y1

    margin_x = w * margin_ratio
    margin_y = h * margin_ratio

    return (
        (x1 - margin_x) <= px <= (x2 + margin_x)
        and
        (y1 - margin_y) <= py <= (y2 + margin_y)
    )


def get_normalized_center_distance(box, vcb_box):
    """
    detection bbox 중심과 VCB bbox 중심 사이의 normalized distance.

    여러 VCB bbox에 동시에 포함되는 경우
    어느 VCB에 연결할지 결정하기 위한 보조 기준.
    """
    cx, cy = get_box_center(box)
    vcb_cx, vcb_cy = get_box_center(vcb_box)

    x1, y1, x2, y2, _ = vcb_box

    vcb_w = max(float(x2 - x1), 1.0)
    vcb_h = max(float(y2 - y1), 1.0)

    dx = (cx - vcb_cx) / vcb_w
    dy = (cy - vcb_cy) / vcb_h

    return dx * dx + dy * dy


def find_associated_vcb_index(
    detection_box,
    vcb_boxes,
    margin_ratio=0.0,
):
    """
    detection bbox 중심이 들어가는 VCB를 찾는다.

    후보 VCB가 하나면 바로 선택.
    후보가 여러 개면 normalized center distance가
    가장 작은 VCB를 선택.

    해당 VCB가 없으면 None 반환.
    """
    center = get_box_center(detection_box)

    candidates = []

    for idx, vcb_box in enumerate(vcb_boxes):
        if point_in_box(
            center,
            vcb_box,
            margin_ratio=margin_ratio,
        ):
            distance = get_normalized_center_distance(
                detection_box,
                vcb_box,
            )

            candidates.append(
                (distance, idx)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


def associate_detections_to_vcbs(
    vcb_boxes,
    label_results,
    status_boxes,
    margin_ratio=0.0,
):
    """
    YOLO raw detection과 OCR 결과를 VCB별로 association.

    중요:
    - label은 OCR까지 수행된 label_results 사용
    - status는 HSV 수행 전 raw bbox만 사용
    - HSV는 target VCB가 결정된 후에만 수행
    """

    vcb_instances = []

    for vcb_box in vcb_boxes:
        vcb_instances.append(
            {
                "vcb_box": vcb_box,
                "label_results": [],
                "status_boxes": [],
            }
        )

    unassociated_labels = []
    unassociated_statuses = []

    # ---------------------------------------------------------
    # LABEL -> VCB
    # ---------------------------------------------------------
    for label_result in label_results:
        label_box = label_result["box"]

        vcb_idx = find_associated_vcb_index(
            label_box,
            vcb_boxes,
            margin_ratio=margin_ratio,
        )

        if vcb_idx is None:
            unassociated_labels.append(
                label_result
            )
        else:
            vcb_instances[vcb_idx][
                "label_results"
            ].append(
                label_result
            )

    # ---------------------------------------------------------
    # raw STATUS bbox -> VCB
    # ---------------------------------------------------------
    for status_box in status_boxes:

        vcb_idx = find_associated_vcb_index(
            status_box,
            vcb_boxes,
            margin_ratio=margin_ratio,
        )

        if vcb_idx is None:
            unassociated_statuses.append(
                status_box
            )
        else:
            vcb_instances[vcb_idx][
                "status_boxes"
            ].append(
                status_box
            )

    return (
        vcb_instances,
        unassociated_labels,
        unassociated_statuses,
    )

def normalize_label_text(text):
    if text is None:
        return ""

    return str(text).strip().upper()


def find_target_instance(
    vcb_instances,
    target_label,
):
    """
    작업자 target_label과 OCR 결과를 exact match.

    return:
        target_instance_idx,
        target_instance,
        match_status

    match_status:
        MATCHED
        NOT_FOUND
        AMBIGUOUS
    """

    target_label = normalize_label_text(
        target_label
    )

    matches = []

    for idx, instance in enumerate(
        vcb_instances
    ):
        for label_result in instance[
            "label_results"
        ]:

            ocr_text = normalize_label_text(
                label_result.get(
                    "ocr_text",
                    "",
                )
            )

            if ocr_text == target_label:
                matches.append(
                    (idx, instance)
                )

    # target 없음
    if len(matches) == 0:
        return None, None, "NOT_FOUND"

    # 같은 label을 가진 VCB가 여러 개 검출됨
    # 안전상 자동 선택하지 않음
    if len(matches) > 1:
        return None, None, "AMBIGUOUS"

    idx, instance = matches[0]

    return idx, instance, "MATCHED"


def hsv_label_to_state(hsv_label):
    """
    현재 현장 정의:
        green -> OPEN
        red   -> CLOSE

    나중에 status가 문자 방식으로 바뀌면
    이 계층만 교체하면 됨.
    """

    label = str(
        hsv_label
    ).strip().lower()

    if label == "green":
        return "OPEN"

    if label == "red":
        return "CLOSE"

    return "UNKNOWN"


def analyze_target_status(
    frame_rotated,
    status_boxes,
    hsv_cfg,
):
    """
    target VCB에 association된 status에 대해서만 HSV 수행.

    일반적으로 status_boxes는 1개일 것으로 예상하지만,
    여러 개가 들어와도 처리 가능하게 구성.
    """

    status_results = []
    valid_states = []

    # status detection 자체가 없음
    if not status_boxes:
        return (
            status_results,
            "UNKNOWN",
            "STATUS_NOT_FOUND",
        )

    for status_box in status_boxes:

        status_crop = crop_image(
            frame_rotated,
            status_box,
        )

        if status_crop is None:
            continue

        hsv_out = run_hsv_on_crop(
            status_crop,
            hsv_cfg,
        )

        hsv_label = hsv_out.get(
            "label",
            "N/A",
        )

        hsv_green = hsv_out.get(
            "green_pixels",
            -1,
        )

        hsv_red = hsv_out.get(
            "red_pixels",
            -1,
        )

        current_state = hsv_label_to_state(
            hsv_label
        )

        status_results.append(
            {
                "box": status_box,
                "hsv_label": hsv_label,
                "hsv_green": hsv_green,
                "hsv_red": hsv_red,
                "current_state": current_state,
                "status_text": (
                    f"{hsv_label} "
                    f"(g={hsv_green}, "
                    f"r={hsv_red})"
                ),
            }
        )

        if current_state in (
            "OPEN",
            "CLOSE",
        ):
            valid_states.append(
                current_state
            )

    if not status_results:
        return (
            status_results,
            "UNKNOWN",
            "STATUS_CROP_FAILED",
        )

    unique_states = set(
        valid_states
    )

    # 전부 동일 상태
    if len(unique_states) == 1:
        current_state = next(
            iter(unique_states)
        )

        return (
            status_results,
            current_state,
            "OK",
        )

    # HSV 결과는 나왔지만 OPEN/CLOSE 결정 불가
    if len(unique_states) == 0:
        return (
            status_results,
            "UNKNOWN",
            "HSV_UNKNOWN",
        )

    # 한 VCB에서 서로 다른 상태 결과 발생
    return (
        status_results,
        "UNKNOWN",
        "STATUS_CONFLICT",
    )

def build_instance_summary(
    vcb_instances,
    target_instance_idx=None,
):
    parts = []

    for idx, instance in enumerate(
        vcb_instances
    ):
        label_texts = [
            str(
                item.get(
                    "ocr_text",
                    "N/A",
                )
            )
            for item in instance[
                "label_results"
            ]
        ]

        label_str = (
            ",".join(label_texts)
            if label_texts
            else "-"
        )

        status_count = len(
            instance["status_boxes"]
        )

        target_mark = (
            "*"
            if idx == target_instance_idx
            else ""
        )

        parts.append(
            f"I{idx}{target_mark}:"
            f"L={label_str};"
            f"SB={status_count}"
        )

    return " | ".join(parts)

def process_frame(
    frame_rotated,
    model,
    ocr_reader,
    cfg,
    command_active=False,
    target_label=None,
    command_id=None,
    diagnostic_hsv=False,
):
    yolo_cfg = cfg["yolo"]
    ocr_cfg = cfg["ocr"]
    hsv_cfg = cfg["hsv"]

    conf = yolo_cfg.get("conf", 0.25)
    imgsz = yolo_cfg.get("imgsz", 640)

    t0 = time.time()

    # =========================================================
    # 1. YOLO
    # 모든 VCB / LABEL / STATUS 검출
    # =========================================================
    results = model.predict(
        frame_rotated,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )[0]

    yolo_time = time.time() - t0

    # =========================================================
    # 2-A. 클래스별 모든 bbox 유지
    # =========================================================
    boxes_by_class = get_boxes_by_class(
        results,
        model.names,
    )

    # =========================================================
    # 2-B. 모든 LABEL -> OCR
    #
    # STATUS는 여기서 HSV하지 않는다.
    # =========================================================
    label_results = []

    for label_box in boxes_by_class["label"]:
        label_crop = crop_image(
            frame_rotated,
            label_box,
        )

        if label_crop is None:
            continue

        ocr_out = run_easyocr_on_crop(
            crop=label_crop,
            reader=ocr_reader,
            lang_list=ocr_cfg.get("lang", ["en"]),
            use_gpu=ocr_cfg.get("use_gpu", True),

            resize=get_nested(
                ocr_cfg,
                ["preprocess", "resize"],
                2.0,
            ),

            grayscale=get_nested(
                ocr_cfg,
                ["preprocess", "grayscale"],
                True,
            ),

            detail=get_nested(
                ocr_cfg,
                ["detection", "detail"],
                1,
            ),

            min_confidence=get_nested(
                ocr_cfg,
                ["detection", "min_confidence"],
                0.3,
            ),

            text_selection_method=get_nested(
                ocr_cfg,
                ["text_selection", "method"],
                "top_line",
            ),

            y_weight=get_nested(
                ocr_cfg,
                ["text_selection", "y_weight"],
                1.0,
            ),

            candidate_threshold=get_nested(
                ocr_cfg,
                ["refine", "candidate_threshold"],
                10,
            ),

            similarity_candidate=get_nested(
                ocr_cfg,
                ["refine", "similarity_threshold", "candidate"],
                0.70,
            ),

            similarity_normalized=get_nested(
                ocr_cfg,
                ["refine", "similarity_threshold", "normalized"],
                0.60,
            ),
        )

        label_results.append(
            {
                "box": label_box,
                "ocr_text": ocr_out.get(
                    "refined",
                    "N/A",
                ),
                "ocr_raw": str(
                    ocr_out.get(
                        "raw_joined",
                        "",
                    )
                ),
            }
        )

    # =========================================================
    # 2-C. Spatial association
    #
    # VCB
    # ├─ label_results
    # └─ status_boxes
    #
    # 중요:
    # status는 아직 HSV하지 않은 raw YOLO bbox이다.
    # =========================================================
    (
        vcb_instances,
        unassociated_labels,
        unassociated_statuses,
    ) = associate_detections_to_vcbs(
        vcb_boxes=boxes_by_class["vcb"],
        label_results=label_results,
        status_boxes=boxes_by_class["status"],
        margin_ratio=0.0,
    )

    # =========================================================
    # 2-D. Diagnostic HSV (선택)
    #
    # command 유무와 무관하게 모든 VCB 인스턴스의 status에 대해
    # HSV를 수행한다. 테스트/파이프라인 검증 전용이며,
    # 아래 target selection / decision 로직에는 전혀 사용하지 않는다.
    # (run_vcb_command.sh는 diagnostic_hsv=False로 호출하므로
    #  명령 기반 동작에는 영향이 없다)
    # =========================================================
    diagnostic_status_results = []

    if diagnostic_hsv:
        for instance_idx, instance in enumerate(vcb_instances):
            inst_status_results, _, _ = analyze_target_status(
                frame_rotated,
                instance["status_boxes"],
                hsv_cfg,
            )

            for item in inst_status_results:
                diagnostic_status_results.append(
                    {**item, "vcb_instance_idx": instance_idx}
                )

    # =========================================================
    # 3. Operator command 기반 target selection
    # =========================================================

    # status_results는 이제
    # "모든 status의 HSV 결과"가 아니라
    # "target VCB에 대해 수행한 HSV 결과"만 저장한다.
    status_results = []

    target_found = False
    target_instance_idx = None

    target_match_status = "NO_COMMAND"
    status_reason = "NOT_RUN"

    current_state = "UNKNOWN"
    decision = "NO_COMMAND"

    # FoundationPose를 아직 실제 trigger하지 않는다.
    # temporal stability 전 단계의 후보 flag.
    foundationpose_candidate = False

    # CLOSE 상태 알림(/vcb/status_notice)도 아직 실제 발행하지 않는다.
    # temporal stability 전 단계의 후보 flag.
    close_notice_candidate = False

    # ---------------------------------------------------------
    # 작업자 명령 normalization
    # ---------------------------------------------------------
    target_label_norm = normalize_label_text(
        target_label
    )

    # =========================================================
    # 4. 활성 명령이 있을 때만 target 검색
    # =========================================================
    if command_active:

        # -----------------------------------------------------
        # command validation
        # -----------------------------------------------------
        if not target_label_norm:
            decision = "INVALID_COMMAND"
            target_match_status = "INVALID_COMMAND"

        else:
            (
                target_instance_idx,
                target_instance,
                target_match_status,
            ) = find_target_instance(
                vcb_instances,
                target_label_norm,
            )

            # =================================================
            # Target 없음
            # =================================================
            if target_match_status == "NOT_FOUND":
                decision = "TARGET_NOT_FOUND"

            # =================================================
            # 같은 OCR target이 여러 VCB에서 발견
            # =================================================
            elif target_match_status == "AMBIGUOUS":
                decision = "TARGET_AMBIGUOUS"

            # =================================================
            # Target 정확히 하나 발견
            # =================================================
            elif target_match_status == "MATCHED":
                target_found = True

                # =============================================
                # 5. Target VCB의 STATUS에만 HSV 수행
                # =============================================
                (
                    status_results,
                    current_state,
                    status_reason,
                ) = analyze_target_status(
                    frame_rotated,
                    target_instance["status_boxes"],
                    hsv_cfg,
                )

                # =============================================
                # 6. 현재 상태만으로 작업 가능 여부 판단
                #
                # WORK_READY_STATE(OPEN)이면 작업 진행 -> FoundationPose.
                # 그 외(CLOSE)면 작업을 진행하지 않고 CLOSE 알림만 발행.
                # =============================================

                # status 없음 / HSV unknown / conflict 등
                if status_reason != "OK":
                    decision = status_reason

                elif current_state == WORK_READY_STATE:
                    decision = "READY_FOR_WORK"

                    # 아직 실제 FoundationPose 호출은 하지 않음
                    foundationpose_candidate = True

                else:
                    decision = "BLOCKED_CLOSE"

                    # 아직 실제 알림 발행은 하지 않음
                    close_notice_candidate = True

    # =========================================================
    # 7. Instance summary
    # =========================================================
    instance_summary = build_instance_summary(
        vcb_instances,
        target_instance_idx=target_instance_idx,
    )

    # =========================================================
    # 8. OCR summary
    # =========================================================
    if label_results:
        ocr_text = " | ".join(
            str(item["ocr_text"])
            for item in label_results
        )

        ocr_raw = " | ".join(
            str(item["ocr_raw"])
            for item in label_results
        )
    else:
        ocr_text = "N/A"
        ocr_raw = ""

    # =========================================================
    # 9. Target HSV summary
    #
    # 여기의 status_results는 target에 대해서만 존재한다.
    # =========================================================
    if status_results:
        status_text = " | ".join(
            str(item["status_text"])
            for item in status_results
        )

        hsv_label = " | ".join(
            str(item["hsv_label"])
            for item in status_results
        )

        hsv_green = " | ".join(
            str(item["hsv_green"])
            for item in status_results
        )

        hsv_red = " | ".join(
            str(item["hsv_red"])
            for item in status_results
        )

    else:
        status_text = "N/A"
        hsv_label = ""
        hsv_green = ""
        hsv_red = ""

    # =========================================================
    # 10. Timing
    # =========================================================
    total_time = time.time() - t0

    fps = (
        1.0 / total_time
        if total_time > 0
        else 0.0
    )

    # =========================================================
    # 11. Visualization
    #
    # YOLO bbox는 모두 표시
    # OCR은 모든 label 표시
    # HSV text는 target status만 표시
    # (단, target status가 없고 diagnostic_hsv 결과가 있으면
    #  그것을 대신 표시한다 - command 없는 테스트 화면 확인용)
    # =========================================================
    vis_status_results = (
        status_results
        if status_results
        else diagnostic_status_results
    )

    vis = draw_result(
        frame_rotated,
        boxes_by_class,
        label_results,
        vis_status_results,
    )

    # ---------------------------------------------------------
    # timing
    # ---------------------------------------------------------
    cv2.putText(
        vis,
        (
            f"YOLO:{yolo_time*1000:.1f}ms "
            f"TOTAL:{total_time*1000:.1f}ms "
            f"FPS:{fps:.2f}"
        ),
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    # =========================================================
    # 12. Association debug
    # =========================================================
    instance_debug_parts = []

    for idx, instance in enumerate(
        vcb_instances
    ):
        num_labels = len(
            instance["label_results"]
        )

        # 주의:
        # 기존 status_results가 아니라
        # raw status_boxes 개수를 확인한다.
        num_status = len(
            instance["status_boxes"]
        )

        target_mark = (
            "*"
            if idx == target_instance_idx
            else ""
        )

        instance_debug_parts.append(
            f"I{idx}{target_mark}:"
            f"L{num_labels}/S{num_status}"
        )

    instance_debug = " ".join(
        instance_debug_parts
    )

    association_debug_text = (
        f"INST:{len(vcb_instances)} "
        f"{instance_debug} "
        f"UA-L:{len(unassociated_labels)} "
        f"UA-S:{len(unassociated_statuses)}"
    )

    cv2.putText(
        vis,
        association_debug_text,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
    )

    # =========================================================
    # 13. Operator command 표시
    # =========================================================
    command_debug_text = (
        f"CMD id:{command_id} "
        f"active:{command_active} "
        f"target:{target_label_norm or '-'} "
        f"required:{WORK_READY_STATE}"
    )

    cv2.putText(
        vis,
        command_debug_text,
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
    )

    # =========================================================
    # 14. 현재 상태 / decision 표시
    # =========================================================
    state_debug_text = (
        f"CURRENT:{current_state} "
        f"DECISION:{decision} "
        f"FP_CAND:{foundationpose_candidate} "
        f"CLOSE_CAND:{close_notice_candidate}"
    )

    cv2.putText(
        vis,
        state_debug_text,
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
    )

    # =========================================================
    # 15. Target VCB 강조 표시
    # =========================================================
    if (
        target_instance_idx is not None
        and 0 <= target_instance_idx < len(vcb_instances)
    ):
        target_vcb_box = (
            vcb_instances[
                target_instance_idx
            ]["vcb_box"]
        )

        x1, y1, x2, y2, _ = target_vcb_box

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            4,
        )

        cv2.putText(
            vis,
            "TARGET",
            (x1, max(30, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

    # =========================================================
    # 16. 전체 YOLO bbox 개수
    # =========================================================
    num_boxes = sum(
        len(box_list)
        for box_list in boxes_by_class.values()
    )

    # =========================================================
    # 17. Result
    # =========================================================
    result_row = {
        # -----------------------------------------------------
        # OCR / target HSV 결과
        # -----------------------------------------------------
        "ocr_text": ocr_text,
        "ocr_raw": ocr_raw,

        "status_text": status_text,
        "hsv_label": hsv_label,
        "hsv_green": hsv_green,
        "hsv_red": hsv_red,

        # -----------------------------------------------------
        # 모든 raw YOLO bbox
        # -----------------------------------------------------
        "vcb_boxes": boxes_by_class["vcb"],
        "label_boxes": boxes_by_class["label"],
        "status_boxes": boxes_by_class["status"],

        # -----------------------------------------------------
        # 상세 perception result
        # -----------------------------------------------------
        "label_results": label_results,

        # target에 대해서만 HSV 수행된 결과
        "status_results": status_results,

        # -----------------------------------------------------
        # association
        # -----------------------------------------------------
        "vcb_instances": vcb_instances,
        "unassociated_labels": unassociated_labels,
        "unassociated_statuses": unassociated_statuses,

        "num_vcb_instances": len(
            vcb_instances
        ),

        "num_unassociated_labels": len(
            unassociated_labels
        ),

        "num_unassociated_statuses": len(
            unassociated_statuses
        ),

        "instance_summary": instance_summary,

        # -----------------------------------------------------
        # operator command
        # -----------------------------------------------------
        "command_active": command_active,
        "command_id": command_id,

        "target_label": target_label_norm,

        # -----------------------------------------------------
        # target perception / decision
        # -----------------------------------------------------
        "target_found": target_found,

        "target_instance_idx": (
            target_instance_idx
            if target_instance_idx is not None
            else -1
        ),

        "target_match_status": target_match_status,
        "current_state": current_state,
        "status_reason": status_reason,

        "decision": decision,

        "foundationpose_candidate": (
            foundationpose_candidate
        ),

        "close_notice_candidate": (
            close_notice_candidate
        ),

        # -----------------------------------------------------
        # diagnostic HSV (command 유무와 무관, 판정에 미사용)
        # -----------------------------------------------------
        "diagnostic_hsv_enabled": diagnostic_hsv,
        "diagnostic_status_results": diagnostic_status_results,

        # -----------------------------------------------------
        # timing
        # -----------------------------------------------------
        "yolo_ms": yolo_time * 1000.0,
        "total_ms": total_time * 1000.0,
        "fps": fps,

        "num_boxes": num_boxes,
    }

    return vis, result_row


def init_video_writer(save_video, save_dir, video_name, fps, frame_w, frame_h):
    if not save_video:
        return None, ""

    ensure_dir(save_dir)
    out_path = os.path.join(save_dir, video_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        print(f"[WARN] VideoWriter open failed: {out_path}")
        return None, out_path

    return writer, out_path

def write_csv_header(csv_writer):
    print("[DEBUG] v4 command/target CSV header enabled")

    csv_writer.writerow([
        "frame_idx",
        "source",

        "ocr_text",
        "ocr_raw",

        "status_text",
        "hsv_label",
        "hsv_green",
        "hsv_red",

        "vcb_boxes",
        "label_boxes",
        "status_boxes",

        "num_boxes",

        "num_vcb_instances",
        "num_unassociated_labels",
        "num_unassociated_statuses",
        "instance_summary",

        # operator command
        "command_active",
        "command_id",
        "target_label",

        # target perception
        "target_found",
        "target_instance_idx",
        "target_match_status",
        "current_state",
        "status_reason",
        "decision",

        # FoundationPose / CLOSE 알림 전 단계 후보
        "foundationpose_candidate",
        "close_notice_candidate",

        # diagnostic HSV (command 유무와 무관, 판정에 미사용)
        "diagnostic_hsv_enabled",
        "diagnostic_hsv_summary",

        "yolo_ms",
        "total_ms",
        "fps",
    ])

def write_csv_row(
    csv_writer,
    frame_idx,
    source,
    row,
):
    csv_writer.writerow([
        frame_idx,
        source,

        row["ocr_text"],
        row["ocr_raw"],

        row["status_text"],
        row["hsv_label"],
        row["hsv_green"],
        row["hsv_red"],

        boxes_to_str(
            row["vcb_boxes"]
        ),

        boxes_to_str(
            row["label_boxes"]
        ),

        boxes_to_str(
            row["status_boxes"]
        ),

        row["num_boxes"],

        row["num_vcb_instances"],
        row["num_unassociated_labels"],
        row["num_unassociated_statuses"],
        row["instance_summary"],

        # operator command
        row["command_active"],
        row["command_id"],
        row["target_label"],

        # target perception
        row["target_found"],
        row["target_instance_idx"],
        row["target_match_status"],
        row["current_state"],
        row["status_reason"],
        row["decision"],

        row["foundationpose_candidate"],
        row["close_notice_candidate"],

        row["diagnostic_hsv_enabled"],
        " | ".join(
            f"I{item['vcb_instance_idx']}:{item['hsv_label']}"
            f"(g={item['hsv_green']},r={item['hsv_red']})"
            for item in row["diagnostic_status_results"]
        ),

        f"{row['yolo_ms']:.3f}",
        f"{row['total_ms']:.3f}",
        f"{row['fps']:.3f}",
    ])

def run_ros_stream(
    source,
    model,
    ocr_reader,
    cfg,
    input_rotation,
    csv_writer,
    save_dir,
    save_video,
    video_name,
    save_frames,
    frames_dir,
    save_selected_frames,
    selected_frames,
    selected_frame_every,
    selected_frames_dir,
    enable_gui,
    diagnostic_hsv=False,
):
    class VCBYoloInferNode(Node):
        def __init__(self):
            super().__init__("vcb_yolo_infer")

            self.bridge = CvBridge()
            self.writer = None
            self.frame_idx = 0
            self.enable_gui = enable_gui
            self.diagnostic_hsv = diagnostic_hsv

            # =====================================================
            # Operator command state
            # =====================================================

            # 현재 유효한 작업 명령이 있는지
            self.command_active = False

            # operator_command.py에서 증가시키는 명령 번호
            self.command_id = None

            # 예:
            # 4SW02-01B
            self.target_label = None

            # 명령 생성 시간
            self.command_timestamp = None

            # =====================================================
            # FoundationPose trigger gate
            # =====================================================

            # READY_FOR_WORK가 몇 frame 연속이어야 하는지
            self.fp_stability_frames = 3

            # 현재 연속 안정 frame 수
            self.fp_stable_count = 0

            # 이전 frame에서 확인한 조건
            self.fp_last_key = None

            # 현재 command에 대해 FP trigger를 이미 보냈는지
            self.fp_request_sent = False

            # 어떤 command_id에 대해 trigger를 보냈는지
            self.fp_request_command_id = None

            # =====================================================
            # FoundationPose result / retry state
            # =====================================================

            # 현재 command에서 FoundationPose를 실제 몇 번 요청했는지
            # 최초 요청도 1회로 계산
            self.fp_attempt_count = 0

            # FoundationPose request를 보낸 뒤
            # /foundation_pose/result를 기다리고 있는 상태인지
            self.fp_waiting_result = False

            # 현재 command에 대해 FoundationPose 성공 여부
            self.fp_success = False

            # object_found=False 후 재시도할 때 사용할 ROS timer
            self.fp_retry_timer = None

            # =====================================================
            # CLOSE 알림 gate (BLOCKED_CLOSE 전용, FP gate와 별도)
            #
            # FP gate는 attempt/waiting_result/success 등 재시도 상태가
            # 얽혀 있어 재사용하지 않고, 동일한 "N frame 안정화 후
            # command당 1회" 패턴만 가볍게 복제한다.
            # =====================================================

            # 현재 연속 안정 frame 수
            self.close_notice_stable_count = 0

            # 이전 frame에서 확인한 조건
            self.close_notice_last_key = None

            # 현재 command에 대해 CLOSE 알림을 이미 보냈는지
            self.close_notice_sent = False

            # 어떤 command_id에 대해 알림을 보냈는지
            self.close_notice_command_id = None

            # 가장 최근 YOLO/OCR/HSV 판단 결과 저장
            # retry 직전에 여전히 READY_FOR_WORK인지 확인할 때 사용
            self.latest_perception_row = None

            # =====================================================
            # Camera subscriber
            # =====================================================

            self.subscription = self.create_subscription(
                Image,
                source,
                self.callback,
                1,
            )

            # =====================================================
            # FoundationPose request publisher
            # =====================================================

            self.fp_request_pub = self.create_publisher(
                String,
                FOUNDATIONPOSE_REQUEST_TOPIC,
                10,
            )

            self.get_logger().info(
                "FoundationPose request publisher started: "
                f"{FOUNDATIONPOSE_REQUEST_TOPIC}"
            )

            # =====================================================
            # CLOSE 상태 알림 publisher
            #
            # target이 BLOCKED_CLOSE로 안정되면(N frame) command당
            # 1회만 발행한다. FoundationPose는 트리거하지 않는다.
            # =====================================================

            self.close_notice_pub = self.create_publisher(
                String,
                CLOSE_NOTICE_TOPIC,
                10,
            )

            self.get_logger().info(
                "Close notice publisher started: "
                f"{CLOSE_NOTICE_TOPIC}"
            )

            # =====================================================
            # Perception result publisher (headless 소비자용)
            #
            # 매 frame의 OCR/HSV/instance/decision 요약을
            # JSON(String)으로 발행한다.
            # =====================================================

            self.perception_pub = self.create_publisher(
                String,
                PERCEPTION_TOPIC,
                10,
            )

            self.get_logger().info(
                "Perception publisher started: "
                f"{PERCEPTION_TOPIC}"
            )

            # =====================================================
            # FoundationPose result subscriber
            # =====================================================

            self.fp_result_sub = self.create_subscription(
                String,
                FOUNDATIONPOSE_RESULT_TOPIC,
                self.foundationpose_result_callback,
                10,
            )

            self.get_logger().info(
                "FoundationPose result subscriber started: "
                f"{FOUNDATIONPOSE_RESULT_TOPIC}"
            )

            # =====================================================
            # Operator command subscriber
            #
            # Publisher 쪽과 동일하게 TRANSIENT_LOCAL 사용.
            # 먼저 명령을 입력하고 v3를 나중에 실행해도
            # 가장 최근 명령을 받을 수 있게 함.
            # =====================================================

            command_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )

            self.command_subscription = self.create_subscription(
                String,
                COMMAND_TOPIC,
                self.command_callback,
                command_qos,
            )

            self.get_logger().info(
                f"ROS2 image subscriber started: {source}"
            )

            self.get_logger().info(
                f"Operator command subscriber started: "
                f"{COMMAND_TOPIC}"
            )
        
        def reset_foundationpose_gate(
            self,
            reason="",
        ):
            """
            FoundationPose / CLOSE 알림 관련 상태를
            새 command / clear 시점에 초기화한다.
            """

            # ---------------------------------------------
            # temporal stability / one-shot
            # ---------------------------------------------
            self.fp_stable_count = 0
            self.fp_last_key = None

            self.fp_request_sent = False
            self.fp_request_command_id = None

            # ---------------------------------------------
            # FoundationPose retry state
            # ---------------------------------------------
            self.fp_attempt_count = 0
            self.fp_waiting_result = False
            self.fp_success = False

            # ---------------------------------------------
            # CLOSE 알림 gate
            # ---------------------------------------------
            self.close_notice_stable_count = 0
            self.close_notice_last_key = None
            self.close_notice_sent = False
            self.close_notice_command_id = None

            # 이전 command의 perception 결과를 재사용하지 않도록 제거
            self.latest_perception_row = None

            # 예약된 retry timer가 있으면 제거
            if self.fp_retry_timer is not None:
                try:
                    self.destroy_timer(
                        self.fp_retry_timer
                    )
                except Exception:
                    pass

                self.fp_retry_timer = None

            if reason:
                self.get_logger().info(
                    f"FoundationPose/Close-notice gate reset: {reason}"
                )

        def update_foundationpose_gate(
            self,
            row,
        ):
            """
            process_frame()의 단일-frame 판단을 받아
            temporal stability + one-shot trigger를 관리한다.

            최초 FoundationPose 요청만 이 함수에서 발생한다.
            object_found=False 이후 retry는
            foundationpose_result_callback() -> retry timer 경로에서 처리한다.
            """

            candidate = bool(
                row.get(
                    "foundationpose_candidate",
                    False,
                )
            )

            command_id = row.get(
                "command_id"
            )

            target_label = row.get(
                "target_label",
                "",
            )

            current_state = row.get(
                "current_state",
                "UNKNOWN",
            )

            decision = row.get(
                "decision",
                "",
            )

            # =====================================================
            # READY_FOR_WORK가 아니면 연속 frame count 초기화
            #
            # fp_request_sent는 여기서 초기화하지 않는다.
            # 이미 FoundationPose 단계에 진입한 command의 중복 trigger를
            # 막기 위함이다.
            # =====================================================

            if not candidate:
                self.fp_stable_count = 0
                self.fp_last_key = None

                return {
                    "fp_stable_count": self.fp_stable_count,
                    "fp_stability_frames": self.fp_stability_frames,
                    "fp_ready": False,
                    "fp_triggered_now": False,
                    "fp_request_sent": self.fp_request_sent,
                }

            # =====================================================
            # 이미 이 command에 대해 FoundationPose 단계에 진입했다면
            # 카메라 callback에서는 다시 trigger하지 않는다.
            # retry는 result callback 경로에서만 수행한다.
            # =====================================================

            if (
                self.fp_request_sent
                and
                self.fp_request_command_id == command_id
            ):
                return {
                    "fp_stable_count": self.fp_stable_count,
                    "fp_stability_frames": self.fp_stability_frames,
                    "fp_ready": False,
                    "fp_triggered_now": False,
                    "fp_request_sent": True,
                }

            # =====================================================
            # stability 비교용 key
            # =====================================================

            current_key = (
                command_id,
                target_label,
                current_state,
                decision,
            )

            if current_key == self.fp_last_key:
                self.fp_stable_count += 1

            else:
                self.fp_last_key = current_key
                self.fp_stable_count = 1

            # =====================================================
            # N frame 연속 확인
            # =====================================================

            fp_ready = (
                self.fp_stable_count
                >= self.fp_stability_frames
            )

            fp_triggered_now = False

            # =====================================================
            # 최초 one-shot trigger
            # =====================================================

            if (
                fp_ready
                and
                not self.fp_request_sent
            ):

                publish_ok = (
                    self.handle_foundationpose_trigger(
                        row
                    )
                )

                if publish_ok:
                    # publish 성공 후에만 one-shot latch 설정
                    self.fp_request_sent = True
                    self.fp_request_command_id = command_id
                    fp_triggered_now = True

                else:
                    self.get_logger().error(
                        "FoundationPose trigger was ready, "
                        "but request publish failed."
                    )

            return {
                "fp_stable_count": self.fp_stable_count,
                "fp_stability_frames": self.fp_stability_frames,
                "fp_ready": fp_ready,
                "fp_triggered_now": fp_triggered_now,
                "fp_request_sent": self.fp_request_sent,
            }

        def update_close_notice_gate(
            self,
            row,
        ):
            """
            BLOCKED_CLOSE가 fp_stability_frames만큼 연속되면
            command당 1회만 CLOSE 알림을 발행한다.

            FP gate와 동일한 "N frame 안정화 후 one-shot" 패턴이지만,
            FP gate는 attempt/waiting_result/success 등 재시도 상태가
            얽혀 있어 재사용하지 않고 여기서 가볍게 별도로 관리한다.
            """

            candidate = bool(
                row.get(
                    "close_notice_candidate",
                    False,
                )
            )

            command_id = row.get(
                "command_id"
            )

            target_label = row.get(
                "target_label",
                "",
            )

            decision = row.get(
                "decision",
                "",
            )

            if not candidate:
                self.close_notice_stable_count = 0
                self.close_notice_last_key = None

                return {
                    "close_notice_stable_count": (
                        self.close_notice_stable_count
                    ),
                    "close_notice_sent": self.close_notice_sent,
                }

            if (
                self.close_notice_sent
                and
                self.close_notice_command_id == command_id
            ):
                return {
                    "close_notice_stable_count": (
                        self.close_notice_stable_count
                    ),
                    "close_notice_sent": True,
                }

            current_key = (
                command_id,
                target_label,
                decision,
            )

            if current_key == self.close_notice_last_key:
                self.close_notice_stable_count += 1
            else:
                self.close_notice_last_key = current_key
                self.close_notice_stable_count = 1

            if (
                self.close_notice_stable_count
                >= self.fp_stability_frames
                and not self.close_notice_sent
            ):
                publish_ok = self.publish_close_notice(row)

                if publish_ok:
                    self.close_notice_sent = True
                    self.close_notice_command_id = command_id

            return {
                "close_notice_stable_count": (
                    self.close_notice_stable_count
                ),
                "close_notice_sent": self.close_notice_sent,
            }

        def publish_close_notice(
            self,
            row,
        ):
            """
            CLOSE 상태 알림을 CLOSE_NOTICE_TOPIC으로 1회 발행한다.
            FoundationPose는 트리거하지 않는다.
            """

            command_id = row.get(
                "command_id"
            )

            target_label = str(
                row.get(
                    "target_label",
                    "",
                )
            ).strip().upper()

            if command_id is None or not target_label:
                self.get_logger().error(
                    "Close notice aborted: "
                    "command_id/target_label is empty."
                )
                return False

            if (
                not self.command_active
                or command_id != self.command_id
                or target_label != self.target_label
            ):
                self.get_logger().warn(
                    "Close notice aborted: "
                    "command context changed."
                )
                return False

            payload = {
                "command_id": command_id,
                "target_label": target_label,
                "status": "CLOSE",
                "message": (
                    f"{target_label}: CLOSE 상태입니다. "
                    "작업을 진행할 수 없습니다."
                ),
                "timestamp": (
                    self.get_clock().now().nanoseconds
                    / 1e9
                ),
            }

            msg = String()
            msg.data = json.dumps(
                payload,
                ensure_ascii=False,
            )

            try:
                self.close_notice_pub.publish(
                    msg
                )

            except Exception as exc:
                self.get_logger().error(
                    f"Close notice publish failed: {exc}"
                )
                return False

            self.get_logger().warn(
                "========================================"
            )
            self.get_logger().warn(
                "CLOSE NOTICE PUBLISHED"
            )
            self.get_logger().warn(
                f"command_id={command_id}"
            )
            self.get_logger().warn(
                f"target={target_label}"
            )
            self.get_logger().warn(
                "========================================"
            )

            return True

        def publish_foundationpose_request(
            self,
            row,
            is_retry=False,
        ):
            """
            FoundationPose request를 실제로 publish한다.

            최초 요청과 retry가 공통으로 사용하는 함수.
            최초 시도 포함 총 FP_MAX_ATTEMPTS회까지만 허용한다.
            """

            if self.fp_success:
                self.get_logger().warn(
                    "FoundationPose request skipped: "
                    "pose already succeeded for current command."
                )
                return False

            if self.fp_waiting_result:
                self.get_logger().warn(
                    "FoundationPose request skipped: "
                    "previous result is still pending."
                )
                return False

            if self.fp_attempt_count >= FP_MAX_ATTEMPTS:
                self.get_logger().error(
                    "FoundationPose request aborted: "
                    "maximum attempts reached."
                )
                return False

            command_id = row.get(
                "command_id"
            )

            target_label = str(
                row.get(
                    "target_label",
                    "",
                )
            ).strip().upper()

            current_state = str(
                row.get(
                    "current_state",
                    "",
                )
            ).strip().upper()

            # -----------------------------------------------------
            # 최소 validation
            # -----------------------------------------------------

            if command_id is None:
                self.get_logger().error(
                    "FoundationPose request aborted: "
                    "command_id is None."
                )
                return False

            if not target_label:
                self.get_logger().error(
                    "FoundationPose request aborted: "
                    "target_label is empty."
                )
                return False

            # 작업 진행 조건은 WORK_READY_STATE(OPEN)뿐이다.
            if current_state != WORK_READY_STATE:
                self.get_logger().error(
                    "FoundationPose request aborted: "
                    f"current_state={current_state} != {WORK_READY_STATE}"
                )
                return False

            # 현재 활성 command와 동일한 요청인지 확인
            if (
                not self.command_active
                or command_id != self.command_id
                or target_label != self.target_label
            ):
                self.get_logger().warn(
                    "FoundationPose request aborted: "
                    "command context changed."
                )
                return False

            next_attempt = self.fp_attempt_count + 1

            request = {
                "command_id": command_id,
                "target_label": target_label,
                "current_state": current_state,
                "attempt": next_attempt,
                "timestamp": (
                    self.get_clock().now().nanoseconds
                    / 1e9
                ),
            }

            msg = String()
            msg.data = json.dumps(
                request,
                ensure_ascii=False,
            )

            try:
                self.fp_request_pub.publish(
                    msg
                )

            except Exception as exc:
                self.get_logger().error(
                    "FoundationPose request publish failed: "
                    f"{exc}"
                )
                return False

            # publish가 성공한 경우에만 attempt 증가
            self.fp_attempt_count = next_attempt
            self.fp_waiting_result = True

            request_type = (
                "RETRY"
                if is_retry
                else "INITIAL"
            )

            self.get_logger().warn(
                "========================================"
            )
            self.get_logger().warn(
                f"FOUNDATIONPOSE {request_type} REQUEST PUBLISHED"
            )
            self.get_logger().warn(
                f"attempt={self.fp_attempt_count}/{FP_MAX_ATTEMPTS}"
            )
            self.get_logger().warn(
                f"command_id={command_id}"
            )
            self.get_logger().warn(
                f"target={target_label}"
            )
            self.get_logger().warn(
                f"current={current_state}"
            )
            self.get_logger().warn(
                "========================================"
            )

            return True

        def handle_foundationpose_trigger(
            self,
            row,
        ):
            """
            Temporal stability 조건을 최초로 만족했을 때 호출된다.
            """

            return self.publish_foundationpose_request(
                row,
                is_retry=False,
            )

        def foundationpose_result_callback(
            self,
            msg,
        ):
            """
            /foundation_pose/result 수신.

            object_found=True:
                현재 command의 FoundationPose 성공.

            object_found=False:
                남은 attempt가 있으면 retry를 예약한다.
            """

            try:
                result = json.loads(
                    msg.data
                )

            except json.JSONDecodeError as exc:
                self.get_logger().error(
                    "Invalid FoundationPose result JSON: "
                    f"{exc}"
                )
                return

            result_command_id = result.get(
                "command_id"
            )

            result_target_label = str(
                result.get(
                    "target_label",
                    "",
                )
            ).strip().upper()

            object_found = bool(
                result.get(
                    "object_found",
                    False,
                )
            )

            confidence = result.get(
                "confidence",
                None,
            )

            # -----------------------------------------------------
            # stale / unrelated result 차단
            # -----------------------------------------------------

            if not self.command_active:
                self.get_logger().warn(
                    "Ignoring FoundationPose result: "
                    "no active command."
                )
                return

            if result_command_id != self.command_id:
                self.get_logger().warn(
                    "Ignoring FoundationPose result for different command: "
                    f"result_id={result_command_id}, "
                    f"current_id={self.command_id}"
                )
                return

            if result_target_label != self.target_label:
                self.get_logger().warn(
                    "Ignoring FoundationPose result for different target: "
                    f"result_target={result_target_label}, "
                    f"current_target={self.target_label}"
                )
                return

            if not self.fp_request_sent:
                self.get_logger().warn(
                    "Ignoring FoundationPose result: "
                    "current command has not entered FoundationPose stage."
                )
                return

            if not self.fp_waiting_result:
                self.get_logger().warn(
                    "Ignoring duplicate/unexpected FoundationPose result: "
                    f"command_id={result_command_id}, "
                    f"target={result_target_label}"
                )
                return

            # 현재 attempt의 결과를 정상 수신함
            self.fp_waiting_result = False

            # =====================================================
            # SUCCESS
            # =====================================================

            if object_found:
                self.fp_success = True

                # 혹시 예약 timer가 남아 있으면 제거
                if self.fp_retry_timer is not None:
                    try:
                        self.destroy_timer(
                            self.fp_retry_timer
                        )
                    except Exception:
                        pass
                    self.fp_retry_timer = None

                self.get_logger().info(
                    "========================================"
                )
                self.get_logger().info(
                    "FOUNDATIONPOSE SUCCESS"
                )
                self.get_logger().info(
                    f"attempt={self.fp_attempt_count}/{FP_MAX_ATTEMPTS}"
                )
                self.get_logger().info(
                    f"command_id={result_command_id}, "
                    f"target={result_target_label}, "
                    f"confidence={confidence}"
                )
                self.get_logger().info(
                    "========================================"
                )
                return

            # =====================================================
            # object_found=False
            # =====================================================

            self.get_logger().warn(
                "FoundationPose object not found: "
                f"attempt={self.fp_attempt_count}/{FP_MAX_ATTEMPTS}, "
                f"command_id={result_command_id}, "
                f"target={result_target_label}"
            )

            if self.fp_attempt_count >= FP_MAX_ATTEMPTS:
                self.get_logger().error(
                    "========================================"
                )
                self.get_logger().error(
                    "FOUNDATIONPOSE FAILED: maximum attempts reached"
                )
                self.get_logger().error(
                    f"command_id={result_command_id}, "
                    f"target={result_target_label}"
                )
                self.get_logger().error(
                    "========================================"
                )
                return

            self.schedule_foundationpose_retry()

        def schedule_foundationpose_retry(
            self,
        ):
            """
            object_found=False 이후 FP_RETRY_DELAY_SEC만큼 기다렸다가
            최신 perception 상태를 확인한 뒤 1회 retry한다.
            """

            if self.fp_success:
                return

            if self.fp_waiting_result:
                return

            if self.fp_attempt_count >= FP_MAX_ATTEMPTS:
                return

            # 중복 timer 방지
            if self.fp_retry_timer is not None:
                try:
                    self.destroy_timer(
                        self.fp_retry_timer
                    )
                except Exception:
                    pass
                self.fp_retry_timer = None

            self.get_logger().warn(
                "FoundationPose retry scheduled: "
                f"next attempt={self.fp_attempt_count + 1}/"
                f"{FP_MAX_ATTEMPTS}, "
                f"delay={FP_RETRY_DELAY_SEC:.1f}s"
            )

            self.fp_retry_timer = self.create_timer(
                FP_RETRY_DELAY_SEC,
                self.retry_foundationpose_once,
            )

        def retry_foundationpose_once(
            self,
        ):
            """
            ROS timer callback.
            반복 timer를 즉시 제거한 뒤 조건이 유효할 때만
            FoundationPose inference를 1회 재요청한다.
            """

            timer = self.fp_retry_timer
            self.fp_retry_timer = None

            if timer is not None:
                try:
                    self.destroy_timer(
                        timer
                    )
                except Exception:
                    pass

            if self.fp_success:
                return

            if self.fp_waiting_result:
                self.get_logger().warn(
                    "FoundationPose retry skipped: "
                    "previous result is still pending."
                )
                return

            if self.fp_attempt_count >= FP_MAX_ATTEMPTS:
                self.get_logger().error(
                    "FoundationPose retry skipped: "
                    "maximum attempts reached."
                )
                return

            if not self.command_active:
                self.get_logger().warn(
                    "FoundationPose retry aborted: "
                    "command is no longer active."
                )
                return

            row = self.latest_perception_row

            if row is None:
                self.get_logger().warn(
                    "FoundationPose retry aborted: "
                    "no recent perception result."
                )
                return

            # -----------------------------------------------------
            # 여전히 동일한 작업인지 확인
            # -----------------------------------------------------

            if row.get("command_id") != self.command_id:
                self.get_logger().warn(
                    "FoundationPose retry aborted: command changed."
                )
                return

            row_target_label = str(
                row.get(
                    "target_label",
                    "",
                )
            ).strip().upper()

            if row_target_label != self.target_label:
                self.get_logger().warn(
                    "FoundationPose retry aborted: target changed."
                )
                return

            # -----------------------------------------------------
            # 현재도 작업이 필요한 상태인지 확인
            # -----------------------------------------------------

            if not bool(
                row.get(
                    "foundationpose_candidate",
                    False,
                )
            ):
                self.get_logger().warn(
                    "FoundationPose retry aborted: "
                    "latest perception is no longer READY_FOR_WORK "
                    f"(decision={row.get('decision')})."
                )
                return

            publish_ok = self.publish_foundationpose_request(
                row,
                is_retry=True,
            )

            if not publish_ok:
                self.get_logger().error(
                    "FoundationPose retry publish failed."
                )

        def publish_perception(self, row):
            """
            매 frame의 perception 요약을 /vcb/perception 에 JSON으로 발행.

            headless 운영 시 하위 시스템이 OCR/HSV/decision 상태를
            실시간으로 구독할 수 있게 한다.
            """

            def _box(b):
                if b is None:
                    return None
                try:
                    return [int(v) for v in b]
                except (TypeError, ValueError):
                    return None

            instances = []
            for idx, inst in enumerate(
                row.get("vcb_instances") or []
            ):
                labels = [
                    str(item.get("ocr_text", ""))
                    for item in inst.get("label_results", [])
                ]
                instances.append({
                    "index": idx,
                    "labels": labels,
                    "vcb_box": _box(inst.get("vcb_box")),
                    "num_status_boxes": len(
                        inst.get("status_boxes", [])
                    ),
                })

            payload = {
                "timestamp": (
                    self.get_clock().now().nanoseconds / 1e9
                ),
                "frame_idx": int(self.frame_idx),

                # perception (OCR / HSV)
                "num_vcb_instances": int(
                    row.get("num_vcb_instances", 0)
                ),
                "instances": instances,
                "ocr_text": row.get("ocr_text"),
                "hsv_label": row.get("hsv_label"),
                "current_state": row.get("current_state"),

                # operator command / decision
                "command_active": bool(
                    row.get("command_active", False)
                ),
                "command_id": row.get("command_id"),
                "target_label": row.get("target_label"),
                "target_found": bool(
                    row.get("target_found", False)
                ),
                "decision": row.get("decision"),

                # FoundationPose gate 상태
                "fp_stable_count": int(
                    row.get("fp_stable_count", 0)
                ),
                "fp_request_sent": bool(
                    row.get("fp_request_sent", False)
                ),

                # CLOSE 알림 gate 상태
                "close_notice_stable_count": int(
                    row.get("close_notice_stable_count", 0)
                ),
                "close_notice_sent": bool(
                    row.get("close_notice_sent", False)
                ),

                # diagnostic HSV (command 유무와 무관, 판정에 미사용)
                "diagnostic_hsv_enabled": bool(
                    row.get("diagnostic_hsv_enabled", False)
                ),
                "diagnostic_status": [
                    {
                        "vcb_instance_idx": item.get("vcb_instance_idx"),
                        "hsv_label": item.get("hsv_label"),
                        "current_state": item.get("current_state"),
                    }
                    for item in (row.get("diagnostic_status_results") or [])
                ],

                # timing
                "fps": round(float(row.get("fps", 0.0)), 2),
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.perception_pub.publish(msg)

        def command_callback(self, msg):
            """
            /vcb/operator_command 에서 작업자 명령 수신.

            예상 JSON:
            {
                "active": true,
                "command_id": 1,
                "target_label": "4SW02-01B",
                "timestamp": 1234567890.0
            }
            """

            try:
                data = json.loads(msg.data)

            except json.JSONDecodeError as exc:
                self.get_logger().error(
                    f"Invalid operator command JSON: {exc}"
                )
                return

            # ---------------------------------------------------------
            # active 필드 확인
            # ---------------------------------------------------------

            active = bool(
                data.get("active", False)
            )

            command_id = data.get(
                "command_id",
                None,
            )

            timestamp = data.get(
                "timestamp",
                None,
            )

            # =========================================================
            # CLEAR / inactive command
            # =========================================================

            if not active:
                self.reset_foundationpose_gate(
                    reason=(
                        f"command cleared "
                        f"id={command_id}"
                    )
                )

                self.command_active = False
                self.command_id = command_id
                self.target_label = None
                self.command_timestamp = timestamp

                self.get_logger().info(
                    f"Operator command cleared "
                    f"(command_id={command_id})"
                )

                return

            # =========================================================
            # Active command
            # =========================================================

            target_label = str(
                data.get("target_label", "")
            ).strip().upper()

            # ---------------------------------------------------------
            # validation
            # ---------------------------------------------------------

            if not target_label:
                self.get_logger().error(
                    "Received active command "
                    "with empty target_label."
                )
                return

            # ---------------------------------------------------------
            # 정상 명령 저장
            # ---------------------------------------------------------

            # 새로운 command이면 FP/CLOSE 알림 gate 초기화
            if (
                command_id != self.command_id
                or target_label != self.target_label
            ):
                self.reset_foundationpose_gate(
                    reason=(
                        f"new command "
                        f"id={command_id}, "
                        f"target={target_label}"
                    )
                )

            self.command_active = True
            self.command_id = command_id
            self.target_label = target_label
            self.command_timestamp = timestamp

            self.get_logger().info(
                "Operator command received: "
                f"id={self.command_id}, "
                f"target={self.target_label}"
            )

        def callback(self, msg):
            try:
                frame_original = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )
                frame_rotated = rotate_image(frame_original, input_rotation)

                # ==============================================
                # command debug
                # ==============================================
                if self.frame_idx % 30 == 0:
                    self.get_logger().info(
                        "Current command state: "
                        f"active={self.command_active}, "
                        f"id={self.command_id}, "
                        f"target={self.target_label}"
                    )

                vis, row = process_frame(
                    frame_rotated,
                    model,
                    ocr_reader,
                    cfg,
                    command_active=self.command_active,
                    target_label=self.target_label,
                    command_id=self.command_id,
                    diagnostic_hsv=self.diagnostic_hsv,
                )

                # FoundationPose retry 판단에 사용할
                # 가장 최근 perception 결과 저장
                self.latest_perception_row = row.copy()

                # =====================================================
                # Temporal stability + one-shot gate
                # =====================================================

                fp_gate_result = (
                    self.update_foundationpose_gate(
                        row
                    )
                )

                row.update(
                    fp_gate_result
                )

                close_notice_gate_result = (
                    self.update_close_notice_gate(
                        row
                    )
                )

                row.update(
                    close_notice_gate_result
                )

                # =====================================================
                # Perception 결과 토픽 발행 (headless 소비자용)
                # 발행 실패가 인식 파이프라인을 멈추지 않도록 보호
                # =====================================================
                try:
                    self.publish_perception(row)
                except Exception as exc:
                    self.get_logger().warn(
                        f"Perception publish failed: {exc}"
                    )

                # =====================================================
                # Gate 상태 화면 표시
                # =====================================================

                fp_gate_text = (
                    f"FP_STABLE:"
                    f"{row['fp_stable_count']}/"
                    f"{row['fp_stability_frames']} "
                    f"READY:{row['fp_ready']} "
                    f"SENT:{row['fp_request_sent']}"
                )

                cv2.putText(
                    vis,
                    fp_gate_text,
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                )

                fp_retry_text = (
                    f"FP_ATTEMPT:{self.fp_attempt_count}/{FP_MAX_ATTEMPTS} "
                    f"WAIT:{self.fp_waiting_result} "
                    f"SUCCESS:{self.fp_success}"
                )

                cv2.putText(
                    vis,
                    fp_retry_text,
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                )

                close_notice_text = (
                    f"CLOSE_STABLE:"
                    f"{row['close_notice_stable_count']}/"
                    f"{self.fp_stability_frames} "
                    f"SENT:{row['close_notice_sent']}"
                )

                cv2.putText(
                    vis,
                    close_notice_text,
                    (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                )

                write_csv_row(csv_writer, self.frame_idx, source, row)

                h, w = vis.shape[:2]

                if self.writer is None and save_video:
                    writer_fps = 20.0

                    tmp_writer, writer_path = init_video_writer(
                        save_video=save_video,
                        save_dir=save_dir,
                        video_name=video_name,
                        fps=writer_fps,
                        frame_w=w,
                        frame_h=h,
                    )

                    self.writer = tmp_writer

                    if self.writer is not None:
                        self.get_logger().info(f"Saving ROS2 video: {writer_path}")

                if self.writer is not None:
                    self.writer.write(vis)

                if save_frames:
                    out_path = os.path.join(frames_dir, f"frame_{self.frame_idx:06d}.jpg")
                    cv2.imwrite(out_path, vis)

                if save_selected_frames and should_save_selected_frame(
                    self.frame_idx, selected_frames, selected_frame_every
                ):
                    selected_path = os.path.join(
                        selected_frames_dir,
                        f"frame_{self.frame_idx:06d}.jpg",
                    )
                    cv2.imwrite(selected_path, vis)

                if self.enable_gui:
                    try:
                        cv2.imshow("main_infer_ros2", vis)
                        key = cv2.waitKey(1) & 0xFF

                        if key == 27 or key == ord("q"):
                            self.get_logger().info("Exit key pressed.")
                            rclpy.shutdown()

                    except cv2.error as e:
                        self.get_logger().warn(f"GUI unavailable: {e}")
                        self.enable_gui = False

                self.frame_idx += 1

            except Exception as e:
                self.get_logger().error(f"callback error: {e}")

        def cleanup(self):
            if self.fp_retry_timer is not None:
                try:
                    self.destroy_timer(
                        self.fp_retry_timer
                    )
                except Exception:
                    pass
                self.fp_retry_timer = None

            if self.writer is not None:
                self.writer.release()

    rclpy.init(args=None)
    node = VCBYoloInferNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main():
    print(f"[DEBUG] Running script: {__file__}")
    config_path = BASE_DIR / "configs" / "infer_config.yaml"
    cfg = load_config(config_path)

    # required / existing config
    model_path = str(BASE_DIR / cfg["yolo"]["model_path"])
    input_mode = get_nested(cfg, ["input", "mode"], "opencv")
    source = cfg["input"]["source"]
    input_rotation = normalize_rotation(
        get_nested(cfg, ["input", "rotation"], "none")
    )

    if input_mode == "opencv":
        if isinstance(source, str):
            if not source.startswith("/dev/video"):
                source = str(BASE_DIR / source)
    elif input_mode == "ros":
        # ROS topic은 절대경로처럼 보이지만 파일 경로가 아니므로 BASE_DIR을 붙이면 안 됨
        pass
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    save_dir = str(BASE_DIR / cfg["output"]["save_dir"])

    ocr_cfg = cfg["ocr"]

    # optional output config (yaml에 없어도 기본값으로 동작)
    is_camera_source = isinstance(source, str) and source.startswith("/dev/video")
    enable_gui = get_nested(cfg, ["output", "enable_gui"], is_camera_source)

    # CLI 플래그가 yaml 설정보다 우선한다 (.sh에서 제어 용이)
    if "--headless" in sys.argv:
        enable_gui = False
        print("[INFO] --headless: GUI disabled")
    elif "--gui" in sys.argv:
        enable_gui = True
        print("[INFO] --gui: GUI enabled")

    diagnostic_hsv = "--diagnostic-hsv" in sys.argv
    if diagnostic_hsv:
        print(
            "[INFO] --diagnostic-hsv: HSV will run on every detected VCB "
            "instance regardless of operator command "
            "(diagnostic only, not used for decision)"
        )

    save_video = get_nested(cfg, ["output", "save_video"], True)
    save_frames = get_nested(cfg, ["output", "save_frames"], False)
    save_selected_frames = get_nested(cfg, ["output", "save_selected_frames"], False)
    selected_frames = parse_selected_frames(get_nested(cfg, ["output", "selected_frames"], []))
    selected_frame_every = get_nested(cfg, ["output", "selected_frame_every"], 0)

    video_name = get_nested(cfg, ["output", "video_name"], "result.mp4")
    csv_name = get_nested(cfg, ["output", "csv_name"], "result.csv")
    image_name = get_nested(cfg, ["output", "image_name"], "result_image.jpg")
    frames_dir_name = get_nested(cfg, ["output", "frames_dir_name"], "frames")
    selected_frames_dir_name = get_nested(cfg, ["output", "selected_frames_dir_name"], "selected_frames")

    ensure_dir(save_dir)

    frames_dir = os.path.join(save_dir, frames_dir_name)
    selected_frames_dir = os.path.join(save_dir, selected_frames_dir_name)

    if save_frames:
        ensure_dir(frames_dir)

    if save_selected_frames:
        ensure_dir(selected_frames_dir)

    csv_path = os.path.join(save_dir, csv_name)
    csv_fp = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    write_csv_header(csv_writer)

    print(f"[INFO] Input mode: {input_mode}")
    print(f"[INFO] Input source: {source}")
    print(f"[INFO] Input rotation: {input_rotation}")
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    print("[INFO] Initializing EasyOCR reader...")
    ocr_reader = init_easyocr_reader(
        lang_list=ocr_cfg.get("lang", ["en"]),
        use_gpu=ocr_cfg.get("use_gpu", True)
    )

    writer = None
    writer_path = ""
    processed_count = 0

    try:
        # case 0) ros
        if input_mode == "ros":
            run_ros_stream(
                source=source,
                model=model,
                ocr_reader=ocr_reader,
                cfg=cfg,
                input_rotation=input_rotation,
                csv_writer=csv_writer,
                save_dir=save_dir,
                save_video=save_video,
                video_name=video_name,
                save_frames=save_frames,
                frames_dir=frames_dir,
                save_selected_frames=save_selected_frames,
                selected_frames=selected_frames,
                selected_frame_every=selected_frame_every,
                selected_frames_dir=selected_frames_dir,
                enable_gui=enable_gui,
                diagnostic_hsv=diagnostic_hsv,
            )

            return
        # case 1) single image
        if is_image_file(source):
            print(f"[INFO] Image input detected: {source}")
            frame_original = cv2.imread(source, cv2.IMREAD_COLOR)
            if frame_original is None:
                raise RuntimeError(f"Cannot read image: {source}")

            frame_rotated = rotate_image(frame_original, input_rotation)
            vis, row = process_frame(frame_rotated, model, ocr_reader, cfg)
            write_csv_row(csv_writer, 0, source, row)
            processed_count += 1

            # single image는 annotated image 저장
            image_out_path = os.path.join(save_dir, image_name)
            cv2.imwrite(image_out_path, vis)
            print(f"[INFO] Saved annotated image: {image_out_path}")

            # 필요하면 selected frame에도 저장
            if save_selected_frames and should_save_selected_frame(
                0, selected_frames, selected_frame_every
            ):
                selected_path = os.path.join(selected_frames_dir, "frame_000000.jpg")
                cv2.imwrite(selected_path, vis)
                print(f"[INFO] Saved selected image: {selected_path}")

            if enable_gui:
                try:
                    cv2.imshow("main_infer", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"[WARN] GUI unavailable, skipping imshow: {e}")

        # case 2) video file or camera
        else:
            input_type = "camera"
            if is_video_file(source):
                input_type = "video_file"

            print(f"[INFO] Opening {input_type}: {source}")
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {source}")

            src_fps = cap.get(cv2.CAP_PROP_FPS)

            if src_fps is None or src_fps <= 1e-6 or src_fps != src_fps:
                src_fps = 20.0

            # 90도 회전 시 출력 폭/높이가 바뀌므로 VideoWriter는
            # 첫 번째 회전 프레임 처리 후 실제 출력 크기로 생성한다.
            frame_idx = 0

            while True:
                ret, frame_original = cap.read()

                if not ret:
                    print("[INFO] End of stream or failed frame read.")
                    break

                frame_rotated = rotate_image(frame_original, input_rotation)
                vis, row = process_frame(frame_rotated, model, ocr_reader, cfg)
                write_csv_row(csv_writer, frame_idx, str(source), row)
                processed_count += 1

                if writer is None and save_video:
                    out_h, out_w = vis.shape[:2]
                    writer, writer_path = init_video_writer(
                        save_video=save_video,
                        save_dir=save_dir,
                        video_name=video_name,
                        fps=src_fps,
                        frame_w=out_w,
                        frame_h=out_h,
                    )
                    if writer is not None:
                        print(f"[INFO] Saving annotated video: {writer_path}")

                if writer is not None:
                    writer.write(vis)

                if save_frames:
                    out_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(out_path, vis)

                if save_selected_frames and should_save_selected_frame(
                    frame_idx, selected_frames, selected_frame_every
                ):
                    selected_path = os.path.join(
                        selected_frames_dir, f"frame_{frame_idx:06d}.jpg"
                    )
                    cv2.imwrite(selected_path, vis)

                if enable_gui:
                    try:
                        cv2.imshow("main_infer", vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            print("[INFO] ESC pressed. Stopping.")
                            break
                    except cv2.error as e:
                        print(f"[WARN] GUI unavailable, disabling imshow: {e}")
                        enable_gui = False

                frame_idx += 1

            cap.release()

    finally:
        if writer is not None:
            writer.release()
        csv_fp.close()
        if enable_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"[INFO] Done. Processed frames: {processed_count}")
    print(f"[INFO] CSV saved: {csv_path}")
    if writer_path:
        print(f"[INFO] Video saved: {writer_path}")


if __name__ == "__main__":
    main()