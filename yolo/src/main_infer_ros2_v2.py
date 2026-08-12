import os
import sys
import cv2
import csv
import time
import yaml
import rclpy
from rclpy.node import Node
from pathlib import Path
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rotation_utils import normalize_rotation, rotate_image
from easyocr_val_data_rule import init_easyocr_reader, run_easyocr_on_crop
from hsv_val_data import run_hsv_on_crop

BASE_DIR = PROJECT_ROOT / "yolo"
CLASS_NAMES = ["vcb", "label", "status"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v"}


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
    status_results,
    margin_ratio=0.0,
):
    """
    label/status 결과를 VCB bbox와 공간적으로 association.

    중요한 점:
    - label/status evidence 자체는 삭제하지 않는다.
    - VCB에 연결되지 않은 결과는 unassociated_* 로 유지한다.
    - 아직 target label 선택은 하지 않는다.

    return:
        vcb_instances,
        unassociated_labels,
        unassociated_statuses
    """

    vcb_instances = []

    # VCB별 instance 기본 생성
    for vcb_box in vcb_boxes:
        vcb_instances.append(
            {
                "vcb_box": vcb_box,

                # 지금은 단일 결과를 강제하지 않고
                # candidate list 형태로 유지
                "label_results": [],
                "status_results": [],
            }
        )

    unassociated_labels = []
    unassociated_statuses = []

    # ---------------------------------------------------------
    # label -> VCB association
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
    # status -> VCB association
    # ---------------------------------------------------------
    for status_result in status_results:
        status_box = status_result["box"]

        vcb_idx = find_associated_vcb_index(
            status_box,
            vcb_boxes,
            margin_ratio=margin_ratio,
        )

        if vcb_idx is None:
            unassociated_statuses.append(
                status_result
            )
        else:
            vcb_instances[vcb_idx][
                "status_results"
            ].append(
                status_result
            )

    return (
        vcb_instances,
        unassociated_labels,
        unassociated_statuses,
    )


def build_instance_summary(vcb_instances):
    """
    CSV/debug용 간단한 instance 요약 문자열.
    """

    parts = []

    for idx, instance in enumerate(vcb_instances):
        label_texts = [
            str(item.get("ocr_text", "N/A"))
            for item in instance["label_results"]
        ]

        status_texts = [
            str(item.get("hsv_label", "N/A"))
            for item in instance["status_results"]
        ]

        label_str = (
            ",".join(label_texts)
            if label_texts
            else "-"
        )

        status_str = (
            ",".join(status_texts)
            if status_texts
            else "-"
        )

        parts.append(
            f"I{idx}:L={label_str};S={status_str}"
        )

    return " | ".join(parts)

def process_frame(frame_rotated, model, ocr_reader, cfg):
    yolo_cfg = cfg["yolo"]
    ocr_cfg = cfg["ocr"]
    hsv_cfg = cfg["hsv"]

    conf = yolo_cfg.get("conf", 0.25)
    imgsz = yolo_cfg.get("imgsz", 640)

    t0 = time.time()

    results = model.predict(
        frame_rotated,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )[0]

    yolo_time = time.time() - t0

    # ---------------------------------------------------------
    # 2-A
    # 클래스별 최고 bbox 하나가 아니라 모든 bbox 유지
    # ---------------------------------------------------------
    boxes_by_class = get_boxes_by_class(
        results,
        model.names,
    )

    # ---------------------------------------------------------
    # 2-B
    # 각 label / status 처리 결과 저장
    # ---------------------------------------------------------
    label_results = []
    status_results = []

    # =========================================================
    # LABEL -> OCR
    # =========================================================
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
    # STATUS -> HSV
    #
    # OCR 결과와 완전히 독립적
    # =========================================================
    for status_box in boxes_by_class["status"]:
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

        status_results.append(
            {
                "box": status_box,
                "hsv_label": hsv_label,
                "hsv_green": hsv_green,
                "hsv_red": hsv_red,
                "status_text": (
                    f"{hsv_label} "
                    f"(g={hsv_green}, r={hsv_red})"
                ),
            }
        )

    # =========================================================
    # 2-C prototype
    # VCB <-> label/status spatial association
    # =========================================================

    vcb_instances, unassociated_labels, unassociated_statuses = (
        associate_detections_to_vcbs(
            vcb_boxes=boxes_by_class["vcb"],
            label_results=label_results,
            status_results=status_results,

            # 현재 실제 CSV에서는 center containment가 안정적이므로
            # 일단 margin 없이 시작
            margin_ratio=0.0,
        )
    )

    instance_summary = build_instance_summary(
        vcb_instances
    )

    # ---------------------------------------------------------
    # 기존 CSV / 출력 구조와 최대한 비슷하게 유지하기 위해
    # 여러 결과를 문자열로 묶음
    #
    # 실제 target association에서는 label_results /
    # status_results 리스트 자체를 사용할 예정
    # ---------------------------------------------------------

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

    total_time = time.time() - t0

    fps = (
        1.0 / total_time
        if total_time > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # 모든 bbox + 각각의 OCR/HSV 결과 표시
    # ---------------------------------------------------------
    vis = draw_result(
        frame_rotated,
        boxes_by_class,
        label_results,
        status_results,
    )

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

    # ---------------------------------------------------------
    # association debug 표시
    # ---------------------------------------------------------

    instance_debug_parts = []

    for idx, instance in enumerate(vcb_instances):
        num_labels = len(
            instance["label_results"]
        )

        num_status = len(
            instance["status_results"]
        )

        instance_debug_parts.append(
            f"I{idx}:L{num_labels}/S{num_status}"
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

    # 실제 처리된 label/status bbox
    processed_label_boxes = [
        item["box"]
        for item in label_results
    ]

    processed_status_boxes = [
        item["box"]
        for item in status_results
    ]

    # 전체 YOLO bbox 개수
    num_boxes = sum(
        len(box_list)
        for box_list in boxes_by_class.values()
    )

    result_row = {
        # 기존 결과 필드
        "ocr_text": ocr_text,
        "ocr_raw": ocr_raw,
        "status_text": status_text,
        "hsv_label": hsv_label,
        "hsv_green": hsv_green,
        "hsv_red": hsv_red,

        # multiple bbox
        "vcb_boxes": boxes_by_class["vcb"],
        "label_boxes": processed_label_boxes,
        "status_boxes": processed_status_boxes,

        # 나중 target association에서 바로 쓸 수 있는 구조
        "label_results": label_results,
        "status_results": status_results,

        # association result
        "vcb_instances": vcb_instances,
        "unassociated_labels": unassociated_labels,
        "unassociated_statuses": unassociated_statuses,

        "num_vcb_instances": len(vcb_instances),
        "num_unassociated_labels": len(unassociated_labels),
        "num_unassociated_statuses": len(unassociated_statuses),

        "instance_summary": instance_summary,

        # timing
        "yolo_ms": yolo_time * 1000.0,
        "total_ms": total_time * 1000.0,
        "fps": fps,

        # 이제 클래스 수가 아니라 실제 bbox 수
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
    print("[DEBUG] association CSV header enabled")
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
):
    class VCBYoloInferNode(Node):
        def __init__(self):
            super().__init__("vcb_yolo_infer")

            self.bridge = CvBridge()
            self.writer = None
            self.frame_idx = 0
            self.enable_gui = enable_gui

            self.subscription = self.create_subscription(
                Image,
                source,
                self.callback,
                1,
            )

            self.get_logger().info(f"ROS2 subscriber started: {source}")

        def callback(self, msg):
            try:
                frame_original = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )
                frame_rotated = rotate_image(frame_original, input_rotation)

                vis, row = process_frame(frame_rotated, model, ocr_reader, cfg)

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