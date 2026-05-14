import os
import cv2
import csv
import time
import yaml
from ultralytics import YOLO
from easyocr_val_data_rule import init_easyocr_reader, run_easyocr_on_crop
from hsv_val_data import run_hsv_on_crop

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


def get_best_box_by_class(results, model_names):
    """
    클래스별 최고 confidence bbox 1개씩 선택
    return:
        {
            "vcb": (x1, y1, x2, y2, conf),
            "label": (...),
            "status": (...)
        }
    """
    best = {}

    if results.boxes is None or len(results.boxes) == 0:
        return best

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        cls_name = model_names[cls_id]
        if cls_name not in CLASS_NAMES:
            continue

        if cls_name not in best or conf > best[cls_name][4]:
            x1, y1, x2, y2 = map(int, box)
            best[cls_name] = (x1, y1, x2, y2, float(conf))

    return best


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


def draw_result(frame, boxes, ocr_text, status_text):
    vis = frame.copy()

    for cls_name, box in boxes.items():
        x1, y1, x2, y2, conf = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{cls_name}:{conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        vis,
        f"OCR: {ocr_text}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"STATUS: {status_text}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    return vis


def box_to_str(box):
    if box is None:
        return ""
    x1, y1, x2, y2, conf = box
    return f"{x1},{y1},{x2},{y2},{conf:.4f}"


def process_frame(frame, model, ocr_reader, cfg):
    yolo_cfg = cfg["yolo"]
    ocr_cfg = cfg["ocr"]
    hsv_cfg = cfg["hsv"]

    conf = yolo_cfg.get("conf", 0.25)
    imgsz = yolo_cfg.get("imgsz", 640)

    t0 = time.time()
    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    yolo_time = time.time() - t0

    best_boxes = get_best_box_by_class(results, model.names)

    ocr_text = "N/A"
    status_text = "N/A"
    ocr_raw = ""
    hsv_label = ""
    hsv_green = -1
    hsv_red = -1

    if all(k in best_boxes for k in ["vcb", "label", "status"]):
        label_crop = crop_image(frame, best_boxes["label"])
        status_crop = crop_image(frame, best_boxes["status"])

        if label_crop is not None:
            ocr_out = run_easyocr_on_crop(
                crop=label_crop,
                reader=ocr_reader,
                lang_list=ocr_cfg.get("lang", ["en"]),
                use_gpu=ocr_cfg.get("use_gpu", True),
                resize=get_nested(ocr_cfg, ["preprocess", "resize"], 2.0),
                grayscale=get_nested(ocr_cfg, ["preprocess", "grayscale"], True),
                detail=get_nested(ocr_cfg, ["detection", "detail"], 1),
                min_confidence=get_nested(ocr_cfg, ["detection", "min_confidence"], 0.3),
                text_selection_method=get_nested(ocr_cfg, ["text_selection", "method"], "top_line"),
                y_weight=get_nested(ocr_cfg, ["text_selection", "y_weight"], 1.0),
                candidate_threshold=get_nested(ocr_cfg, ["refine", "candidate_threshold"], 10),
                similarity_candidate=get_nested(
                    ocr_cfg, ["refine", "similarity_threshold", "candidate"], 0.70
                ),
                similarity_normalized=get_nested(
                    ocr_cfg, ["refine", "similarity_threshold", "normalized"], 0.60
                ),
            )

            ocr_text = ocr_out.get("refined", "N/A")
            ocr_raw = str(ocr_out.get("raw_joined", ""))

        if status_crop is not None and ocr_text not in ["", None, "NO_TEXT", "NO_PATTERN", "N/A"]:
            hsv_out = run_hsv_on_crop(status_crop, hsv_cfg)
            hsv_label = hsv_out.get("label", "N/A")
            hsv_green = hsv_out.get("green_pixels", -1)
            hsv_red = hsv_out.get("red_pixels", -1)
            status_text = f"{hsv_label} (g={hsv_green}, r={hsv_red})"

    total_time = time.time() - t0
    fps = 1.0 / total_time if total_time > 0 else 0.0

    vis = draw_result(frame, best_boxes, ocr_text, status_text)
    cv2.putText(
        vis,
        f"YOLO:{yolo_time*1000:.1f}ms TOTAL:{total_time*1000:.1f}ms FPS:{fps:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    result_row = {
        "ocr_text": ocr_text,
        "ocr_raw": ocr_raw,
        "status_text": status_text,
        "hsv_label": hsv_label,
        "hsv_green": hsv_green,
        "hsv_red": hsv_red,
        "vcb_box": best_boxes.get("vcb"),
        "label_box": best_boxes.get("label"),
        "status_box": best_boxes.get("status"),
        "yolo_ms": yolo_time * 1000.0,
        "total_ms": total_time * 1000.0,
        "fps": fps,
        "num_boxes": len(best_boxes),
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
    csv_writer.writerow([
        "frame_idx",
        "source",
        "ocr_text",
        "ocr_raw",
        "status_text",
        "hsv_label",
        "hsv_green",
        "hsv_red",
        "vcb_box",
        "label_box",
        "status_box",
        "num_boxes",
        "yolo_ms",
        "total_ms",
        "fps",
    ])


def write_csv_row(csv_writer, frame_idx, source, row):
    csv_writer.writerow([
        frame_idx,
        source,
        row["ocr_text"],
        row["ocr_raw"],
        row["status_text"],
        row["hsv_label"],
        row["hsv_green"],
        row["hsv_red"],
        box_to_str(row["vcb_box"]),
        box_to_str(row["label_box"]),
        box_to_str(row["status_box"]),
        row["num_boxes"],
        f"{row['yolo_ms']:.3f}",
        f"{row['total_ms']:.3f}",
        f"{row['fps']:.3f}",
    ])


def main():
    config_path = "/home/robot/Workspace/FoundationPose_VCB/yolo/configs/infer_config.yaml"
    cfg = load_config(config_path)

    # required / existing config
    model_path = cfg["yolo"]["model_path"]
    source = cfg["input"]["source"]
    save_dir = cfg["output"]["save_dir"]

    ocr_cfg = cfg["ocr"]

    # optional output config (yaml에 없어도 기본값으로 동작)
    enable_gui = get_nested(cfg, ["output", "enable_gui"], False)
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
        # case 1) single image
        if is_image_file(source):
            print(f"[INFO] Image input detected: {source}")
            frame = cv2.imread(source)
            if frame is None:
                raise RuntimeError(f"Cannot read image: {source}")

            vis, row = process_frame(frame, model, ocr_reader, cfg)
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

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_fps = cap.get(cv2.CAP_PROP_FPS)


            if src_fps is None or src_fps <= 1e-6 or src_fps != src_fps:
                src_fps = 20.0

            if save_video:
                writer, writer_path = init_video_writer(
                    save_video=save_video,
                    save_dir=save_dir,
                    video_name=video_name,
                    fps=src_fps,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
                if writer is not None:
                    print(f"[INFO] Saving annotated video: {writer_path}")

            frame_idx = 0

            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[INFO] End of stream or failed frame read.")
                    break
                
                vis, row = process_frame(frame, model, ocr_reader, cfg)
                write_csv_row(csv_writer, frame_idx, str(source), row)
                processed_count += 1

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