import os
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# standalone batch test용 기본 설정
# =========================
MODEL_PATH = "/home/energy/jan/two-label/runs/breaker_yolo11n3/weights/best.pt"
SOURCE_DIR = "/home/energy/jan/two-label/dataset/valid/images"
OUTPUT_DIR = "/home/energy/jan/two-label/status_outputs"

CONF_THRES = 0.25
STATUS_CLASS_NAME = "status"

RAW_RESULT_PATH = os.path.join(OUTPUT_DIR, "status_results.txt")


def classify_status_color(
    crop_bgr,
    lower_green,
    upper_green,
    lower_red_1,
    upper_red_1,
    lower_red_2,
    upper_red_2,
    min_color_pixels=5,
    use_morphology=True,
    kernel_size=3,
):
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "label": "unknown",
            "green_pixels": 0,
            "red_pixels": 0,
            "green_ratio": 0.0,
            "red_ratio": 0.0,
            "green_mask": None,
            "red_mask": None,
        }

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array(lower_green, dtype=np.uint8)
    upper_green = np.array(upper_green, dtype=np.uint8)
    lower_red_1 = np.array(lower_red_1, dtype=np.uint8)
    upper_red_1 = np.array(upper_red_1, dtype=np.uint8)
    lower_red_2 = np.array(lower_red_2, dtype=np.uint8)
    upper_red_2 = np.array(upper_red_2, dtype=np.uint8)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    if use_morphology:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    green_pixels = int(cv2.countNonZero(green_mask))
    red_pixels = int(cv2.countNonZero(red_mask))

    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0

    if green_pixels < min_color_pixels and red_pixels < min_color_pixels:
        label = "unknown"
    elif green_pixels > red_pixels:
        label = "green"
    elif red_pixels > green_pixels:
        label = "red"
    else:
        label = "unknown"

    return {
        "label": label,
        "green_pixels": green_pixels,
        "red_pixels": red_pixels,
        "green_ratio": green_ratio,
        "red_ratio": red_ratio,
        "green_mask": green_mask,
        "red_mask": red_mask,
    }


def run_hsv_on_crop(crop_bgr, hsv_cfg):
    """
    main_infer.py에서 직접 호출할 함수
    """
    return classify_status_color(
        crop_bgr=crop_bgr,
        lower_green=hsv_cfg["green"]["lower"],
        upper_green=hsv_cfg["green"]["upper"],
        lower_red_1=hsv_cfg["red"]["lower1"],
        upper_red_1=hsv_cfg["red"]["upper1"],
        lower_red_2=hsv_cfg["red"]["lower2"],
        upper_red_2=hsv_cfg["red"]["upper2"],
        min_color_pixels=hsv_cfg["min_color_pixels"],
        use_morphology=hsv_cfg.get("use_morphology", True),
        kernel_size=hsv_cfg.get("kernel_size", 3),
    )


def run_batch_validation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "crops"), exist_ok=True)

    model = YOLO(MODEL_PATH)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # standalone 기본값
    hsv_cfg = {
        "green": {
            "lower": [25, 20, 20],
            "upper": [95, 255, 255],
        },
        "red": {
            "lower1": [0, 50, 50],
            "upper1": [10, 255, 255],
            "lower2": [170, 50, 50],
            "upper2": [179, 255, 255],
        },
        "min_color_pixels": 5,
        "use_morphology": True,
        "kernel_size": 3,
    }

    with open(RAW_RESULT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("image\tcrop\tpredicted_color\tgreen_pixels\tred_pixels\tgreen_ratio\tred_ratio\n")

        for file_name in sorted(os.listdir(SOURCE_DIR)):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in image_exts:
                continue

            img_path = os.path.join(SOURCE_DIR, file_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[WARN] 이미지 로드 실패: {img_path}")
                continue

            results = model.predict(source=img, conf=CONF_THRES, verbose=False)
            result = results[0]
            names = result.names

            if result.boxes is None or len(result.boxes) == 0:
                f_out.write(f"{file_name}\tNO_DETECTION\tunknown\t0\t0\t0.0\t0.0\n")
                continue

            status_count = 0

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names[cls_id]

                if cls_name != STATUS_CLASS_NAME:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                h, w = img.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                status_count += 1
                crop_name = f"{os.path.splitext(file_name)[0]}_status_{status_count}.jpg"
                crop_path = os.path.join(OUTPUT_DIR, "crops", crop_name)
                cv2.imwrite(crop_path, crop)

                color_info = run_hsv_on_crop(crop, hsv_cfg)

                pred_color = color_info["label"]
                green_pixels = color_info["green_pixels"]
                red_pixels = color_info["red_pixels"]
                green_ratio = color_info["green_ratio"]
                red_ratio = color_info["red_ratio"]

                green_mask_path = os.path.join(
                    OUTPUT_DIR, "crops", f"{os.path.splitext(file_name)[0]}_status_{status_count}_green_mask.png"
                )
                red_mask_path = os.path.join(
                    OUTPUT_DIR, "crops", f"{os.path.splitext(file_name)[0]}_status_{status_count}_red_mask.png"
                )

                cv2.imwrite(green_mask_path, color_info["green_mask"])
                cv2.imwrite(red_mask_path, color_info["red_mask"])

                print(f"[{file_name}] {crop_name} -> {pred_color} "
                      f"(green={green_pixels}, red={red_pixels})")

                f_out.write(
                    f"{file_name}\t{crop_name}\t{pred_color}\t"
                    f"{green_pixels}\t{red_pixels}\t{green_ratio:.6f}\t{red_ratio:.6f}\n"
                )

            if status_count == 0:
                f_out.write(f"{file_name}\tNO_STATUS\tunknown\t0\t0\t0.0\t0.0\n")

    print(f"\n완료. 결과 저장: {RAW_RESULT_PATH}")


if __name__ == "__main__":
    run_batch_validation()