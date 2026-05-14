import os
import re
import cv2
import easyocr
from difflib import SequenceMatcher
from ultralytics import YOLO


# =========================
# 기본 설정 (standalone 테스트용 기본값)
# main_infer.py에서는 config로 덮어쓰는 구조 권장
# =========================
MODEL_PATH = "/home/energy/jan/two-label/runs/breaker_yolo11n3/weights/best.pt"
SOURCE_DIR = "/home/energy/jan/two-label/dataset/valid/images"
OUTPUT_DIR = "/home/energy/jan/two-label/ocr_outputs"

CONF_THRES = 0.25
LABEL_CLASS_NAME = "label"
OCR_LANG_LIST = ["en"]
USE_GPU = True

RAW_RESULT_PATH = os.path.join(OUTPUT_DIR, "ocr_results_raw.txt")
REFINED_RESULT_PATH = os.path.join(OUTPUT_DIR, "ocr_results_refined.txt")


# =========================
# 가능한 정답 코드 전체 생성
# =========================
def build_valid_codes():
    valid_codes = []
    for first in ["3", "4"]:
        for sw in ["01", "02"]:
            max_num = 13 if sw == "01" else 14
            for num in range(1, max_num + 1):
                for suffix in ["A", "B"]:
                    valid_codes.append(f"{first}SW{sw}-{num:02d}{suffix}")
    return valid_codes


VALID_CODES = build_valid_codes()
VALID_CODE_SET = set(VALID_CODES)


# =========================
# OCR 문자열 정규화
# =========================
def normalize_ocr_text(text: str) -> str:
    text = text.upper().strip()

    text = text.replace(" ", "")
    text = text.replace("_", "-")
    text = text.replace("—", "-")
    text = text.replace("–", "-")

    chars = []
    for ch in text:
        if ch.isalnum() or ch == "-":
            chars.append(ch)
    text = "".join(chars)

    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("L", "1")
    text = text.replace("|", "1")

    text = text.replace("5W", "SW")
    text = text.replace("SWW", "SW")
    text = text.replace("3W", "3SW")
    text = text.replace("4W", "4SW")

    return text


# =========================
# OCR 텍스트 선택
# =========================
def pick_top_line_text(ocr_results, y_weight=1.0):
    """
    EasyOCR 결과 중 bbox 평균 y값이 가장 작은 텍스트 선택
    return: top_text, raw_joined
    """
    if not ocr_results:
        return "", "NO_TEXT"

    items = []
    raw_parts = []

    for item in ocr_results:
        bbox, text, conf = item
        y_mean = sum(pt[1] for pt in bbox) / 4.0
        items.append((y_mean * y_weight, text, conf))
        raw_parts.append(f"{text}({conf:.3f})")

    items.sort(key=lambda x: x[0])
    top_text = items[0][1].strip()
    raw_joined = " | ".join(raw_parts)

    return top_text, raw_joined


def concat_all_text(ocr_results):
    if not ocr_results:
        return "", "NO_TEXT"

    texts = []
    raw_parts = []
    for bbox, text, conf in ocr_results:
        texts.append(text.strip())
        raw_parts.append(f"{text}({conf:.3f})")

    merged = " ".join([t for t in texts if t])
    raw_joined = " | ".join(raw_parts)
    return merged, raw_joined


# =========================
# 패턴 후보 추출
# =========================
def extract_pattern_candidates(text: str):
    candidates = set()

    if not text:
        return []

    candidates.add(text)

    compact = text.replace("-", "")
    if len(compact) >= 8:
        for i in range(len(compact) - 7):
            chunk = compact[i:i + 8]
            cand = chunk[:5] + "-" + chunk[5:]
            candidates.add(cand)

    pattern_like = re.findall(r"[34][A-Z0-9]{2}\d{2}-?\d{2}[A-Z0-9]", text)
    for p in pattern_like:
        if "-" not in p and len(p) == 8:
            p = p[:5] + "-" + p[5:]
        candidates.add(p)

    return list(candidates)


# =========================
# 규칙 점수
# =========================
def rule_score(code: str) -> int:
    score = 0

    if len(code) == 9:
        score += 1

    if re.fullmatch(r"[34]SW(01|02)-\d{2}[AB]", code):
        score += 5

    m = re.fullmatch(r"([34])SW(01|02)-(\d{2})([AB])", code)
    if m:
        score += 5
        sw = m.group(2)
        num = int(m.group(3))

        if sw == "01" and 1 <= num <= 13:
            score += 5
        elif sw == "02" and 1 <= num <= 14:
            score += 5

    if code in VALID_CODE_SET:
        score += 20

    return score


# =========================
# 가장 가까운 유효 코드 찾기
# =========================
def best_match_from_valid_codes(text: str):
    best_code = None
    best_score = -1.0

    for code in VALID_CODES:
        score = SequenceMatcher(None, text, code).ratio()
        if score > best_score:
            best_score = score
            best_code = code

    return best_code, best_score


# =========================
# 최종 보정
# =========================
def refine_ocr_result(
    selected_text: str,
    candidate_threshold: int = 10,
    similarity_candidate: float = 0.70,
    similarity_normalized: float = 0.60,
):
    if not selected_text:
        return "", "", "NO_TEXT", "none"

    best_raw = selected_text.strip()
    normalized = normalize_ocr_text(best_raw)

    candidates = extract_pattern_candidates(normalized)

    scored = []
    for cand in candidates:
        cand = cand.upper()
        if len(cand) == 8 and "-" not in cand:
            cand = cand[:5] + "-" + cand[5:]
        scored.append((cand, rule_score(cand)))

    scored.sort(key=lambda x: x[1], reverse=True)

    if scored and scored[0][1] >= candidate_threshold:
        top_candidate = scored[0][0]

        if top_candidate in VALID_CODE_SET:
            return best_raw, normalized, top_candidate, "direct_valid"

        nearest, sim = best_match_from_valid_codes(top_candidate)
        if sim >= similarity_candidate:
            return best_raw, normalized, nearest, f"nearest_from_candidate({sim:.3f})"

    nearest, sim = best_match_from_valid_codes(normalized)
    if sim >= similarity_normalized:
        return best_raw, normalized, nearest, f"nearest_from_normalized({sim:.3f})"

    return best_raw, normalized, "NO_PATTERN", "failed"


# =========================
# EasyOCR Reader 관리
# =========================
_reader = None


def init_easyocr_reader(lang_list=None, use_gpu=True):
    global _reader
    if _reader is None:
        if lang_list is None:
            lang_list = ["en"]
        _reader = easyocr.Reader(lang_list, gpu=use_gpu)
    return _reader


# =========================
# crop 하나에 대해 OCR 수행
# =========================
def run_easyocr_on_crop(
    crop,
    reader=None,
    lang_list=None,
    use_gpu=True,
    resize=2.0,
    grayscale=True,
    detail=1,
    min_confidence=0.3,
    text_selection_method="top_line",
    y_weight=1.0,
    candidate_threshold=10,
    similarity_candidate=0.70,
    similarity_normalized=0.60,
):
    if crop is None or crop.size == 0:
        return {
            "top_text": "",
            "raw_joined": "NO_TEXT",
            "best_raw": "",
            "normalized": "",
            "refined": "NO_TEXT",
            "method": "invalid_crop",
        }

    if reader is None:
        reader = init_easyocr_reader(lang_list=lang_list, use_gpu=use_gpu)

    proc = crop.copy()

    if grayscale:
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    if resize and resize != 1.0:
        proc = cv2.resize(
            proc,
            None,
            fx=resize,
            fy=resize,
            interpolation=cv2.INTER_CUBIC,
        )

    ocr_results = reader.readtext(proc, detail=detail)

    if min_confidence is not None and detail == 1:
        ocr_results = [x for x in ocr_results if x[2] >= min_confidence]

    if text_selection_method == "all_concat":
        selected_text, raw_joined = concat_all_text(ocr_results)
    else:
        selected_text, raw_joined = pick_top_line_text(ocr_results, y_weight=y_weight)

    best_raw, normalized, refined, method = refine_ocr_result(
        selected_text=selected_text,
        candidate_threshold=candidate_threshold,
        similarity_candidate=similarity_candidate,
        similarity_normalized=similarity_normalized,
    )

    return {
        "top_text": selected_text,
        "raw_joined": raw_joined,
        "best_raw": best_raw,
        "normalized": normalized,
        "refined": refined,
        "method": method,
    }

# def run_easyocr_on_crop(
#     crop,
#     reader=None,
#     lang_list=None,
#     use_gpu=True,
#     resize=2.0,
#     grayscale=True,
#     detail=1,
#     min_confidence=0.3,
#     text_selection_method="top_line",
#     y_weight=1.0,
#     candidate_threshold=10,
#     similarity_candidate=0.70,
#     similarity_normalized=0.60,
# ):
#     if crop is None or crop.size == 0:
#         return {
#             "top_text": "",
#             "raw_joined": "NO_TEXT",
#             "best_raw": "",
#             "normalized": "",
#             "refined": "NO_TEXT",
#             "method": "invalid_crop",
#             "raw": [],
#         }

#     if reader is None:
#         reader = init_easyocr_reader(lang_list=lang_list, use_gpu=use_gpu)

#     proc = crop.copy()

#     if grayscale:
#         proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

#     if resize and resize != 1.0:
#         proc = cv2.resize(
#             proc,
#             None,
#             fx=resize,
#             fy=resize,
#             interpolation=cv2.INTER_CUBIC,
#         )

#     # EasyOCR recognize()는 보통 3채널 이미지를 기대하므로
#     # grayscale이면 다시 BGR로 맞춰주는 게 안전함
#     if len(proc.shape) == 2:
#         proc_for_ocr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
#     else:
#         proc_for_ocr = proc

#     h, w = proc_for_ocr.shape[:2]

#     # YOLO가 이미 label 영역을 잘라줬으므로
#     # crop 전체를 하나의 텍스트 영역으로 보고 recognize만 수행
#     horizontal_list = [[0, w, 0, h]]
#     free_list = []

#     try:
#         ocr_results = reader.recognize(
#             proc_for_ocr,
#             horizontal_list=horizontal_list,
#             free_list=free_list,
#             decoder="greedy",
#             beamWidth=5,
#             batch_size=1,
#             workers=0,
#             detail=1,         # 후처리 일관성을 위해 항상 detail=1로 받기
#             paragraph=False,
#         )
#     except Exception as e:
#         return {
#             "top_text": "",
#             "raw_joined": "NO_TEXT",
#             "best_raw": "",
#             "normalized": "",
#             "refined": "NO_TEXT",
#             "method": f"recognize_error:{type(e).__name__}",
#             "raw": [],
#         }

#     # recognize 결과도 [(box, text, conf), ...] 형태라 기존 후처리 재사용 가능
#     if min_confidence is not None:
#         filtered = []
#         for x in ocr_results:
#             if len(x) >= 3 and x[2] >= min_confidence:
#                 filtered.append(x)
#         ocr_results = filtered

#     if text_selection_method == "all_concat":
#         selected_text, raw_joined = concat_all_text(ocr_results)
#     else:
#         selected_text, raw_joined = pick_top_line_text(ocr_results, y_weight=y_weight)

#     best_raw, normalized, refined, method = refine_ocr_result(
#         selected_text=selected_text,
#         candidate_threshold=candidate_threshold,
#         similarity_candidate=similarity_candidate,
#         similarity_normalized=similarity_normalized,
#     )

#     return {
#         "top_text": selected_text,
#         "raw_joined": raw_joined,
#         "best_raw": best_raw,
#         "normalized": normalized,
#         "refined": refined,
#         "method": method,
#         "raw": ocr_results,
#     }

# =========================
# standalone batch test
# =========================
def run_batch_validation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "crops"), exist_ok=True)

    model = YOLO(MODEL_PATH)
    reader = init_easyocr_reader(lang_list=OCR_LANG_LIST, use_gpu=USE_GPU)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with open(RAW_RESULT_PATH, "w", encoding="utf-8") as f_raw, \
         open(REFINED_RESULT_PATH, "w", encoding="utf-8") as f_refined:

        f_raw.write("image\tcrop\ttop_line_raw\tall_ocr_raw\n")
        f_refined.write("image\tcrop\tbest_raw\tnormalized\trefined\tmethod\n")

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
                f_raw.write(f"{file_name}\tNO_DETECTION\tNO_TEXT\tNO_TEXT\n")
                f_refined.write(f"{file_name}\tNO_DETECTION\t\t\tNO_TEXT\tnone\n")
                continue

            label_count = 0

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names[cls_id]

                if cls_name != LABEL_CLASS_NAME:
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

                label_count += 1
                crop_name = f"{os.path.splitext(file_name)[0]}_label_{label_count}.jpg"
                crop_path = os.path.join(OUTPUT_DIR, "crops", crop_name)
                cv2.imwrite(crop_path, crop)

                ocr_out = run_easyocr_on_crop(
                    crop=crop,
                    reader=reader,
                    lang_list=OCR_LANG_LIST,
                    use_gpu=USE_GPU,
                    resize=2.0,
                    grayscale=True,
                    detail=1,
                    min_confidence=0.3,
                    text_selection_method="top_line",
                    y_weight=1.0,
                    candidate_threshold=10,
                    similarity_candidate=0.70,
                    similarity_normalized=0.60,
                )

                print(f"[{file_name}] {crop_name}")
                print(f"  TOP_RAW  : {ocr_out['top_text'] if ocr_out['top_text'] else 'NO_TEXT'}")
                print(f"  ALL_RAW  : {ocr_out['raw_joined']}")
                print(f"  NORMAL   : {ocr_out['normalized']}")
                print(f"  REFINED  : {ocr_out['refined']} ({ocr_out['method']})")

                f_raw.write(
                    f"{file_name}\t{crop_name}\t"
                    f"{ocr_out['top_text'] if ocr_out['top_text'] else 'NO_TEXT'}\t"
                    f"{ocr_out['raw_joined']}\n"
                )
                f_refined.write(
                    f"{file_name}\t{crop_name}\t{ocr_out['best_raw']}\t"
                    f"{ocr_out['normalized']}\t{ocr_out['refined']}\t{ocr_out['method']}\n"
                )

            if label_count == 0:
                f_raw.write(f"{file_name}\tNO_LABEL\tNO_TEXT\tNO_TEXT\n")
                f_refined.write(f"{file_name}\tNO_LABEL\t\t\tNO_PATTERN\tnone\n")

    print(f"\nRAW 결과 저장: {RAW_RESULT_PATH}")
    print(f"보정 결과 저장: {REFINED_RESULT_PATH}")


if __name__ == "__main__":
    run_batch_validation()