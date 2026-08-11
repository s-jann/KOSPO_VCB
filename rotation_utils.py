from typing import Tuple

import cv2
import numpy as np


SUPPORTED_ROTATIONS = {
    "none",
    "90_cw",
}


def normalize_rotation(rotation: str) -> str:
    """
    회전 설정 문자열을 검증하고 정규화한다.

    지원 값:
      - none
      - 90_cw
    """
    normalized = str(rotation).strip().lower()

    if normalized not in SUPPORTED_ROTATIONS:
        raise ValueError(
            f"Unsupported rotation: {rotation}. "
            f"Supported values: {sorted(SUPPORTED_ROTATIONS)}"
        )

    return normalized


def rotate_image(
    image_original: np.ndarray,
    rotation: str = "none",
) -> np.ndarray:
    """
    RGB, BGR 또는 2차원 Depth 이미지를 지정된 방향으로 회전한다.

    cv2.rotate는 보간을 수행하지 않으므로 픽셀 값과 dtype이 유지된다.
    """
    if image_original is None:
        raise ValueError("image_original is None")

    rotation = normalize_rotation(rotation)

    if rotation == "none":
        return image_original

    if rotation == "90_cw":
        return cv2.rotate(
            image_original,
            cv2.ROTATE_90_CLOCKWISE,
        )

    raise RuntimeError(
        f"Unhandled rotation: {rotation}"
    )


def rotate_camera_matrix(
    k_original: np.ndarray,
    original_width: int,
    original_height: int,
    rotation: str = "none",
) -> np.ndarray:
    """
    이미지 회전에 맞게 카메라 내부 파라미터 K를 변환한다.

    시계 방향 90도 회전:
        fx_rot = fy
        fy_rot = fx
        cx_rot = H - 1 - cy
        cy_rot = cx

    회전 후 이미지 크기:
        width_rot = original_height
        height_rot = original_width
    """
    rotation = normalize_rotation(rotation)

    k_original = np.asarray(k_original)

    if k_original.shape != (3, 3):
        raise ValueError(
            f"K must have shape (3, 3), "
            f"but got {k_original.shape}"
        )

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            "original_width and original_height "
            "must be positive"
        )

    if rotation == "none":
        return k_original.copy()

    fx = float(k_original[0, 0])
    fy = float(k_original[1, 1])
    cx = float(k_original[0, 2])
    cy = float(k_original[1, 2])

    if rotation == "90_cw":
        return np.array(
            [
                [
                    fy,
                    0.0,
                    float(original_height - 1) - cy,
                ],
                [0.0, fx, cx],
                [0.0, 0.0, 1.0],
            ],
            dtype=k_original.dtype,
        )

    raise RuntimeError(
        f"Unhandled rotation: {rotation}"
    )


def rotate_rgb_and_camera_matrix(
    rgb_original: np.ndarray,
    k_original: np.ndarray,
    rotation: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RGB 이미지와 K를 동일한 회전 조건으로 변환한다.
    """
    if rgb_original is None:
        raise ValueError("rgb_original is None")

    original_height, original_width = (
        rgb_original.shape[:2]
    )

    rgb_rotated = rotate_image(
        rgb_original,
        rotation,
    )

    k_rotated = rotate_camera_matrix(
        k_original=k_original,
        original_width=original_width,
        original_height=original_height,
        rotation=rotation,
    )

    return rgb_rotated, k_rotated