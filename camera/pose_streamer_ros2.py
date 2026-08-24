#!/usr/bin/env python3
"""
연속 자동 6D Pose 스트리머

카메라 피드가 들어오면 즉시 연속으로 pose를 추정하여
ROS topic에 발행합니다. 수동 키 입력 없이 자동 실행합니다.

입력 회전 보정:
    --rotation none
        ROS RGB/Depth가 이미 정방향일 때 사용합니다.

    --rotation 90_cw
        ROS RGB/Depth가 반시계 방향으로 90도 누워 들어올 때 사용합니다.
        RGB, aligned depth, CameraInfo K를 모두 같은 기준으로 보정합니다.

사용법:
    # GUI 모드, 회전 없음
    python3 camera/pose_streamer_ros2.py

    # GUI 모드, 90도 시계 방향 보정
    python3 camera/pose_streamer_ros2.py --rotation 90_cw

    # Headless 모드
    python3 camera/pose_streamer_ros2.py --rotation 90_cw --headless

키 조작 (GUI 모드):
    'q': 종료

ROS Topics:
    /foundation_pose/pose    (PoseStamped)
    /foundation_pose/result  (String, JSON)
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging
import time

import cv2
import numpy as np
import rclpy

from pose_estimator_ros2 import RealtimePoseEstimator, StatusMonitor


class PoseStreamer(RealtimePoseEstimator):
    """연속 자동 pose 추정 스트리머.

    RealtimePoseEstimator를 상속하여 다음 기능을 사용합니다.
    - ROS RGB/Depth 수신 및 회전 보정
    - CameraInfo K 회전 보정
    - Mask 생성, FoundationPose, ROS 발행
    - GUI/Headless 실행

    이 클래스에서는 매 추정마다 track_one()이 아니라 register()를 호출합니다.
    """

    def __init__(self, args):
        super().__init__(args)
        self.headless = args.headless

        # FPS 카운터
        self._fps_times = []
        self._fps = 0.0

    def _update_fps(self):
        """추정 완료 시마다 호출하여 최근 완료 기준 FPS 계산."""
        now = time.time()
        self._fps_times.append(now)
        self._fps_times = self._fps_times[-30:]

        if len(self._fps_times) >= 2:
            dt = self._fps_times[-1] - self._fps_times[0]
            if dt > 0:
                self._fps = (len(self._fps_times) - 1) / dt

    def _estimate_pose_impl(self):
        """매 추정마다 register()를 호출하는 연속 추정 구현."""
        if self.current_rgb is None:
            logging.warning("RGB 프레임 없음")
            self.status.set(StatusMonitor.IDLE)
            return

        if self.K is None:
            logging.warning("카메라 정보 대기 중...")
            self.status.set(StatusMonitor.IDLE)
            return

        self.status.begin()

        # RealtimePoseEstimator의 콜백에서 이미 다음 처리가 완료된 상태입니다.
        # - current_rgb: self.rotation 적용 후 BGR
        # - current_depth: self.rotation 적용 후 aligned depth
        # - self.K: self.rotation 적용 후 실제 추론용 K
        with self.lock:
            rgb_bgr = self.current_rgb.copy()
            depth_raw = (
                self.current_depth.copy()
                if self.current_depth is not None
                else None
            )
            K_used = self.K.copy()

        # BGR -> RGB
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Depth 변환: uint16 mm -> float32 m
        depth = None
        if depth_raw is not None and self.args.input_mode == 'rgbd':
            depth = depth_raw.astype(np.float32) / 1000.0

        # 1단계: 마스킹 (floor로 검출 → mask_conf로 판정)
        self.status.set(StatusMonitor.MASKING)
        mask, mask_info = self.mask_generator.get_mask_with_depth(
            rgb,
            depth,
            depth_refine=self.args.mask_depth_refine,
        )

        if mask is None:
            logging.warning(
                f"마스크: 검출 없음 (floor={self.mask_det_floor})"
            )
            self.status.set(StatusMonitor.MASK_FAIL)
            self._publish_result_json(object_found=False)
            return

        mask_conf = float(mask_info.get('confidence', 0.0))
        if mask_conf < self.args.mask_conf:
            logging.warning(
                f"마스크: best score {mask_conf:.4f} < "
                f"판정 임계값 {self.args.mask_conf} "
                f"(검출 {mask_info.get('num_detections', '?')}개)"
            )
            self.status.set(StatusMonitor.MASK_FAIL)
            self._publish_result_json(object_found=False)
            return

        logging.info(f"Mask confidence: {mask_conf:.4f}")

        if self.args.mask_dilate > 0:
            kernel = np.ones(
                (self.args.mask_dilate, self.args.mask_dilate),
                dtype=np.uint8,
            )
            mask_u8 = (mask * 255).astype(np.uint8)
            mask = cv2.dilate(mask_u8, kernel, iterations=2) > 127
        else:
            mask = mask.astype(bool)

        depth_input = depth if self.args.input_mode == 'rgbd' else None

        # 항상 전체 register() 수행
        self.status.set(StatusMonitor.POSE_INIT)
        pose = self.estimator.register(
            K=K_used,
            rgb=rgb,
            depth=depth_input,
            ob_mask=mask,
            iteration=self.args.est_refine_iter,
        )

        pose = self._correct_pose(pose)
        self.last_pose = pose
        self.pose_count += 1
        self.status.finish()

        # ROS 발행
        self._publish_pose(pose)
        self._publish_result_json(object_found=True, pose=pose)

        # FPS 업데이트
        self._update_fps()

        # 시각화는 headless가 아닐 때만 생성
        if not self.headless:
            self.vis_frame = self._make_vis(
                rgb,
                pose,
                mask,
                K_used,
            )

        logging.info(
            f"[#{self.pose_count}] 완료 "
            f"({self.status.total_ms:.0f}ms, {self._fps:.1f} FPS, "
            f"rotation={self.rotation})"
        )

    def run(self):
        """카메라 대기 후 연속 자동 추정."""
        print("\n" + "=" * 56)
        print(" 연속 자동 6D Pose 스트리머")
        print("=" * 56)
        print(f" 입력 회전 보정: {self.rotation}")

        if self.headless:
            print(" 모드: HEADLESS (ROS만 발행)")
        else:
            print(" 모드: GUI ('q'로 종료)")

        print("=" * 56 + "\n")

        logging.info("카메라 피드 대기 중...")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            # 카메라 데이터가 준비되면 직전 추정 완료 후 다음 추정 시작
            if self.current_rgb is not None and self.K is not None:
                if not self._worker_busy:
                    self.request_estimate()

            if not self.headless and self.current_rgb is not None:
                if self.vis_frame is not None:
                    display = self.vis_frame.copy()
                else:
                    # current_rgb 자체가 이미 회전 보정된 BGR입니다.
                    with self.lock:
                        display = self.current_rgb.copy()

                if self.status.text in (
                    StatusMonitor.MASK_FAIL[0],
                    StatusMonitor.PIPE_ERROR[0],
                ):
                    display = self._draw_error_overlay(display)

                display = self._draw_status_bar(display)

                h, w = display.shape[:2]

                # 회전 상태 표시
                cv2.putText(
                    display,
                    f"ROT: {self.rotation}",
                    (max(10, w - 180), 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                # FPS 표시
                cv2.putText(
                    display,
                    f"{self._fps:.1f} FPS",
                    (max(10, w - 130), h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 0),
                    1,
                )

                cv2.imshow("Pose Streamer", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("종료")
                    break

            time.sleep(1.0 / 30.0)

        if not self.headless:
            cv2.destroyAllWindows()


def parse_args():
    code_dir = _PROJECT_ROOT

    parser = argparse.ArgumentParser(
        description='연속 자동 6D Pose 스트리머',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ROS topics
    parser.add_argument(
        '--rgb_topic',
        type=str,
        default='/camera/camera/color/image_raw',
    )
    parser.add_argument(
        '--depth_topic',
        type=str,
        default='/camera/camera/aligned_depth_to_color/image_raw',
    )
    parser.add_argument(
        '--camera_info_topic',
        type=str,
        default='/camera/camera/color/camera_info',
    )
    parser.add_argument(
        '--camera_frame',
        type=str,
        default='/camera/camera/camera_color_optical_frame',
    )
    parser.add_argument(
        '--rotation',
        type=str,
        default='none',
        choices=['none', '90_cw'],
        help=(
            '입력 RGB/Depth와 CameraInfo K에 적용할 회전 보정. '
            '카메라 영상이 반시계 방향 90도로 누워 있으면 90_cw 사용'
        ),
    )

    # Mesh
    parser.add_argument(
        '--mesh_file',
        type=str,
        default=f'{code_dir}/vcb/ref_views/ob_000001/model/model_vc.ply',
    )
    parser.add_argument('--mesh_scale', type=float, default=0.01)

    # Mask
    parser.add_argument(
        '--mask_model',
        type=str,
        default=f'{code_dir}/weights/2026-02-12-13-41-52/model_best.pth',
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='maskrcnn',
        choices=['yolo', 'maskrcnn'],
    )
    parser.add_argument('--mask_conf', type=float, default=0.9)
    parser.add_argument(
        '--mask_depth_refine',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Refine mask using depth information',
    )
    parser.add_argument('--mask_dilate', type=int, default=0)

    # Pose
    parser.add_argument(
        '--input_mode',
        type=str,
        default='rgb',
        choices=['rgb', 'rgbd'],
    )
    parser.add_argument('--symmetry', type=str, default='z180')
    parser.add_argument('--symmetry_step', type=float, default=5.0)
    parser.add_argument('--fix_z_axis', action='store_true', default=True)
    parser.add_argument('--use_mask_iou', action='store_true', default=True)
    parser.add_argument('--min_n_views', type=int, default=40)
    parser.add_argument('--inplane_step', type=int, default=60)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument(
        '--use_light',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Use Phong shading (True) or constant shading (False)',
    )

    # 저장
    parser.add_argument(
        '--save_dir',
        type=str,
        default=f'{code_dir}/camera/output',
    )

    # 스트리머 전용
    parser.add_argument(
        '--headless',
        action='store_true',
        help='GUI 없이 ROS만 발행 (배포용)',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    streamer = None

    try:
        streamer = PoseStreamer(args)
        streamer.run()
    except KeyboardInterrupt:
        pass
    finally:
        if streamer is not None:
            streamer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()