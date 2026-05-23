#!/usr/bin/env python3
"""
FoundationPose 디버깅 전용 실시간 Pose 추정

pose_estimator.py의 RealtimePoseEstimator를 상속하여:
  - ROS publisher (PoseStamped, JSON) 제거
  - FoundationPose debug=2로 vis_score/vis_refiner 자동 생성
  - vis_score 캔버스를 별도 OpenCV 창에 표시
  - track_vis + vis_score를 camera/debug/{session}/frame_{NNNN}/에 저장

사용법:
    python camera/pose_debugger_ros2.py [옵션]

키 조작:
    'p' 또는 스페이스: Pose 추정 (단일)
    't': 트래킹 모드 ON/OFF (연속 추정)
    'r': 리셋 (트래킹 초기화)
    'q': 종료

사전 준비:
    1. ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
    2. FoundationPose weights가 weights/ 에 있어야 함
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import glob
import logging
import time
from datetime import datetime

import cv2
import numpy as np

import rclpy

import nvdiffrast.torch as dr
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

from pose_estimator_ros2 import (
    RealtimePoseEstimator, StatusMonitor, parse_args, SymmetryGenerator,
)


class PoseDebugger(RealtimePoseEstimator):
    """debug=2 모드로 vis_score/vis_refiner를 저장·표시하는 디버깅 추정기."""

    def __init__(self, args):
        # 세션 디렉토리 미리 생성 (부모 __init__ 에서 _init_model 호출)
        self.session_dir = os.path.join(
            _SCRIPT_DIR, 'debug',
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        )
        os.makedirs(self.session_dir, exist_ok=True)
        self.debug_frame_idx = 0
        self.last_vis_score_img = None
        super().__init__(args)
        logging.info(f"Debug 세션: {self.session_dir}")

    # ------------------------------------------------------------------
    # Override 1: debug=2로 FoundationPose 생성
    # ------------------------------------------------------------------

    def _init_model(self):
        """FoundationPose를 debug=2로 초기화."""

        mesh = self._load_mesh()
        self.mesh = mesh
        self.extents = mesh.extents
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0)

        symmetry_tfs = SymmetryGenerator.create(
            self.args.symmetry, self.args.symmetry_step
        )
        logging.info(f"대칭: {self.args.symmetry} ({len(symmetry_tfs)}개)")

        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            symmetry_tfs=symmetry_tfs,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=2,
            debug_dir=self.session_dir,
            min_n_views=self.args.min_n_views,
            inplane_step=self.args.inplane_step,
            front_hemisphere_only=self.args.fix_z_axis,
            use_mask_iou=self.args.use_mask_iou,
            use_light=self.args.use_light,
        )

        from mask_generator import create_mask_generator
        self.mask_generator = create_mask_generator(
            model_path=self.args.mask_model,
            model_type=self.args.mask_type,
            conf_threshold=self.args.mask_conf,
        )
        logging.info(f"마스크: {self.args.mask_type} (conf={self.args.mask_conf})")

    # ------------------------------------------------------------------
    # ROS publisher 비활성화
    # ------------------------------------------------------------------

    def _publish_pose(self, pose):
        pass

    def _publish_result_json(self, object_found, pose=None):
        pass

    # ------------------------------------------------------------------
    # Override 2: 프레임별 debug_dir 리다이렉트 + vis_score 캡처
    # ------------------------------------------------------------------

    def _estimate_pose_impl(self):
        """매 프레임마다 debug_dir을 리다이렉트하고 vis_score를 표시."""
        if self.current_rgb is None:
            logging.warning("RGB 프레임 없음")
            self.status.set(StatusMonitor.IDLE)
            return
        if self.K is None:
            logging.warning("카메라 정보 대기 중...")
            self.status.set(StatusMonitor.IDLE)
            return

        # 프레임별 디렉토리 설정
        frame_dir = os.path.join(
            self.session_dir, f'frame_{self.debug_frame_idx:04d}'
        )
        os.makedirs(frame_dir, exist_ok=True)

        # estimater.py가 이 디렉토리에 vis_score, vis_refiner 등 저장
        self.estimator.debug_dir = frame_dir
        # 프레임 카운터 리셋 (파일명이 000000으로 시작)
        self.estimator._debug_frame_count = 0

        self.status.begin()

        with self.lock:
            rgb_bgr = self.current_rgb.copy()
            depth_raw = (
                self.current_depth.copy()
                if self.current_depth is not None
                else None
            )

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = None
        if depth_raw is not None and self.args.input_mode == 'rgbd':
            depth = depth_raw.astype(np.float32) / 1000.0

        # 1단계: R-CNN 마스킹
        self.status.set(StatusMonitor.MASKING)
        mask, mask_info = self.mask_generator.get_mask_with_depth(
            rgb, depth, depth_refine=self.args.mask_depth_refine)

        if mask is None:
            logging.warning("R-CNN: 객체를 찾을 수 없습니다!")
            self.status.set(StatusMonitor.MASK_FAIL)
            self.debug_frame_idx += 1
            return

        if self.args.mask_dilate > 0:
            kernel = np.ones(
                (self.args.mask_dilate, self.args.mask_dilate), np.uint8
            )
            mask_u8 = (mask * 255).astype(np.uint8)
            mask = cv2.dilate(mask_u8, kernel, iterations=2) > 127
        else:
            mask = mask.astype(bool)

        depth_input = depth if self.args.input_mode == 'rgbd' else None

        if self.last_pose is not None and self.is_tracking:
            self.status.set(StatusMonitor.TRACKING)
            pose = self.estimator.track_one(
                rgb=rgb, depth=depth_input, K=self.K,
                iteration=self.args.track_refine_iter,
            )
        else:
            self.status.set(StatusMonitor.POSE_INIT)
            pose = self.estimator.register(
                K=self.K, rgb=rgb, depth=depth_input,
                ob_mask=mask, iteration=self.args.est_refine_iter,
            )

        pose = self._correct_pose(pose)
        self.last_pose = pose
        self.pose_count += 1
        self.status.finish()

        # 시각화 프레임 생성 및 저장
        self.vis_frame = self._make_vis(rgb, pose, mask)
        cv2.imwrite(os.path.join(frame_dir, 'track_vis.png'), self.vis_frame)
        np.savetxt(os.path.join(frame_dir, 'ob_in_cam.txt'), pose)

        # vis_score 이미지 로드 (estimater.py가 저장한 것)
        vis_score_pattern = os.path.join(frame_dir, 'vis_score', 'vis_score_*.png')
        vis_score_files = sorted(glob.glob(vis_score_pattern))
        if vis_score_files:
            self.last_vis_score_img = cv2.imread(vis_score_files[-1])
        else:
            self.last_vis_score_img = None

        self.debug_frame_idx += 1
        logging.info(
            f"[#{self.pose_count}] 완료 (총 {self.status.total_ms:.0f}ms) "
            f"→ {frame_dir}"
        )

    # ------------------------------------------------------------------
    # Override 3: vis_score 별도 창 표시 + 세션 요약
    # ------------------------------------------------------------------

    def run(self):
        """vis_score 별도 창이 추가된 메인 루프."""

        print("\n" + "=" * 50)
        print(" FoundationPose Debugger")
        print("=" * 50)
        print(f" Session: {self.session_dir}")
        print(" 'p'/Space : Pose 추정 (단일)")
        print(" 't'       : 트래킹 모드 ON/OFF")
        print(" 'r'       : 리셋 (트래킹 초기화)")
        print(" 'q'       : 종료")
        print("=" * 50 + "\n")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            if self.current_rgb is not None:
                if self.is_tracking and self.K is not None and not self._worker_busy:
                    self.request_estimate()

                if self.vis_frame is not None:
                    display = self.vis_frame.copy()
                else:
                    display = self.current_rgb.copy()

                if self.status.text == StatusMonitor.MASK_FAIL[0]:
                    display = self._draw_error_overlay(display)

                display = self._draw_status_bar(display)

                h, w = display.shape[:2]
                if self.is_tracking:
                    mode_text = "MODE: TRACK"
                    mode_color = (0, 150, 255)
                else:
                    mode_text = "MODE: DEBUG"
                    mode_color = (0, 200, 255)
                cv2.putText(
                    display, mode_text,
                    (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, mode_color, 1,
                )

                help_text = "'p':Pose  't':Track  'r':Reset  'q':Quit"
                cv2.putText(
                    display, help_text,
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1,
                )

                # 프레임 카운트 표시
                frame_text = f"Frame #{self.debug_frame_idx}"
                cv2.putText(
                    display, frame_text,
                    (w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 1,
                )

                cv2.imshow("Pose Debugger", display)

                # vis_score 별도 창
                if self.last_vis_score_img is not None:
                    cv2.imshow("Vis Score", self.last_vis_score_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('p') or key == 32:
                if not self.is_tracking and not self._worker_busy:
                    logging.info("Pose 추정 요청 (debug)")
                    self.request_estimate()

            elif key == ord('t'):
                self.is_tracking = not self.is_tracking
                if self.is_tracking:
                    logging.info("트래킹 모드 ON")
                    if self.last_pose is None and not self._worker_busy:
                        self.request_estimate()
                else:
                    logging.info("트래킹 모드 OFF")

            elif key == ord('r'):
                self.last_pose = None
                self.is_tracking = False
                self.vis_frame = None
                self.last_vis_score_img = None
                self.estimator.pose_last = None
                self.status.set(StatusMonitor.IDLE)
                logging.info("리셋 완료")

            elif key == ord('q'):
                break

            time.sleep(1.0 / 30.0)

        cv2.destroyAllWindows()

        # 세션 요약
        print("\n" + "=" * 50)
        print(" Debug Session Summary")
        print("=" * 50)
        print(f" Session dir : {self.session_dir}")
        print(f" Total frames: {self.debug_frame_idx}")
        print(f" Pose count  : {self.pose_count}")
        print("=" * 50 + "\n")


def main():
    args = parse_args()

    rclpy.init()
    debugger = None
    try:
        debugger = PoseDebugger(args)
        debugger.run()
    except KeyboardInterrupt:
        pass
    finally:
        if debugger is not None:
            debugger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
