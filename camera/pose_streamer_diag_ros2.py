#!/usr/bin/env python3
"""
6D Pose 실시간 진단 스트리머 (pose_streamer_ros2.py 기반)

"Object not found"의 실제 원인을 찾기 위한 계측 버전입니다.
연속 자동 추정을 수행하면서, 매 시도마다 다음을 기록합니다.

  1. 실패 사유 분류
     - NO_FRAME        : RGB 프레임 미수신
     - NO_CAMERA_INFO  : CameraInfo 미수신
     - NO_DETECTION    : det_floor(기본 0.05)에서도 Mask R-CNN 검출 0개
     - BELOW_THRESHOLD : 검출은 됐지만 best score < 운영 임계값(mask_conf)
     - EXCEPTION       : register() 등 파이프라인 예외 (traceback 저장)
     - SUCCESS         : 정상 추정

  2. 매 시도 CSV 기록 (attempts.csv)
     - best score, 검출 수, 마스크 픽셀 수
     - 프레임 품질: 블러(Laplacian variance), 밝기 평균
     - register 소요 시간, 최종 pose confidence

  3. 실패/성공 프레임 덤프 (run_est_rotation.py로 그대로 재생 가능)
     - <session>/replay_fail/rgb/NNNNNN.png, depth/NNNNNN.png, cam_K.txt
     - <session>/replay_success/... (성공은 save_success_every 회마다 1회)
     - 덤프 프레임은 이미 회전 보정된 상태이므로 재생 시 --rotation none 사용

재생(오프라인 재현) 예:
    python3 run_est_rotation.py \
        --test_scene_dir camera/diag_sessions/<session>/replay_fail \
        --rotation none \
        --mask_conf 0.1 \
        --debug_dir vcb/debug_replay_fail

사용법:
    # 실험실 카메라 연결 후 (운영과 동일 조건: conf 0.9, 90도 보정)
    python3 camera/pose_streamer_diag_ros2.py --rotation 90_cw

    # Headless
    python3 camera/pose_streamer_diag_ros2.py --rotation 90_cw --headless
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import argparse
import csv
import logging
import time
import traceback

import cv2
import numpy as np
import rclpy

from pose_estimator_ros2 import StatusMonitor
from pose_streamer_ros2 import PoseStreamer


# 실패 사유 상수
REASON_SUCCESS = 'SUCCESS'
REASON_NO_FRAME = 'NO_FRAME'
REASON_NO_INFO = 'NO_CAMERA_INFO'
REASON_NO_DETECTION = 'NO_DETECTION'
REASON_BELOW_THRESHOLD = 'BELOW_THRESHOLD'
REASON_EXCEPTION = 'EXCEPTION'

CSV_FIELDS = [
    'attempt', 'timestamp', 'reason',
    'best_score', 'num_detections', 'mask_px',
    'blur_lapvar', 'brightness',
    'mask_ms', 'register_ms', 'total_ms',
    'pose_confidence', 'dump_id',
]


class PoseStreamerDiag(PoseStreamer):
    """실패 사유/프레임을 기록하는 진단 스트리머.

    주의: 운영 판정 임계값은 args.decision_conf 입니다.
    Mask R-CNN 자체는 args.det_floor(낮은 값)로 생성하여,
    임계값 미달 검출의 실제 score를 관측할 수 있게 합니다.
    """

    def __init__(self, args):
        super().__init__(args)

        self.decision_conf = args.decision_conf
        self.attempt_count = 0
        self.reason_counts = {}
        self.fail_scores = []
        self.last_fail_reason = None

        # 세션 디렉토리 구성
        session_name = time.strftime('session_%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(args.diag_dir, session_name)
        self.replay_fail_dir = os.path.join(self.session_dir, 'replay_fail')
        self.replay_success_dir = os.path.join(self.session_dir, 'replay_success')
        self.error_dir = os.path.join(self.session_dir, 'errors')

        for d in (self.session_dir, self.error_dir):
            os.makedirs(d, exist_ok=True)
        for d in (self.replay_fail_dir, self.replay_success_dir):
            os.makedirs(os.path.join(d, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(d, 'depth'), exist_ok=True)

        self.csv_path = os.path.join(self.session_dir, 'attempts.csv')
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_FIELDS)

        self._dump_count = 0

        logging.info(
            f"[DIAG] 세션 디렉토리: {self.session_dir}\n"
            f"[DIAG] 검출 floor={args.det_floor} / "
            f"운영 판정 임계값={self.decision_conf}"
        )

    # ----- 기록 유틸 -----

    def _write_csv(self, **row):
        row.setdefault('attempt', self.attempt_count)
        row.setdefault('timestamp', round(time.time(), 3))
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(k, '') for k in CSV_FIELDS])

    def _record_reason(self, reason, best_score=None):
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        if reason != REASON_SUCCESS:
            self.last_fail_reason = reason
            if best_score is not None:
                self.fail_scores.append(best_score)

    def _dump_frame(self, replay_dir, rgb_bgr, depth_raw, K_used):
        """YcbineoatReader 호환 구조로 프레임 저장. dump id 반환."""
        if self._dump_count >= self.args.max_dump:
            return ''

        dump_id = f"{self.attempt_count:06d}"
        cv2.imwrite(
            os.path.join(replay_dir, 'rgb', f'{dump_id}.png'), rgb_bgr
        )
        if depth_raw is not None:
            cv2.imwrite(
                os.path.join(replay_dir, 'depth', f'{dump_id}.png'), depth_raw
            )

        cam_k_path = os.path.join(replay_dir, 'cam_K.txt')
        if not os.path.exists(cam_k_path):
            np.savetxt(cam_k_path, K_used, fmt='%.8f')

        self._dump_count += 1
        return dump_id

    # ----- 워커: 예외를 사유로 기록 -----

    def _estimate_worker(self):
        try:
            self._estimate_pose_impl()
        except Exception:
            tb = traceback.format_exc()
            logging.error(f"[DIAG] EXCEPTION:\n{tb}")

            # traceback + GPU 메모리 상태 저장
            err_path = os.path.join(
                self.error_dir, f'attempt_{self.attempt_count:06d}.txt'
            )
            gpu_info = ''
            try:
                import torch
                gpu_info = (
                    f"\ncuda allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
                    f" reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"
                )
            except Exception:
                pass
            with open(err_path, 'w') as f:
                f.write(tb + gpu_info)

            self._record_reason(REASON_EXCEPTION)
            self._write_csv(reason=REASON_EXCEPTION)
            self.status.set(StatusMonitor.MASK_FAIL)
            self._publish_result_json(object_found=False)
        finally:
            self._worker_busy = False

    # ----- 진단 추정 구현 -----

    def _estimate_pose_impl(self):
        self.attempt_count += 1

        if self.current_rgb is None:
            logging.warning("RGB 프레임 없음")
            self.status.set(StatusMonitor.IDLE)
            self._record_reason(REASON_NO_FRAME)
            self._write_csv(reason=REASON_NO_FRAME)
            return

        if self.K is None:
            logging.warning("카메라 정보 대기 중...")
            self.status.set(StatusMonitor.IDLE)
            self._record_reason(REASON_NO_INFO)
            self._write_csv(reason=REASON_NO_INFO)
            return

        self.status.begin()
        t_start = time.time()

        with self.lock:
            rgb_bgr = self.current_rgb.copy()
            depth_raw = (
                self.current_depth.copy()
                if self.current_depth is not None
                else None
            )
            K_used = self.K.copy()

        # 프레임 품질 지표
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        blur_lapvar = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
        brightness = round(float(gray.mean()), 1)

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth = None
        if depth_raw is not None and self.args.input_mode == 'rgbd':
            depth = depth_raw.astype(np.float32) / 1000.0

        # ── 1단계: 마스킹 (det_floor로 검출 → decision_conf로 판정) ──
        self.status.set(StatusMonitor.MASKING)
        t_mask = time.time()
        mask, mask_info = self.mask_generator.get_mask_with_depth(
            rgb, depth, depth_refine=self.args.mask_depth_refine,
        )
        mask_ms = round((time.time() - t_mask) * 1000, 1)

        common = dict(
            blur_lapvar=blur_lapvar,
            brightness=brightness,
            mask_ms=mask_ms,
        )

        if mask is None:
            # det_floor(0.05)에서도 검출 자체가 없음
            logging.warning(
                f"[DIAG] NO_DETECTION (floor={self.args.det_floor}) | "
                f"blur={blur_lapvar} bright={brightness}"
            )
            dump_id = self._dump_frame(
                self.replay_fail_dir, rgb_bgr, depth_raw, K_used
            )
            self._record_reason(REASON_NO_DETECTION)
            self._write_csv(
                reason=REASON_NO_DETECTION, dump_id=dump_id, **common
            )
            self.status.set(StatusMonitor.MASK_FAIL)
            self._publish_result_json(object_found=False)
            return

        best_score = float(mask_info.get('confidence', 0.0))
        num_det = int(mask_info.get('num_detections', 0))

        if best_score < self.decision_conf:
            # 운영 임계값(0.9) 기준으로는 실패 → 실제 score를 기록
            logging.warning(
                f"[DIAG] BELOW_THRESHOLD: best={best_score:.4f} < "
                f"{self.decision_conf} (det={num_det}) | "
                f"blur={blur_lapvar} bright={brightness}"
            )
            dump_id = self._dump_frame(
                self.replay_fail_dir, rgb_bgr, depth_raw, K_used
            )
            self._record_reason(REASON_BELOW_THRESHOLD, best_score)
            self._write_csv(
                reason=REASON_BELOW_THRESHOLD,
                best_score=round(best_score, 4),
                num_detections=num_det,
                mask_px=int(mask.sum()),
                dump_id=dump_id,
                **common,
            )
            self.status.set(StatusMonitor.MASK_FAIL)
            self._publish_result_json(object_found=False)
            return

        logging.info(
            f"[DIAG] Mask OK: conf={best_score:.4f} (det={num_det}) | "
            f"blur={blur_lapvar} bright={brightness}"
        )

        if self.args.mask_dilate > 0:
            kernel = np.ones(
                (self.args.mask_dilate, self.args.mask_dilate),
                dtype=np.uint8,
            )
            mask_u8 = (mask * 255).astype(np.uint8)
            mask = cv2.dilate(mask_u8, kernel, iterations=2) > 127
        else:
            mask = mask.astype(bool)

        mask_px = int(mask.sum())
        depth_input = depth if self.args.input_mode == 'rgbd' else None

        # ── 2단계: register ──
        self.status.set(StatusMonitor.POSE_INIT)
        t_reg = time.time()
        pose = self.estimator.register(
            K=K_used,
            rgb=rgb,
            depth=depth_input,
            ob_mask=mask,
            iteration=self.args.est_refine_iter,
        )
        register_ms = round((time.time() - t_reg) * 1000, 1)

        pose = self._correct_pose(pose)
        self.last_pose = pose
        self.pose_count += 1
        self.status.finish()

        self._publish_pose(pose)
        self._publish_result_json(object_found=True, pose=pose)
        self._update_fps()

        # pose confidence (base의 sigmoid 정규화와 동일)
        raw = self.last_confidence - 100.0
        pose_conf = round(float(1.0 / (1.0 + np.exp(-raw))), 4)

        self._record_reason(REASON_SUCCESS)

        # 성공 프레임도 주기적으로 덤프 (실패 프레임과 비교용)
        dump_id = ''
        n_success = self.reason_counts.get(REASON_SUCCESS, 0)
        if self.args.save_success_every > 0 and \
                n_success % self.args.save_success_every == 1:
            dump_id = self._dump_frame(
                self.replay_success_dir, rgb_bgr, depth_raw, K_used
            )

        total_ms = round((time.time() - t_start) * 1000, 1)
        self._write_csv(
            reason=REASON_SUCCESS,
            best_score=round(best_score, 4),
            num_detections=num_det,
            mask_px=mask_px,
            register_ms=register_ms,
            total_ms=total_ms,
            pose_confidence=pose_conf,
            dump_id=dump_id,
            **common,
        )

        if not self.headless:
            self.vis_frame = self._make_vis(rgb, pose, mask, K_used)

        logging.info(
            f"[DIAG #{self.pose_count}] 완료 "
            f"({total_ms:.0f}ms, mask={best_score:.3f}, pose={pose_conf:.3f})"
        )

    # ----- GUI: 실패 사유 표시 -----

    def _draw_error_overlay(self, frame):
        frame = super()._draw_error_overlay(frame)
        if self.last_fail_reason:
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                f"reason: {self.last_fail_reason}",
                (max(10, w // 2 - 160), h // 2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        return frame

    # ----- 종료 시 요약 -----

    def print_summary(self):
        if getattr(self, '_summary_printed', False):
            return
        self._summary_printed = True

        total = sum(self.reason_counts.values())
        print("\n" + "=" * 56)
        print(" 진단 요약")
        print("=" * 56)
        print(f" 총 시도: {total}")
        for reason, cnt in sorted(
            self.reason_counts.items(), key=lambda x: -x[1]
        ):
            pct = 100.0 * cnt / total if total else 0.0
            print(f"   {reason:<16}: {cnt:5d} ({pct:.1f}%)")

        if self.fail_scores:
            arr = np.array(self.fail_scores)
            print(
                f" BELOW_THRESHOLD score: "
                f"min={arr.min():.3f} med={np.median(arr):.3f} "
                f"max={arr.max():.3f}"
            )

        print(f"\n CSV      : {self.csv_path}")
        print(f" 실패 재생: {self.replay_fail_dir}")
        print(f" 성공 재생: {self.replay_success_dir}")
        print(
            "\n 오프라인 재현:\n"
            f"   python3 run_est_rotation.py \\\n"
            f"       --test_scene_dir {self.replay_fail_dir} \\\n"
            f"       --rotation none --mask_conf 0.1 \\\n"
            f"       --debug_dir vcb/debug_replay_fail"
        )
        print("=" * 56 + "\n")

    def run(self):
        try:
            super().run()
        finally:
            self.print_summary()


def parse_args():
    code_dir = _PROJECT_ROOT

    parser = argparse.ArgumentParser(
        description='6D Pose 실시간 진단 스트리머',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ROS topics
    parser.add_argument('--rgb_topic', type=str,
        default='/camera/camera/color/image_raw')
    parser.add_argument('--depth_topic', type=str,
        default='/camera/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--camera_info_topic', type=str,
        default='/camera/camera/color/camera_info')
    parser.add_argument('--camera_frame', type=str,
        default='/camera/camera/camera_color_optical_frame')
    parser.add_argument('--rotation', type=str, default='none',
        choices=['none', '90_cw'])

    # Mesh
    parser.add_argument('--mesh_file', type=str,
        default=f'{code_dir}/vcb/ref_views/ob_000001/model/model_vc.ply')
    parser.add_argument('--mesh_scale', type=float, default=0.01)

    # Mask
    parser.add_argument('--mask_model', type=str,
        default=f'{code_dir}/weights/2026-02-12-13-41-52/model_best.pth')
    parser.add_argument('--mask_type', type=str, default='maskrcnn',
        choices=['yolo', 'maskrcnn'])
    parser.add_argument('--mask_conf', type=float, default=0.9,
        help='운영 판정 임계값 (이 값 미만이면 BELOW_THRESHOLD로 기록)')
    parser.add_argument('--mask_depth_refine',
        type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--mask_dilate', type=int, default=0)

    # Pose
    parser.add_argument('--input_mode', type=str, default='rgb',
        choices=['rgb', 'rgbd'])
    parser.add_argument('--symmetry', type=str, default='z180')
    parser.add_argument('--symmetry_step', type=float, default=5.0)
    parser.add_argument('--fix_z_axis', action='store_true', default=True)
    parser.add_argument('--use_mask_iou', action='store_true', default=True)
    parser.add_argument('--min_n_views', type=int, default=40)
    parser.add_argument('--inplane_step', type=int, default=60)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--use_light',
        type=lambda x: x.lower() == 'true', default=True)

    # 저장 (base 클래스 호환용)
    parser.add_argument('--save_dir', type=str,
        default=f'{code_dir}/camera/output')

    # 진단 전용
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--diag_dir', type=str,
        default=f'{code_dir}/camera/diag_sessions',
        help='진단 세션 저장 루트')
    parser.add_argument('--det_floor', type=float, default=0.05,
        help='Mask R-CNN 내부 검출 floor (score 관측용, 판정에는 미사용)')
    parser.add_argument('--save_success_every', type=int, default=20,
        help='성공 N회마다 1회 프레임 덤프 (0=저장 안 함)')
    parser.add_argument('--max_dump', type=int, default=500,
        help='프레임 덤프 최대 개수 (디스크 보호)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 운영 판정 임계값은 decision_conf로 보관하고,
    # Mask R-CNN predictor 자체는 det_floor로 생성한다.
    # (SCORE_THRESH_TEST가 predictor에 박히므로, 낮게 생성해야
    #  임계값 미달 검출의 실제 score를 관측할 수 있음)
    args.decision_conf = args.mask_conf
    args.mask_conf = args.det_floor

    rclpy.init()
    streamer = None

    try:
        streamer = PoseStreamerDiag(args)
        streamer.run()
    except KeyboardInterrupt:
        pass
    finally:
        if streamer is not None:
            streamer.print_summary()
            streamer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
