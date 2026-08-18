#!/usr/bin/env python3
"""
RealSense 카메라 실시간 6D Pose 추정

ROS RealSense 카메라에서 실시간으로 화면을 받아오다가
버튼을 누르면 FoundationPose를 이용해 6DoF pose를 추정합니다.
추정 과정의 각 단계(R-CNN, Refiner, Scorer 등)를 실시간 모니터링합니다.

사용법:
    # 원본 방향 그대로 사용
    python camera/pose_estimator.py

    # 물리적으로 반시계 방향 90도 설치된 카메라 보정
    python camera/pose_estimator.py --rotation 90_cw

키 조작:
    'p' 또는 스페이스: Pose 추정 (단일)
    't': 트래킹 모드 ON/OFF (연속 추정)
    'r': 리셋 (트래킹 초기화)
    's': 현재 프레임 + 결과 저장
    'q': 종료

사전 준비:
    1. ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
    2. FoundationPose weights가 weights/ 에 있어야 함
"""

import sys
import os

# FoundationPose 루트를 path에 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# FoundationPose imports
import trimesh
import nvdiffrast.torch as dr
from estimater import (
    FoundationPose, ScorePredictor, PoseRefinePredictor,
    set_logging_format, set_seed,
    draw_posed_3d_box, draw_xyz_axis,
)
from mask_generator import create_mask_generator
from rotation_utils import normalize_rotation, rotate_image, rotate_camera_matrix


# =========================================================================
# 유틸리티 (run_est.py 에서 가져옴)
# =========================================================================

class RotationUtils:
    @staticmethod
    def to_euler(R_mat):
        sy = np.sqrt(R_mat[0, 0] ** 2 + R_mat[1, 0] ** 2)
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(R_mat[2, 1], R_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-R_mat[2, 0], sy))
            roll = np.degrees(np.arctan2(R_mat[1, 0], R_mat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R_mat[1, 2], R_mat[1, 1]))
            yaw = np.degrees(np.arctan2(-R_mat[2, 0], sy))
            roll = 0.0
        return pitch, yaw, roll

    @staticmethod
    def normalize_angle(angle):
        if angle > 90:
            return angle - 180
        elif angle < -90:
            return angle + 180
        return angle

    @staticmethod
    def rot_x(a_deg):
        a = np.radians(a_deg)
        return np.array([[1,0,0,0],[0,np.cos(a),-np.sin(a),0],
                         [0,np.sin(a),np.cos(a),0],[0,0,0,1]], dtype=np.float32)

    @staticmethod
    def rot_z(a_deg):
        a = np.radians(a_deg)
        return np.array([[np.cos(a),-np.sin(a),0,0],[np.sin(a),np.cos(a),0,0],
                         [0,0,1,0],[0,0,0,1]], dtype=np.float32)


class SymmetryGenerator:
    AXIS_FN = {'x': RotationUtils.rot_x, 'z': RotationUtils.rot_z}

    @classmethod
    def create(cls, symmetry, step=5.0):
        if symmetry is None or symmetry == 'none':
            return np.array([np.eye(4)])
        symmetry = symmetry.lower()
        tfs = [np.eye(4)]
        if symmetry in cls.AXIS_FN:
            for angle in np.arange(step, 360, step):
                tfs.append(cls.AXIS_FN[symmetry](angle))
        elif symmetry.endswith('180') and symmetry[:-3] in cls.AXIS_FN:
            tfs.append(cls.AXIS_FN[symmetry[:-3]](180))
        return np.array(tfs, dtype=np.float32)


FLIP_X = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64)
FLIP_Z = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)


# =========================================================================
# 상태 모니터
# =========================================================================

class StatusMonitor:
    """파이프라인 단계별 상태 추적."""

    # 상태 정의: (텍스트, BGR 색상)
    IDLE      = ("READY",                    (200, 200, 200))
    MASKING   = ("R-CNN Masking...",          (0, 200, 255))   # 주황
    MASK_FAIL = ("ERROR: Object not found!", (0, 0, 255))     # 빨강
    POSE_INIT = ("Pose Initializing...",     (255, 200, 0))   # 하늘
    REFINING  = ("Refiner Running...",       (255, 150, 0))   # 파랑
    SCORING   = ("Scorer Running...",        (200, 100, 255)) # 보라
    TRACKING  = ("Tracking (Refiner)...",    (255, 150, 0))   # 파랑
    DONE      = ("Done!",                    (0, 220, 0))     # 초록

    def __init__(self):
        self._text, self._color = self.IDLE
        self._step_start = 0.0
        self._total_start = 0.0
        self._elapsed_ms = 0.0
        self._total_ms = 0.0
        self._lock = threading.Lock()

    def set(self, state):
        with self._lock:
            self._text, self._color = state
            self._step_start = time.time()

    def begin(self):
        """전체 추정 시작 시각 기록."""
        self._total_start = time.time()

    def finish(self):
        """전체 추정 완료."""
        self._total_ms = (time.time() - self._total_start) * 1000
        self.set(self.DONE)

    @property
    def text(self):
        with self._lock:
            return self._text

    @property
    def color(self):
        with self._lock:
            return self._color

    @property
    def step_elapsed_ms(self):
        with self._lock:
            if self._step_start > 0:
                return (time.time() - self._step_start) * 1000
            return 0.0

    @property
    def total_ms(self):
        return self._total_ms

    def is_busy(self):
        with self._lock:
            return (self._text, self._color) not in (
                self.IDLE, self.DONE, self.MASK_FAIL
            )


# =========================================================================
# 메인 클래스
# =========================================================================

class RealtimePoseEstimator(Node):
    """ROS RealSense 카메라 + FoundationPose 실시간 6D Pose 추정."""

    def __init__(self, args):
        super().__init__('pose_estimator')

        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # 카메라 상태
        self.rotation = normalize_rotation(self.args.rotation)
        self.K_original = None
        self.K = None                 # 회전 보정 후 실제 추론에 사용할 K
        self.current_rgb = None       # 회전 보정 후 BGR (OpenCV)
        self.current_depth = None     # 회전 보정 후 uint16 mm
        self.frame_stamp = None

        # Pose 상태
        self.last_pose = None
        self.last_confidence = 0.0
        self.is_tracking = False
        self.pose_count = 0

        # 상태 모니터
        self.status = StatusMonitor()

        # 시각화용
        self.vis_frame = None         # 표시할 프레임 (결과 오버레이 포함)

        # 워커 스레드 플래그
        self._worker_busy = False

        logging.info(f"입력 회전 보정: {self.rotation}")

        # FoundationPose 초기화
        logging.info("FoundationPose 초기화 중...")
        set_logging_format()
        set_seed(0)
        self._init_model()
        self._install_hooks()
        logging.info("초기화 완료!")

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/foundation_pose/pose',
            1,
        )
        self.result_pub = self.create_publisher(
            String,
            '/foundation_pose/result',
            1,
        )

        self.request_sub = self.create_subscription(
            String,
            '/foundation_pose/request',
            self._cb_pose_request,
            10,
        )

        self.get_logger().info(
            "FoundationPose request subscriber started: "
            "/foundation_pose/request"
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            self.args.rgb_topic,
            self._cb_rgb,
            image_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.args.depth_topic,
            self._cb_depth,
            image_qos,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.args.camera_info_topic,
            self._cb_info,
            image_qos,
        )

        # 저장 디렉토리
        self.save_dir = args.save_dir
        self.save_count = 0

    # ----- 모델 초기화 -----

    def _init_model(self):
        """FoundationPose + 마스크 생성기 초기화."""
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
            debug=0,
            debug_dir='/tmp/pose_estimator_debug',
            min_n_views=self.args.min_n_views,
            inplane_step=self.args.inplane_step,
            front_hemisphere_only=self.args.fix_z_axis,
            use_mask_iou=self.args.use_mask_iou,
            use_light=self.args.use_light,
        )

        self.mask_generator = create_mask_generator(
            model_path=self.args.mask_model,
            model_type=self.args.mask_type,
            conf_threshold=self.args.mask_conf,
        )
        logging.info(f"마스크: {self.args.mask_type} (conf={self.args.mask_conf})")

    def _install_hooks(self):
        """Scorer/Refiner의 predict()에 상태 모니터링 훅 설치."""
        # Refiner 훅
        _orig_refiner = self.estimator.refiner.predict
        def _hooked_refiner(*a, **kw):
            self.status.set(StatusMonitor.REFINING)
            result = _orig_refiner(*a, **kw)
            return result
        self.estimator.refiner.predict = _hooked_refiner

        # Scorer 훅 (confidence 캡처 포함)
        _orig_scorer = self.estimator.scorer.predict
        _self = self
        def _hooked_scorer(*a, **kw):
            _self.status.set(StatusMonitor.SCORING)
            result = _orig_scorer(*a, **kw)
            # result = (scores_tensor, vis_canvas or None)
            try:
                scores = result[0]
                _self.last_confidence = float(scores.max().cpu())
            except Exception:
                pass
            return result
        self.estimator.scorer.predict = _hooked_scorer

    def _load_mesh(self):
        """run_est.py와 동일한 메시 로딩 파이프라인."""
        mesh = trimesh.load(self.args.mesh_file)
        if isinstance(mesh, trimesh.Scene):
            for geom in mesh.geometry.values():
                if isinstance(geom.visual, trimesh.visual.texture.TextureVisuals):
                    mat = geom.visual.material
                    if hasattr(mat, 'diffuse') and mat.diffuse is not None:
                        diffuse = mat.diffuse[:3]
                        rgba = np.tile(
                            np.append(diffuse, 255).astype(np.uint8),
                            (len(geom.vertices), 1),
                        )
                        geom.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
                trimesh.repair.fix_normals(geom)
            mesh = mesh.dump(concatenate=True)
        else:
            trimesh.repair.fix_normals(mesh)

        # Winding order 통일 + 유효성 검증
        mesh.process(validate=True)

        # Unindex: 면별 고유 버텍스 → 노멀 평균화 방지
        original_vc = np.array(mesh.visual.vertex_colors)
        new_verts = mesh.vertices[mesh.faces.flatten()]
        new_colors = original_vc[mesh.faces.flatten()]
        new_faces = np.arange(len(mesh.faces) * 3).reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=new_verts, faces=new_faces, process=False
        )
        mesh.visual.vertex_colors = new_colors

        if self.args.mesh_scale != 1.0:
            mesh.apply_scale(self.args.mesh_scale)
            logging.info(f"Mesh 스케일: {self.args.mesh_scale} (extents: {mesh.extents})")

        return mesh

    # ----- ROS 콜백 -----
    def _cb_pose_request(self, msg):
        """
        YOLO perception node에서 FoundationPose inference 요청 수신.

        예상 JSON:
        {
            "command_id": 1,
            "target_label": "4SW02-01B",
            "current_state": "OPEN",
            "desired_state": "CLOSE"
        }
        """

        try:
            request_context = json.loads(msg.data)

        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f"Invalid FoundationPose request JSON: {exc}"
            )
            return

        command_id = request_context.get(
            "command_id"
        )

        target_label = str(
            request_context.get(
                "target_label",
                "",
            )
        ).strip().upper()

        current_state = str(
            request_context.get(
                "current_state",
                "",
            )
        ).strip().upper()

        desired_state = str(
            request_context.get(
                "desired_state",
                "",
            )
        ).strip().upper()

        if not target_label:
            self.get_logger().error(
                "FoundationPose request has empty target_label."
            )
            return

        self.get_logger().info(
            "FoundationPose request received: "
            f"id={command_id}, "
            f"target={target_label}, "
            f"current={current_state}, "
            f"desired={desired_state}"
        )

        accepted = self.request_estimate(
            request_context=request_context
        )

        if not accepted:
            self.get_logger().warn(
                "FoundationPose request ignored: "
                "estimator is already busy."
            )

    def _cb_info(self, msg):
        """원본 CameraInfo K를 입력 영상 회전에 맞게 한 번 변환."""
        if self.K is not None:
            return

        K_original = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        K_used = rotate_camera_matrix(
            K_original,
            original_width=int(msg.width),
            original_height=int(msg.height),
            rotation=self.rotation,
        )

        with self.lock:
            self.K_original = K_original
            self.K = K_used

        logging.info(
            "카메라 intrinsics 수신 "
            f"(원본 {msg.width}x{msg.height}, rotation={self.rotation})\n"
            f"K original:\n{self.K_original}\n"
            f"K used:\n{self.K}"
        )

    def _cb_rgb(self, msg):
        """ROS RGB를 BGR로 변환한 뒤 추론 전에 회전 보정."""
        try:
            rgb_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb_bgr = rotate_image(rgb_bgr, self.rotation)

            with self.lock:
                self.current_rgb = rgb_bgr
                self.frame_stamp = msg.header.stamp
        except (CvBridgeError, ValueError) as e:
            self.get_logger().error(f"RGB 변환/회전 오류: {e}")

    def _cb_depth(self, msg):
        """Aligned depth를 RGB와 동일한 방향으로 회전 보정."""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            depth = rotate_image(depth, self.rotation)

            with self.lock:
                self.current_depth = depth
        except (CvBridgeError, ValueError) as e:
            self.get_logger().error(f"Depth 변환/회전 오류: {e}")

    # ----- Pose 추정 (워커 스레드에서 실행) -----

    def _estimate_worker(
        self,
        request_context=None,
    ):
        """백그라운드 스레드에서 pose 추정."""

        try:
            self._estimate_pose_impl(
                request_context=request_context
            )

        except Exception as e:
            logging.error(
                f"Pose 추정 오류: {e}"
            )

            self.status.set(
                StatusMonitor.MASK_FAIL
            )

            # ROS request로 실행된 추론이라면
            # 예외가 발생해도 상위 node가 영원히 기다리지 않도록
            # 실패 결과를 반환한다.
            if request_context is not None:
                try:
                    self._publish_result_json(
                        object_found=False,
                        request_context=request_context,
                    )

                except Exception as pub_exc:
                    logging.error(
                        "Pose 실패 결과 publish 오류: "
                        f"{pub_exc}"
                    )

        finally:
            self._worker_busy = False

    def request_estimate(
        self,
        request_context=None,
    ):
        """
        Pose 추정 요청 (비동기).

        request_context:
            YOLO에서 전달된 command metadata.
            키보드 수동 실행이면 None 가능.
        """

        if self._worker_busy:
            return False

        self._worker_busy = True

        t = threading.Thread(
            target=self._estimate_worker,
            args=(request_context,),
            daemon=True,
        )

        t.start()

        return True

    def _estimate_pose_impl(
        self, 
        request_context=None
        ):
        """Pose 추정 구현 (단계별 상태 업데이트 포함)."""
        if self.current_rgb is None:
            logging.warning("RGB 프레임 없음")
            self.status.set(StatusMonitor.IDLE)

            # ROS request로 실행된 경우에는 반드시 결과 반환
            if request_context is not None:
                self._publish_result_json(
                    object_found=False,
                    request_context=request_context,
                )

            return

        if self.K is None:
            logging.warning("카메라 정보 대기 중...")
            self.status.set(StatusMonitor.IDLE)

            # ROS request로 실행된 경우에는 반드시 결과 반환
            if request_context is not None:
                self._publish_result_json(
                    object_found=False,
                    request_context=request_context,
                )

            return

        self.status.begin()

        with self.lock:
            rgb_bgr = self.current_rgb.copy()
            depth_raw = self.current_depth.copy() if self.current_depth is not None else None
            K_used = self.K.copy()

        # BGR → RGB
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Depth 변환 (mm → m)
        depth = None
        if depth_raw is not None and self.args.input_mode == 'rgbd':
            depth = depth_raw.astype(np.float32) / 1000.0

        # ── 1단계: R-CNN 마스킹 ──
        self.status.set(StatusMonitor.MASKING)
        mask, mask_info = self.mask_generator.get_mask_with_depth(
            rgb, depth, depth_refine=self.args.mask_depth_refine)

        if mask is None:
            logging.warning("R-CNN: 객체를 찾을 수 없습니다!")
            self.status.set(StatusMonitor.MASK_FAIL)
            self._publish_result_json(
                object_found=False,
                request_context=request_context,
            )
            return

        mask_conf = mask_info.get('confidence', None)
        logging.info(f"Mask R-CNN confidence: {mask_conf:.4f}" if mask_conf is not None else "Mask confidence: N/A")

        # 마스크 팽창
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
            # ── 트래킹 모드 ──
            self.status.set(StatusMonitor.TRACKING)
            pose = self.estimator.track_one(
                rgb=rgb, depth=depth_input, K=K_used,
                iteration=self.args.track_refine_iter
            )
        else:
            # ── 2단계: Pose 초기화 (hypothesis 생성) ──
            # register() 내부에서 refiner.predict → scorer.predict 순으로
            # 호출되며, 각각 설치된 훅이 상태를 자동 업데이트합니다.
            self.status.set(StatusMonitor.POSE_INIT)
            pose = self.estimator.register(
                K=K_used, rgb=rgb, depth=depth_input,
                ob_mask=mask, iteration=self.args.est_refine_iter
            )

        # ── 5단계: 완료 ──
        pose = self._correct_pose(pose)
        self.last_pose = pose
        self.pose_count += 1
        self.status.finish()

        # ROS 발행
        self._publish_pose(pose)

        self._publish_result_json(
            object_found=True,
            pose=pose,
            request_context=request_context,
        )

        # GUI 모드에서만 시각화 frame 생성
        if not self.args.headless:
            self.vis_frame = self._make_vis(
                rgb,
                pose,
                mask,
                K_used,
            )

        logging.info(
            f"[#{self.pose_count}] 완료 (총 {self.status.total_ms:.0f}ms)"
        )

    def _correct_pose(self, pose):
        if self.args.fix_z_axis:
            z_axis = pose[:3, 2]
            if z_axis[2] > 0:
                pose = pose @ FLIP_X
        if self.args.symmetry and '180' in self.args.symmetry:
            pitch, _, _ = RotationUtils.to_euler(pose[:3, :3])
            if pitch > 0:
                pose = pose @ FLIP_Z
        return pose

    def _publish_pose(self, pose):
        from scipy.spatial.transform import Rotation as R
        trans = pose[:3, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()

        msg = PoseStamped()
        msg.header.stamp = self.frame_stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.camera_frame
        msg.pose.position.x = trans[0]
        msg.pose.position.y = trans[1]
        msg.pose.position.z = trans[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(msg)

    def _publish_result_json(
        self,
        object_found,
        pose=None,
        request_context=None,
    ):
        """JSON 형태로 추정 결과를 ROS topic에 발행."""
        from scipy.spatial.transform import Rotation as R

        result = {
            "object_found": object_found,
            "error": not object_found,
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
            "rotation": self.rotation,
        }
        
        if request_context is not None:
            result["command_id"] = (
                request_context.get(
                    "command_id"
                )
            )

            result["target_label"] = (
                request_context.get(
                    "target_label"
                )
            )

            result["current_state"] = (
                request_context.get(
                    "current_state"
                )
            )

            result["desired_state"] = (
                request_context.get(
                    "desired_state"
                )
            )

        if object_found and pose is not None:
            trans = pose[:3, 3]
            quat = R.from_matrix(pose[:3, :3]).as_quat()  # [x, y, z, w]
            pitch, yaw, roll = RotationUtils.to_euler(pose[:3, :3])
            pitch = RotationUtils.normalize_angle(pitch)
            yaw = RotationUtils.normalize_angle(yaw)
            roll = RotationUtils.normalize_angle(roll)

            result["pose_6d"] = {
                "translation": {
                    "x": round(float(trans[0]), 6),
                    "y": round(float(trans[1]), 6),
                    "z": round(float(trans[2]), 6),
                },
                "rotation_euler_deg": {
                    "roll": round(float(roll), 2),
                    "pitch": round(float(pitch), 2),
                    "yaw": round(float(yaw), 2),
                },
                "rotation_quaternion": {
                    "x": round(float(quat[0]), 6),
                    "y": round(float(quat[1]), 6),
                    "z": round(float(quat[2]), 6),
                    "w": round(float(quat[3]), 6),
                },
            }
            # raw score = logit + 100(오프셋) + IoU보너스(0~10)
            # → 100을 빼서 logit 복원 후 sigmoid로 0~1 정규화
            raw = self.last_confidence - 100.0
            confidence = 1.0 / (1.0 + np.exp(-raw))
            result["confidence"] = round(float(confidence), 4)
        else:
            result["pose_6d"] = None
            result["confidence"] = 0.0

        msg = String()
        msg.data = json.dumps(result)
        self.result_pub.publish(msg)

    # ----- 시각화 -----

    def _draw_status_bar(self, frame):
        """프레임 상단에 상태 바 그리기."""
        h, w = frame.shape[:2]
        bar_h = 40

        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 상태 텍스트 + 경과시간
        text = self.status.text
        color = self.status.color

        if self.status.is_busy():
            elapsed = self.status.step_elapsed_ms
            text = f"{text} ({elapsed:.0f}ms)"

        cv2.putText(
            frame, text, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        # 진행 표시 (busy일 때 깜빡이는 점)
        if self.status.is_busy():
            n_dots = int(time.time() * 3) % 4
            dot_x = w - 60
            for i in range(n_dots):
                cv2.circle(frame, (dot_x + i * 15, 20), 4, color, -1)

        return frame

    def _draw_error_overlay(self, frame):
        """에러 상태일 때 빨간색 경고 오버레이."""
        h, w = frame.shape[:2]

        # 빨간 테두리
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)

        # 에러 메시지 배경
        msg_y = h // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, msg_y - 30), (w, msg_y + 30), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame, "Object Not Found!", (w // 2 - 160, msg_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        return frame

    def _make_vis(self, rgb, pose, mask, K_used):
        """Pose 오버레이 시각화 생성 (RGB → BGR 반환)."""
        vis = draw_posed_3d_box(
            K_used, img=rgb.copy(), ob_in_cam=pose, bbox=self.bbox
        )
        vis = draw_xyz_axis(
            vis, ob_in_cam=pose, scale=max(self.extents),
            K=K_used, thickness=3, transparency=0, is_input_rgb=True
        )

        # 마스크 오버레이
        mask_overlay = np.zeros_like(vis)
        mask_overlay[mask] = [255, 0, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)

        # Pose 텍스트
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        trans = pose[:3, 3]
        pitch, yaw, roll = RotationUtils.to_euler(pose[:3, :3])
        pitch = RotationUtils.normalize_angle(pitch)
        yaw = RotationUtils.normalize_angle(yaw)
        roll = RotationUtils.normalize_angle(roll)

        font, color = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
        texts = [
            (f'X: {trans[0]*100:+6.2f} cm', 55),
            (f'Y: {trans[1]*100:+6.2f} cm', 80),
            (f'Z: {trans[2]*100:+6.2f} cm', 105),
            (f'Roll:  {roll:+7.2f} deg', 135),
            (f'Pitch: {pitch:+7.2f} deg', 160),
            (f'Yaw:   {yaw:+7.2f} deg', 185),
        ]
        for text, y in texts:
            cv2.putText(vis_bgr, text, (10, y), font, 0.6, color, 2)

        return vis_bgr

    def _save_frame(self):
        """현재 프레임 + 결과 저장."""
        os.makedirs(self.save_dir, exist_ok=True)
        idx = f"{self.save_count:04d}"

        if self.current_rgb is not None:
            cv2.imwrite(os.path.join(self.save_dir, f'{idx}_rgb.png'), self.current_rgb)
        if self.current_depth is not None:
            cv2.imwrite(os.path.join(self.save_dir, f'{idx}_depth.png'), self.current_depth)
        if self.vis_frame is not None:
            cv2.imwrite(os.path.join(self.save_dir, f'{idx}_pose.png'), self.vis_frame)
        if self.last_pose is not None:
            np.savetxt(os.path.join(self.save_dir, f'{idx}_pose.txt'), self.last_pose)
        if self.K is not None and self.save_count == 0:
            # 저장되는 RGB/Depth는 이미 회전 보정된 상태이므로 cam_K.txt에는
            # 실제 사용 K를 기록합니다. 원본 K도 검증용으로 함께 남깁니다.
            np.savetxt(
                os.path.join(self.save_dir, 'cam_K.txt'),
                self.K,
                fmt='%.8f',
            )
            np.savetxt(
                os.path.join(self.save_dir, 'cam_K_used.txt'),
                self.K,
                fmt='%.8f',
            )
            if self.K_original is not None:
                np.savetxt(
                    os.path.join(self.save_dir, 'cam_K_original.txt'),
                    self.K_original,
                    fmt='%.8f',
                )

        self.save_count += 1
        logging.info(f"저장 완료: {self.save_dir}/{idx}_*")

    # ----- 메인 루프 -----

    def run(self):
        """실시간 화면 표시 + 키 입력 처리."""

        print("\n" + "=" * 50)
        print(" RealSense 실시간 6D Pose 추정")
        print(f" 입력 회전 보정: {self.rotation}")
        print("=" * 50)
        print(" 'p'/Space : Pose 추정 (단일)")
        print(" 't'       : 트래킹 모드 ON/OFF")
        print(" 'r'       : 리셋 (트래킹 초기화)")
        print(" 's'       : 현재 프레임 저장")
        print(" 'q'       : 종료")
        print("=" * 50 + "\n")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            if self.current_rgb is not None:
                # 트래킹 모드면 매 프레임 자동 추정 요청
                if self.is_tracking and self.K is not None and not self._worker_busy:
                    self.request_estimate()

                # 표시할 프레임 결정
                if self.vis_frame is not None:
                    display = self.vis_frame.copy()
                else:
                    display = self.current_rgb.copy()

                # 에러 오버레이 (마스크 검출 실패)
                if self.status.text == StatusMonitor.MASK_FAIL[0]:
                    display = self._draw_error_overlay(display)

                # 상태 바 (항상 표시)
                display = self._draw_status_bar(display)

                # 모드 표시 (우측 상단)
                h, w = display.shape[:2]
                if self.is_tracking:
                    mode_text = "MODE: TRACK"
                    mode_color = (0, 150, 255)
                else:
                    mode_text = "MODE: SINGLE"
                    mode_color = (200, 200, 200)
                cv2.putText(
                    display, mode_text,
                    (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, mode_color, 1
                )
                cv2.putText(
                    display, f"ROT: {self.rotation}",
                    (w - 180, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1
                )

                # 도움말 (하단)
                help_text = "'p':Pose  't':Track  'r':Reset  's':Save  'q':Quit"
                cv2.putText(
                    display, help_text,
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1
                )

                cv2.imshow("6D Pose Estimator", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('p') or key == 32:  # 'p' 또는 스페이스
                if not self.is_tracking and not self._worker_busy:
                    logging.info("Pose 추정 요청")
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
                self.estimator.pose_last = None
                self.status.set(StatusMonitor.IDLE)
                logging.info("리셋 완료")

            elif key == ord('s'):
                self._save_frame()

            elif key == ord('q'):
                logging.info("종료")
                break

            time.sleep(1.0 / 30.0)

        cv2.destroyAllWindows()


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    code_dir = _PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description='RealSense 실시간 6D Pose 추정',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument('--mesh_file', type=str,
        default=f'{code_dir}/vcb/ref_views/ob_000001/model/model_vc.ply')
    parser.add_argument('--mesh_scale', type=float, default=0.01)

    # GUI
    parser.add_argument(
        '--headless',
        action='store_true',
        help='GUI 없이 ROS request 기반으로만 FoundationPose 실행'
    )

    # Mask
    parser.add_argument('--mask_model', type=str,
        default=f'{code_dir}/weights/2026-02-12-13-41-52/model_best.pth')
    parser.add_argument('--mask_type', type=str, default='maskrcnn',
        choices=['yolo', 'maskrcnn'])
    parser.add_argument('--mask_conf', type=float, default=0.9)
    parser.add_argument('--mask_depth_refine', type=lambda x: x.lower() == 'true', default=False,
        help='Refine mask using depth information (default: False)')
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
    parser.add_argument('--use_light', type=lambda x: x.lower() == 'true', default=True,
        help='Use Phong shading (True) or constant shading (False)')

    # 저장
    parser.add_argument('--save_dir', type=str,
        default=f'{code_dir}/camera/output')

    return parser.parse_args()


# def main():
#     args = parse_args()

#     rclpy.init()
#     estimator = None
#     try:
#         estimator = RealtimePoseEstimator(args)
#         estimator.run()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         if estimator is not None:
#             estimator.destroy_node()
#         rclpy.shutdown()

def main():
    args = parse_args()

    rclpy.init()
    estimator = None

    try:
        estimator = RealtimePoseEstimator(args)

        if args.headless:
            estimator.get_logger().info(
                "FoundationPose HEADLESS mode started."
            )

            rclpy.spin(estimator)

        else:
            estimator.run()

    except KeyboardInterrupt:
        pass

    finally:
        if estimator is not None:
            estimator.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()