# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
6DoF pose estimation using FoundationPose with YOLO or Mask R-CNN segmentation.

Usage:
    python run_est.py --mesh_file model.obj --test_scene_dir ./scene \
        --mask_model yolo_seg.pt --mask_type yolo

    python run_est.py --mask_model model.pt --symmetry z180 --fix_z_axis
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import imageio
import numpy as np

from rotation_utils import (
    normalize_rotation as normalize_input_rotation,
    rotate_image as rotate_input_image,
    rotate_camera_matrix as rotate_input_camera_matrix,
)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MeshConfig:
    """Mesh 관련 설정."""
    file_path: str
    scale: float = 0.01


@dataclass
class SceneConfig:
    """Scene 관련 설정."""
    directory: str
    frame_start: int = 0
    frame_end: int = -1
    frame_step: int = 1
    rotation: str = 'none'


@dataclass
class PoseConfig:
    """Pose 추정 관련 설정."""
    est_refine_iter: int = 5
    track_refine_iter: int = 2
    use_tracking: bool = False
    symmetry: str = 'z180'
    symmetry_step: float = 5.0
    fix_z_axis: bool = True
    min_n_views: int = 40
    inplane_step: int = 60
    input_mode: str = 'rgb'  # 'rgb' or 'rgbd'
    use_mask_iou: bool = True  # Use mask IoU bonus in scoring
    use_light: bool = False  # False=constant shading (flat color), True=Phong shading


@dataclass
class MaskConfig:
    """Mask 생성 관련 설정."""
    model_path: str
    model_type: str = 'yolo'
    confidence: float = 0.9
    config_file: Optional[str] = None
    depth_refine: bool = False
    dilate_kernel: int = 0
    dilate_iterations: int = 2


@dataclass
class DebugConfig:
    """디버그 관련 설정."""
    level: int = 2
    directory: str = './debug'


@dataclass
class EstimationConfig:
    """전체 설정을 통합하는 최상위 설정 클래스."""
    mesh: MeshConfig
    scene: SceneConfig
    pose: PoseConfig
    mask: MaskConfig
    debug: DebugConfig

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> EstimationConfig:
        """argparse Namespace로부터 설정 객체 생성."""
        return cls(
            mesh=MeshConfig(file_path=args.mesh_file, scale=args.mesh_scale),
            scene=SceneConfig(
                directory=args.test_scene_dir,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                frame_step=args.frame_step,
                rotation=normalize_input_rotation(args.rotation),
            ),
            pose=PoseConfig(
                est_refine_iter=args.est_refine_iter,
                track_refine_iter=args.track_refine_iter,
                use_tracking=args.use_tracking,
                symmetry=args.symmetry,
                symmetry_step=args.symmetry_step,
                fix_z_axis=args.fix_z_axis,
                min_n_views=args.min_n_views,
                inplane_step=args.inplane_step,
                input_mode=args.input_mode,
                use_mask_iou=args.use_mask_iou,
                use_light=args.use_light,
            ),
            mask=MaskConfig(
                model_path=args.mask_model,
                model_type=args.mask_type,
                confidence=args.mask_conf,
                config_file=args.mask_config,
                depth_refine=args.mask_depth_refine,
                dilate_kernel=args.mask_dilate,
                dilate_iterations=args.mask_dilate_iter,
            ),
            debug=DebugConfig(level=args.debug, directory=args.debug_dir),
        )


# =============================================================================
# Rotation Utilities
# =============================================================================

class RotationUtils:
    """회전 관련 유틸리티 함수 모음."""

    @staticmethod
    def to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """3x3 회전 행렬을 오일러 각도(pitch, yaw, roll)로 변환."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            yaw = np.degrees(np.arctan2(-R[2, 0], sy))
            roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            yaw = np.degrees(np.arctan2(-R[2, 0], sy))
            roll = 0.0

        return pitch, yaw, roll

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """각도를 [-90, 90] 범위로 정규화."""
        if angle > 90:
            return angle - 180
        elif angle < -90:
            return angle + 180
        return angle

    @staticmethod
    def rot_x(angle_deg: float) -> np.ndarray:
        """X축 회전 행렬 생성."""
        a = np.radians(angle_deg)
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(a), -np.sin(a), 0],
            [0, np.sin(a), np.cos(a), 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rot_y(angle_deg: float) -> np.ndarray:
        """Y축 회전 행렬 생성."""
        a = np.radians(angle_deg)
        return np.array([
            [np.cos(a), 0, np.sin(a), 0],
            [0, 1, 0, 0],
            [-np.sin(a), 0, np.cos(a), 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rot_z(angle_deg: float) -> np.ndarray:
        """Z축 회전 행렬 생성."""
        a = np.radians(angle_deg)
        return np.array([
            [np.cos(a), -np.sin(a), 0, 0],
            [np.sin(a), np.cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


# =============================================================================
# Symmetry Utilities
# =============================================================================

class SymmetryGenerator:
    """대칭 변환 행렬 생성기."""

    AXIS_ROTATIONS = {
        'x': RotationUtils.rot_x,
        'y': RotationUtils.rot_y,
        'z': RotationUtils.rot_z,
    }

    @classmethod
    def create(cls, symmetry: str, angle_step: float = 5.0) -> np.ndarray:
        """
        대칭 변환 행렬 배열 생성.

        Args:
            symmetry: 대칭 타입 ('none', 'z', 'z180', 'xy', 'xyz' 등)
            angle_step: 연속 대칭 시 각도 단계 (도)

        Returns:
            (N, 4, 4) 변환 행렬 배열
        """
        if symmetry is None or symmetry == 'none':
            return np.array([np.eye(4)])

        symmetry = symmetry.lower()
        tfs = [np.eye(4)]

        # 단일 축 연속 대칭
        if symmetry in cls.AXIS_ROTATIONS:
            rot_fn = cls.AXIS_ROTATIONS[symmetry]
            for angle in np.arange(angle_step, 360, angle_step):
                tfs.append(rot_fn(angle))

        # 단일 축 180도 대칭
        elif symmetry.endswith('180') and symmetry[:-3] in cls.AXIS_ROTATIONS:
            axis = symmetry[:-3]
            tfs.append(cls.AXIS_ROTATIONS[axis](180))

        # 복합 대칭
        elif symmetry in ('xy', 'xz', 'yz', 'xyz'):
            tfs.extend(cls._create_combined_symmetry(symmetry))

        else:
            logging.warning(f"알 수 없는 대칭 타입: {symmetry}, 'none' 사용")

        return np.array(tfs)

    @classmethod
    def _create_combined_symmetry(cls, symmetry: str) -> List[np.ndarray]:
        """복합 대칭 변환 생성."""
        rx = RotationUtils.rot_x(180)
        ry = RotationUtils.rot_y(180)
        rz = RotationUtils.rot_z(180)

        transforms = {
            'xy': [rx, ry, rx @ ry],
            'xz': [rx, rz, rx @ rz],
            'yz': [ry, rz, ry @ rz],
            'xyz': [rx, ry, rz, rx @ ry, rx @ rz, ry @ rz, rx @ ry @ rz],
        }
        return transforms.get(symmetry, [])


# =============================================================================
# Pose Correction
# =============================================================================

class PoseCorrector:
    """Pose 보정 유틸리티."""

    # 180도 X축 플립 행렬 (상수)
    FLIP_X = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # 180도 Z축 플립 행렬 (상수)
    FLIP_Z = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    @classmethod
    def fix_z_axis_direction(cls, pose: np.ndarray) -> np.ndarray:
        """
        Z축이 카메라 방향을 향하도록 보정.

        벽면에 장착된 물체의 경우 Z축이 항상 카메라 쪽을 향해야 함.
        """
        z_axis = pose[:3, 2]
        if z_axis[2] > 0:  # Z축이 카메라 반대 방향
            pose = pose @ cls.FLIP_X
        return pose

    @classmethod
    def fix_pitch_yaw_sign(cls, pose: np.ndarray) -> np.ndarray:
        """
        pitch/yaw 부호 모호성 보정.

        준대칭 객체의 경우 일관된 부호 규칙 적용.
        pitch를 음수(-)로 통일.
        """
        pitch, _, _ = RotationUtils.to_euler(pose[:3, :3])
        if pitch > 0:  # 양수면 플립하여 음수로 통일
            pose = pose @ cls.FLIP_Z
        return pose

    @classmethod
    def convert_for_saving(cls, pose: np.ndarray) -> np.ndarray:
        """저장용 pose 변환. ob_in_cam을 그대로 반환."""
        return pose.copy()


# =============================================================================
# Visualization
# =============================================================================

class PoseVisualizer:
    """Pose 시각화 유틸리티."""

    def __init__(self, K: np.ndarray, bbox: np.ndarray, extents: np.ndarray):
        self.K = K
        self.bbox = bbox
        self.extents = extents
        self.axis_scale = max(extents)

    def create_visualization(
        self,
        color: np.ndarray,
        pose: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Pose, 마스크, 텍스트 오버레이가 포함된 시각화 이미지 생성."""
        from estimater import draw_posed_3d_box, draw_xyz_axis

        vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=pose, bbox=self.bbox)
        vis = self._draw_axes(vis, pose)
        vis = self._overlay_mask(vis, mask)
        vis = self._add_pose_text(vis, pose)

        return vis

    def _draw_axes(self, vis: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """좌표축 그리기."""
        from estimater import draw_xyz_axis

        return draw_xyz_axis(
            vis, ob_in_cam=pose, scale=self.axis_scale, K=self.K,
            thickness=3, transparency=0, is_input_rgb=True
        )

    def _overlay_mask(self, vis: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """마스크 오버레이 (빨강 + 노랑 윤곽선)."""
        mask_overlay = np.zeros_like(vis)
        mask_overlay[mask] = [255, 0, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)

        return vis

    def _add_pose_text(self, vis: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Pose 정보 텍스트 오버레이."""
        trans = pose[:3, 3]
        pitch, yaw, roll = RotationUtils.to_euler(pose[:3, :3])
        pitch = RotationUtils.normalize_angle(pitch)
        yaw = RotationUtils.normalize_angle(yaw)
        roll = RotationUtils.normalize_angle(roll)

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        font, color = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)

        texts = [
            (f'X: {trans[0] * 100:+6.2f} cm', 25),
            (f'Y: {trans[1] * 100:+6.2f} cm', 50),
            (f'Z: {trans[2] * 100:+6.2f} cm', 75),
            (f'Roll:  {roll:+7.2f} deg', 105),
            (f'Pitch: {pitch:+7.2f} deg', 130),
            (f'Yaw:   {yaw:+7.2f} deg', 155),
        ]

        for text, y in texts:
            cv2.putText(vis_bgr, text, (10, y), font, 0.6, color, 2)

        return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# Pipeline
# =============================================================================

class PoseEstimationPipeline:
    """Pose 추정 파이프라인."""

    def __init__(self, config: EstimationConfig):
        self.config = config
        self._setup_directories()
        self._setup_mask_csv()
        self._init_components()

    def _setup_mask_csv(self) -> None:
        """프레임별 마스크 confidence/성공 여부 CSV 초기화."""
        self.mask_csv_path = self.output_dirs['root'] / 'mask_confidence.csv'
        self.mask_csv_fields = [
            'frame_index',
            'frame_id',
            'mask_threshold',
            'mask_found',
            'confidence',
            'skipped',
            'reason',
        ]
        self.mask_total = 0
        self.mask_success = 0
        self.mask_fail = 0

        with self.mask_csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.mask_csv_fields)
            writer.writeheader()

    def _write_mask_csv(
        self,
        frame_idx: int,
        frame_id: str,
        mask_found: bool,
        confidence,
        skipped: bool,
        reason: str = '',
    ) -> None:
        """한 프레임의 마스크 결과를 즉시 CSV에 기록."""
        if confidence is None:
            confidence_value = ''
        else:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = str(confidence)

        row = {
            'frame_index': frame_idx,
            'frame_id': frame_id,
            'mask_threshold': self.config.mask.confidence,
            'mask_found': int(mask_found),
            'confidence': confidence_value,
            'skipped': int(skipped),
            'reason': reason,
        }

        with self.mask_csv_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.mask_csv_fields)
            writer.writerow(row)

        self.mask_total += 1
        if mask_found:
            self.mask_success += 1
        else:
            self.mask_fail += 1

    def _setup_directories(self) -> None:
        """출력 디렉토리 설정."""
        debug_dir = Path(self.config.debug.directory)
        self.output_dirs = {
            'root': debug_dir,
            'vis': debug_dir / 'track_vis',
            'ob_in_cam': debug_dir / 'ob_in_cam',
            'cam_in_ob': debug_dir / 'cam_in_ob',
        }

        for path in self.output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
            for f in path.iterdir():
                if f.is_file():
                    f.unlink()

    def _init_components(self) -> None:
        """파이프라인 구성 요소 초기화."""
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import (
            FoundationPose, ScorePredictor, PoseRefinePredictor,
            set_logging_format, set_seed
        )
        from datareader import YcbineoatReader
        from mask_generator import create_mask_generator

        set_logging_format()
        set_seed(0)

        # Mesh 로드
        self.mesh = self._load_mesh()
        self.extents = self.mesh.extents
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0)

        # FoundationPose 초기화
        symmetry_tfs = SymmetryGenerator.create(
            self.config.pose.symmetry,
            self.config.pose.symmetry_step
        )
        logging.info(f"대칭: {self.config.pose.symmetry} ({len(symmetry_tfs)}개 변환)")

        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            symmetry_tfs=symmetry_tfs,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=str(self.output_dirs['root']),
            debug=self.config.debug.level,
            glctx=dr.RasterizeCudaContext(),
            min_n_views=self.config.pose.min_n_views,
            inplane_step=self.config.pose.inplane_step,
            front_hemisphere_only=self.config.pose.fix_z_axis,
            use_mask_iou=self.config.pose.use_mask_iou,
            use_light=self.config.pose.use_light,
        )
        logging.info(f"FoundationPose 초기화 완료 (front_hemisphere_only={self.config.pose.fix_z_axis}, use_mask_iou={self.config.pose.use_mask_iou}, use_light={self.config.pose.use_light})")

        # 데이터 리더 초기화
        self.reader = YcbineoatReader(
            video_dir=self.config.scene.directory,
            shorter_side=None,
            zfar=np.inf
        )
        logging.info(f"{len(self.reader.color_files)}개 프레임 로드: {self.config.scene.directory}")

        # Input rotation setup: derive the K used by FoundationPose from
        # the original image size and the original camera matrix.
        if len(self.reader.color_files) == 0:
            raise RuntimeError(f"No RGB frames found: {self.config.scene.directory}")

        sample_color_original = self.reader.get_color(0)
        original_height, original_width = sample_color_original.shape[:2]
        self.K_original = np.asarray(self.reader.K).copy()
        self.K_used = rotate_input_camera_matrix(
            k_original=self.K_original,
            original_width=original_width,
            original_height=original_height,
            rotation=self.config.scene.rotation,
        )

        np.savetxt(self.output_dirs['root'] / 'cam_K_used.txt', self.K_used)
        logging.info(
            f"Input rotation: {self.config.scene.rotation} | "
            f"original size={original_width}x{original_height} | "
            f"K saved: {self.output_dirs['root'] / 'cam_K_used.txt'}"
        )

        # 마스크 생성기 초기화
        mask_kwargs = {}
        if self.config.mask.model_type == 'maskrcnn' and self.config.mask.config_file:
            mask_kwargs['config_file'] = self.config.mask.config_file

        self.mask_generator = create_mask_generator(
            model_path=self.config.mask.model_path,
            model_type=self.config.mask.model_type,
            conf_threshold=self.config.mask.confidence,
            **mask_kwargs
        )
        logging.info(f"마스크 생성기: {self.config.mask.model_type} (conf={self.config.mask.confidence})")

        # 시각화 도구
        self.visualizer = PoseVisualizer(self.K_used, self.bbox, self.extents)

    def _load_mesh(self):
        """Mesh 파일 로드 및 스케일 적용.

        Multi-material OBJ의 경우 각 재질의 Kd 색상을 vertex color로 변환하여
        단일 mesh로 병합합니다.
        """
        import trimesh

        mesh = trimesh.load(self.config.mesh.file_path)
        if isinstance(mesh, trimesh.Scene):
            # Multi-material OBJ: Kd 색상을 vertex color로 변환 후 병합
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
                # Fusion 360 export 시 뒤집힌 face normal 수정
                trimesh.repair.fix_normals(geom)
            mesh = mesh.dump(concatenate=True)
        else:
            trimesh.repair.fix_normals(mesh)

        # Face winding order 통일 및 메시 유효성 검증
        mesh.process(validate=True)

        # Unindex: 각 면이 고유 버텍스를 갖도록 분리
        # → 공유 버텍스의 노멀 평균화 방지, 경계면 색상 그라데이션 제거
        original_vc = np.array(mesh.visual.vertex_colors)  # (V, 4)
        new_verts = mesh.vertices[mesh.faces.flatten()]
        new_colors = original_vc[mesh.faces.flatten()]
        new_faces = np.arange(len(mesh.faces) * 3).reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=new_verts, faces=new_faces, process=False
        )
        mesh.visual.vertex_colors = new_colors

        if self.config.mesh.scale != 1.0:
            mesh.apply_scale(self.config.mesh.scale)
            logging.info(f"Mesh 스케일: {self.config.mesh.scale} (extents: {mesh.extents})")

        return mesh

    def _get_frame_indices(self) -> List[int]:
        """처리할 프레임 인덱스 목록 생성."""
        total = len(self.reader.color_files)
        end = self.config.scene.frame_end if self.config.scene.frame_end >= 0 else total

        return list(range(
            self.config.scene.frame_start,
            min(end, total),
            self.config.scene.frame_step
        ))

    def run(self) -> None:
        """Pose 추정 파이프라인 실행."""
        frame_indices = self._get_frame_indices()
        total_frames = len(self.reader.color_files)

        logging.info(
            f"{len(frame_indices)}개 프레임 처리 예정 "
            f"(step={self.config.scene.frame_step}, "
            f"range={self.config.scene.frame_start}-{frame_indices[-1] if frame_indices else 0})"
        )

        pose = None

        for i in frame_indices:
            logging.info(f"프레임 처리 중: {i}/{total_frames - 1}")

            result = self._process_frame(i, pose)
            if result is None:
                continue

            pose = result

        logging.info(f"완료! 결과 저장 위치: {self.output_dirs['root']}/")
        logging.info(f"Mask CSV: {self.mask_csv_path}")
        logging.info(
            f"마스크 통계: total={self.mask_total}, "
            f"success={self.mask_success}, fail={self.mask_fail}"
        )

    def _process_frame(self, frame_idx: int, prev_pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """단일 프레임 처리."""
        color = self.reader.get_color(frame_idx)

        if self.config.pose.input_mode == "rgb":
            depth = None
            logging.info(f"프레임 {frame_idx}: RGB-only 모드, Depth 로드 생략")
        else:
            depth = self.reader.get_depth(frame_idx)

        # Apply the same input rotation to RGB and aligned Depth.
        # K_used was computed once from the original image size in _init_components().
        color_original_shape = color.shape[:2]
        color = rotate_input_image(color, self.config.scene.rotation)

        if depth is not None:
            if depth.shape[:2] != color_original_shape:
                raise ValueError(
                    f"Frame {frame_idx}: RGB/Depth size mismatch before rotation: "
                    f"rgb={color_original_shape}, depth={depth.shape[:2]}"
                )
            depth = rotate_input_image(depth, self.config.scene.rotation)

        logging.info(
            f"프레임 {frame_idx}: rotation={self.config.scene.rotation}, "
            f"RGB shape={color.shape[:2]}, "
            f"Depth shape={None if depth is None else depth.shape[:2]}"
        )

        # 마스크 생성
        # mask, mask_info = self.mask_generator.get_mask_with_depth(
        #     color, depth, depth_refine=self.config.mask.depth_refine)

        if depth is None:
            mask, mask_info = self.mask_generator.get_mask_with_depth(
                color,
                None,
                depth_refine=False,
            )
        else:
            mask, mask_info = self.mask_generator.get_mask_with_depth(
                color,
                depth,
                depth_refine=self.config.mask.depth_refine,
            )

        mask_info = mask_info or {}
        frame_id = self.reader.id_strs[frame_idx]
        mask_confidence = mask_info.get('confidence', None)

        if mask is None:
            reason = mask_info.get('error', 'mask_not_found')
            self._write_mask_csv(
                frame_idx=frame_idx,
                frame_id=frame_id,
                mask_found=False,
                confidence=mask_confidence,
                skipped=True,
                reason=reason,
            )
            logging.warning(
                f"프레임 {frame_idx} ({frame_id}): 마스크 없음 - {reason}"
            )
            return None

        self._write_mask_csv(
            frame_idx=frame_idx,
            frame_id=frame_id,
            mask_found=True,
            confidence=mask_confidence,
            skipped=False,
            reason='',
        )

        if mask_confidence is None:
            conf_text = 'N/A'
        else:
            try:
                conf_text = f"{float(mask_confidence):.4f}"
            except (TypeError, ValueError):
                conf_text = str(mask_confidence)

        logging.info(
            f"프레임 {frame_idx} ({frame_id}): 마스크 OK (conf={conf_text})"
        )

        # 마스크 팽창 (dilation) - tight mask 문제 해결
        if self.config.mask.dilate_kernel > 0:
            kernel_size = self.config.mask.dilate_kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=self.config.mask.dilate_iterations)
            mask = (mask_dilated > 127)
            logging.info(f"프레임 {frame_idx}: 마스크 팽창 (kernel={kernel_size}, iter={self.config.mask.dilate_iterations})")
        else:
            mask = mask.astype(bool)

        # Pose 추정
        pose = self._estimate_pose(color, depth, mask, prev_pose)

        # Pose 보정
        pose = self._correct_pose(pose)

        # 결과 저장
        self._save_results(frame_idx, pose, color, mask)

        return pose

    def _estimate_pose(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        prev_pose: Optional[np.ndarray]
    ) -> np.ndarray:
        """Pose 추정 수행."""
        # RGB 모드: depth=None으로 RGB-only 추론
        if self.config.pose.input_mode == 'rgb':
            depth_input = None
            logging.info("RGB 모드: depth=None (RGB-only)")
        else:
            depth_input = depth

        if prev_pose is None or not self.config.pose.use_tracking:
            pose = self.estimator.register(
                K=self.K_used,
                rgb=color,
                depth=depth_input,
                ob_mask=mask,
                iteration=self.config.pose.est_refine_iter
            )
            self._debug_export(pose, depth, color)
        else:
            pose = self.estimator.track_one(
                rgb=color,
                depth=depth_input,
                K=self.K_used,
                iteration=self.config.pose.track_refine_iter
            )
        return pose

    def _correct_pose(self, pose: np.ndarray) -> np.ndarray:
        """Pose 보정 적용."""
        if self.config.pose.fix_z_axis:
            pose = PoseCorrector.fix_z_axis_direction(pose)

        # 180도 대칭 사용 시 pitch/yaw 부호 보정 자동 적용
        symmetry = self.config.pose.symmetry
        if symmetry and '180' in symmetry:
            pose = PoseCorrector.fix_pitch_yaw_sign(pose)
        return pose

    def _save_results(
        self,
        frame_idx: int,
        pose: np.ndarray,
        color: np.ndarray,
        mask: np.ndarray
    ) -> None:
        """결과 저장 (pose 행렬 및 시각화)."""
        pose_save = PoseCorrector.convert_for_saving(pose)
        frame_id = self.reader.id_strs[frame_idx]

        np.savetxt(self.output_dirs['ob_in_cam'] / f'{frame_id}.txt', pose_save)
        np.savetxt(self.output_dirs['cam_in_ob'] / f'{frame_id}.txt', np.linalg.inv(pose_save))

        if self.config.debug.level >= 1:
            vis = self.visualizer.create_visualization(color, pose, mask)
            if self.config.debug.level >= 2:
                imageio.imwrite(self.output_dirs['vis'] / f'{frame_id}.png', vis)

    def _debug_export(self, pose: np.ndarray, depth: np.ndarray, color: np.ndarray) -> None:
        """디버그용 mesh/pointcloud 내보내기."""
        if self.config.debug.level < 3:
            return

        try:
            import open3d as o3d
            from estimater import depth2xyzmap, toOpen3dCloud

            m = self.mesh.copy()
            m.apply_transform(pose)
            m.export(str(self.output_dirs['root'] / 'model_tf.obj'))

            xyz_map = depth2xyzmap(depth, self.K_used)
            valid = depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(str(self.output_dirs['root'] / 'scene_complete.ply'), pcd)
        except ImportError:
            pass


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """명령줄 인자 파싱."""
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description='FoundationPose를 이용한 6DoF Pose 추정',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mesh 설정
    mesh = parser.add_argument_group('Mesh')
    mesh.add_argument('--mesh_file', type=str,
        default=f'{code_dir}/vcb/ref_views/ob_000001/model/model_vc.ply')
    mesh.add_argument('--mesh_scale', type=float, default=0.01)

    # Scene 설정
    scene = parser.add_argument_group('Scene')
    scene.add_argument('--test_scene_dir', type=str,
        default=f'{code_dir}/vcb/ref_views/test_scene')
    scene.add_argument('--frame_step', type=int, default=1)
    scene.add_argument('--frame_start', type=int, default=0)
    scene.add_argument('--frame_end', type=int, default=-1)
    scene.add_argument(
        '--rotation',
        type=str,
        default='none',
        choices=['none', '90_cw'],
        help=(
            'Input rotation applied consistently to RGB, aligned Depth, and K. '
            'Use 90_cw when stored RGB/Depth are 90 degrees CCW.'
        ),
    )

    # Pose 설정
    pose = parser.add_argument_group('Pose Estimation')
    pose.add_argument('--est_refine_iter', type=int, default=5)
    pose.add_argument('--track_refine_iter', type=int, default=2)
    pose.add_argument('--use_tracking', action='store_true')
    pose.add_argument('--symmetry', type=str, default='z180',
        choices=['none', 'z', 'z180', 'x', 'x180', 'y', 'y180', 'xy', 'xz', 'yz', 'xyz'])
    pose.add_argument('--symmetry_step', type=float, default=5.0)
    pose.add_argument('--fix_z_axis', type=lambda x: x.lower() == 'true', default=True,
        help='Filter back-facing poses (front hemisphere only)')
    pose.add_argument('--min_n_views', type=int, default=40,
        help='Number of viewpoints for pose hypothesis (default: 40)')
    pose.add_argument('--inplane_step', type=int, default=60,
        help='In-plane rotation step in degrees (default: 60)')
    pose.add_argument('--input_mode', type=str, default='rgb',
        choices=['rgb', 'rgbd'], help='Input mode: rgb (RGB only) or rgbd (RGB + Depth)')
    pose.add_argument('--use_mask_iou', type=lambda x: x.lower() == 'true', default=True,
        help='Use mask IoU bonus in scoring')
    pose.add_argument('--use_light', type=lambda x: x.lower() == 'true', default=False,
        help='Use Phong shading (True) or constant shading (False)')

    # Mask 설정
    mask = parser.add_argument_group('Mask Generation')
    mask.add_argument('--mask_model', type=str, default='weights/2026-02-12-13-41-52/model_best.pth')
    mask.add_argument('--mask_type', type=str, default='maskrcnn', choices=['yolo', 'maskrcnn'])
    mask.add_argument('--mask_conf', type=float, default=0.9)
    mask.add_argument('--mask_config', type=str, default=None)
    mask.add_argument('--mask_depth_refine', type=lambda x: x.lower() == 'true', default=False,
        help='Refine mask using depth information (default: False)')
    mask.add_argument('--mask_dilate', type=int, default=0,
        help='Mask dilation kernel size (0=disabled, 3~7 recommended)')
    mask.add_argument('--mask_dilate_iter', type=int, default=2,
        help='Mask dilation iterations')

    # Debug 설정
    debug = parser.add_argument_group('Debug')
    debug.add_argument('--debug', type=int, default=2)
    debug.add_argument('--debug_dir', type=str, default=f'{code_dir}/vcb/debug')

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """메인 진입점."""
    args = parse_args()
    config = EstimationConfig.from_args(args)
    pipeline = PoseEstimationPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()