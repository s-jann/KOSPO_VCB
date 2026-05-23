#!/usr/bin/env python3
"""
RealSense 카메라로 RGB + Depth 이미지 캡처

사용법:
    python capture_100_frames.py [--output OUTPUT_DIR] [--frames NUM_FRAMES]

키 조작:
    's' 또는 스페이스: 한 장 캡처 (사진 모드)
    'b': 연속 녹화 시작
    'e': 연속 녹화 종료
    'q': 종료
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import os
import argparse
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


class FrameCapture(Node):
    def __init__(self, output_dir="captured_frames", max_frames=50,
                 rgb_topic="/camera/color/image_raw",
                 depth_topic="/camera/aligned_depth_to_color/image_raw",
                 camera_info_topic="/camera/color/camera_info"):
        super().__init__('frame_capture')

        self.output_dir = output_dir
        self.max_frames = max_frames
        self.frame_count = 0

        # 디렉토리 생성
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.bridge = CvBridge()

        # 카메라 매트릭스
        self.camera_matrix = None
        self.camera_matrix_saved = False

        # 현재 프레임
        self.current_rgb = None
        self.current_depth = None

        # 녹화 상태
        self.is_recording = False

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, image_qos
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, image_qos
        )
        self.info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.info_callback, image_qos
        )

        print(f"출력 디렉토리: {output_dir}")
        print(f"목표 프레임 수: {max_frames}")
        print("\n키 조작:")
        print("  's' 또는 스페이스: 한 장 캡처 (사진 모드)")
        print("  'b': 연속 녹화 시작")
        print("  'e': 연속 녹화 종료")
        print("  'q': 종료")
        print("\n카메라 대기 중...")

    def info_callback(self, msg):
        if not self.camera_matrix_saved:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)

            # cam_K.txt 저장
            cam_k_path = os.path.join(self.output_dir, "cam_K.txt")
            np.savetxt(cam_k_path, self.camera_matrix, fmt='%.6f')
            print(f"카메라 매트릭스 저장: {cam_k_path}")
            self.camera_matrix_saved = True

    def rgb_callback(self, msg):
        try:
            self.current_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB 변환 오류: {e}")

    def depth_callback(self, msg):
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth 변환 오류: {e}")

    def save_frame(self):
        if self.current_rgb is None or self.current_depth is None:
            return False

        filename = f"{self.frame_count:06d}.png"
        cv2.imwrite(os.path.join(self.rgb_dir, filename), self.current_rgb)
        cv2.imwrite(os.path.join(self.depth_dir, filename), self.current_depth)

        self.frame_count += 1
        print(f"[{self.frame_count}/{self.max_frames}] 저장 완료")

        if self.frame_count >= self.max_frames:
            print(f"\n{self.max_frames}프레임 캡처 완료!")
            self.is_recording = False
            return True

        return False

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            if self.current_rgb is not None:
                # 화면 표시
                display = self.current_rgb.copy()

                # 상태 표시
                if self.is_recording:
                    status_text = f"RECORDING [{self.frame_count}/{self.max_frames}]"
                    status_color = (0, 0, 255)  # 빨간색
                else:
                    status_text = f"Ready [{self.frame_count}/{self.max_frames}]"
                    status_color = (0, 255, 0)  # 초록색

                cv2.putText(display, status_text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(display, "'s'/Space:Capture  'b':Record  'e':Stop  'q':Quit", (10, display.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

                cv2.imshow("Frame Capture", display)

                # 녹화 중이면 프레임 저장
                if self.is_recording:
                    self.save_frame()

            key = cv2.waitKey(1) & 0xFF

            # 's' 또는 스페이스: 한 장 캡처
            if key == ord('s') or key == 32:  # 32 = 스페이스바
                if not self.is_recording:
                    self.save_frame()
            elif key == ord('b'):
                if not self.is_recording:
                    print("연속 녹화 시작")
                    self.is_recording = True
            elif key == ord('e'):
                if self.is_recording:
                    print("연속 녹화 종료")
                    self.is_recording = False
            elif key == ord('q'):
                print("종료")
                break

            time.sleep(1.0 / 50.0)

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='RealSense RGB+Depth 프레임 캡처')
    parser.add_argument('--output', '-o', type=str, default='vcb/ref_views/test_scene', help='출력 디렉토리')
    parser.add_argument('--frames', '-f', type=int, default=50, help='캡처할 프레임 수')
    parser.add_argument('--rgb_topic', type=str,
        default='/camera/color/image_raw',
        help='RGB image topic')
    parser.add_argument('--depth_topic', type=str,
        default='/camera/aligned_depth_to_color/image_raw',
        help='Aligned depth image topic')
    parser.add_argument('--camera_info_topic', type=str,
        default='/camera/color/camera_info',
        help='Camera info topic')
    args = parser.parse_args()

    rclpy.init()
    capture = None
    try:
        capture = FrameCapture(
            output_dir=args.output,
            max_frames=args.frames,
            rgb_topic=args.rgb_topic,
            depth_topic=args.depth_topic,
            camera_info_topic=args.camera_info_topic,
        )
        capture.run()
    except KeyboardInterrupt:
        pass
    finally:
        if capture is not None:
            capture.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()