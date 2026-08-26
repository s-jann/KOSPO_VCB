#!/usr/bin/env python3

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from std_msgs.msg import String


COMMAND_TOPIC = "/vcb/operator_command"


def parse_operator_command(command_text):
    """
    작업자 명령 파싱.

    입력 예:
        4SW02-01B

    라벨만 입력받는다. 상태(OPEN/CLOSE) 판정은 시스템이 직접
    HSV로 확인하며, OPEN이어야 작업을 진행한다.

    return:
        target_label
    """

    parts = command_text.strip().split()

    if len(parts) != 1:
        raise ValueError(
            "명령 형식: <target_label>\n"
            "예: 4SW02-01B"
        )

    target_label = parts[0].strip().upper()

    if not target_label:
        raise ValueError(
            "target label이 비어 있습니다."
        )

    return target_label


class OperatorCommandPublisher(Node):

    def __init__(self):
        super().__init__(
            "vcb_operator_command"
        )


        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.publisher = self.create_publisher(
            String,
            COMMAND_TOPIC,
            qos_profile,
        )

        self.command_id = 0

        self.get_logger().info(
            f"Operator command publisher started: "
            f"{COMMAND_TOPIC}"
        )

    def publish_command(
        self,
        target_label,
    ):
        self.command_id += 1

        payload = {
            "active": True,
            "command_id": self.command_id,
            "target_label": target_label,
            "timestamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(
            payload,
            ensure_ascii=False,
        )

        self.publisher.publish(msg)

        self.get_logger().info(
            "Published command: "
            f"id={self.command_id}, "
            f"target={target_label}"
        )

    def clear_command(self):
        """
        현재 작업 명령 해제.
        """

        self.command_id += 1

        payload = {
            "active": False,
            "command_id": self.command_id,
            "target_label": "",
            "timestamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(
            payload,
            ensure_ascii=False,
        )

        self.publisher.publish(msg)

        self.get_logger().info(
            "Current operator command cleared."
        )


def print_help():
    print()
    print("========================================")
    print(" VCB Operator Command")
    print("========================================")
    print()
    print("명령 형식:")
    print("  <VCB_LABEL>")
    print()
    print("예:")
    print("  4SW02-01B")
    print()
    print("  -> 라벨 확인 후 OPEN이면 작업 진행, CLOSE면 작업을 진행하지")
    print("     않고 /vcb/status_notice로 CLOSE 상태를 알립니다.")
    print()
    print("기타 명령:")
    print("  clear  : 현재 작업 명령 해제")
    print("  help   : 도움말")
    print("  quit   : 프로그램 종료")
    print("========================================")
    print()


def main():
    rclpy.init()

    node = OperatorCommandPublisher()

    print_help()

    try:
        while rclpy.ok():

            try:
                command_text = input(
                    "\nVCB command > "
                ).strip()

            except EOFError:
                break

            if not command_text:
                continue

            command_lower = command_text.lower()

            # -------------------------------------------------
            # 종료
            # -------------------------------------------------
            if command_lower in (
                "quit",
                "exit",
                "q",
            ):
                print(
                    "[INFO] Operator command "
                    "publisher 종료"
                )
                break

            # -------------------------------------------------
            # 도움말
            # -------------------------------------------------
            if command_lower in (
                "help",
                "h",
                "?",
            ):
                print_help()
                continue

            # -------------------------------------------------
            # 현재 명령 해제
            # -------------------------------------------------
            if command_lower == "clear":
                node.clear_command()

                # ROS2 middleware가 publish를 처리할 기회 제공
                rclpy.spin_once(
                    node,
                    timeout_sec=0.05,
                )

                continue

            # -------------------------------------------------
            # 작업 명령
            # -------------------------------------------------
            try:
                target_label = parse_operator_command(
                    command_text
                )

            except ValueError as exc:
                print(
                    f"[ERROR] {exc}"
                )
                continue

            # 사용자에게 최종 해석 결과 표시
            print()
            print(
                "[COMMAND]"
            )
            print(
                f"  target_label  : {target_label}"
            )

            node.publish_command(
                target_label,
            )

            # publish 처리
            rclpy.spin_once(
                node,
                timeout_sec=0.05,
            )

    except KeyboardInterrupt:
        print(
            "\n[INFO] Ctrl+C received."
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()