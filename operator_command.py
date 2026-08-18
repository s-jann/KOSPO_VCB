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

VALID_STATES = {
    "OPEN",
    "CLOSE",
}


def parse_operator_command(command_text):
    """
    작업자 명령 파싱.

    입력 예:
        4SW02-01B open
        4SW02-01B close

    return:
        target_label, desired_state
    """

    parts = command_text.strip().split()

    if len(parts) != 2:
        raise ValueError(
            "명령 형식: <target_label> <open|close>\n"
            "예: 4SW02-01B open"
        )

    target_label = parts[0].strip().upper()
    desired_state = parts[1].strip().upper()

    if not target_label:
        raise ValueError(
            "target label이 비어 있습니다."
        )

    if desired_state not in VALID_STATES:
        raise ValueError(
            "상태는 OPEN 또는 CLOSE만 사용할 수 있습니다."
        )

    return target_label, desired_state


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
        desired_state,
    ):
        self.command_id += 1

        payload = {
            "active": True,
            "command_id": self.command_id,
            "target_label": target_label,
            "desired_state": desired_state,
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
            f"target={target_label}, "
            f"desired={desired_state}"
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
            "desired_state": "",
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
    print("  <VCB_LABEL> <OPEN|CLOSE>")
    print()
    print("예:")
    print("  4SW02-01B open")
    print("  4SW02-01B close")
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
                (
                    target_label,
                    desired_state,
                ) = parse_operator_command(
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
            print(
                f"  desired_state : {desired_state}"
            )

            node.publish_command(
                target_label,
                desired_state,
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