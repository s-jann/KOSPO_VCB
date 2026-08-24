#!/usr/bin/env python3
"""
VCB 토픽 모니터 (headless 터미널 출력용)

run_parallel_test.sh / run_vcb_command.sh가 공용으로 사용한다.

  - /vcb/perception        : 주기적으로 한 줄 요약 출력 (기본 2초 간격)
  - /foundation_pose/result: 수신 즉시 상세 블록 출력
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# perception은 매 프레임 발행되므로 터미널이 넘치지 않게 주기 제한
PERCEPTION_PRINT_PERIOD_SEC = 2.0


class VcbTopicMonitor(Node):

    def __init__(self):
        super().__init__("vcb_topic_monitor")

        self._last_perception_print = 0.0
        self.attempts = {}

        self.create_subscription(
            String, "/vcb/perception", self.perception_cb, 10,
        )
        self.create_subscription(
            String, "/foundation_pose/result", self.result_cb, 10,
        )

    # ----- /vcb/perception : 한 줄 요약 -----

    def perception_cb(self, msg):
        now = time.time()
        if now - self._last_perception_print < PERCEPTION_PRINT_PERIOD_SEC:
            return
        self._last_perception_print = now

        try:
            d = json.loads(msg.data)
        except Exception:
            return

        parts = []
        for inst in d.get("instances") or []:
            labels = ",".join(inst.get("labels") or []) or "-"
            parts.append(f"I{inst.get('index')}:{labels}")

        print(
            f"[PERCEPTION] vcb={d.get('num_vcb_instances')} "
            f"{' | '.join(parts) if parts else '(no vcb)'} "
            f"state={d.get('current_state')} "
            f"decision={d.get('decision')} "
            f"fps={d.get('fps')}",
            flush=True,
        )

    # ----- /foundation_pose/result : 상세 출력 -----

    def result_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception as exc:
            print(f"\n[FP RESULT] Invalid JSON: {exc}", flush=True)
            return

        cid = d.get("command_id")
        self.attempts[cid] = self.attempts.get(cid, 0) + 1

        print()
        print("=" * 60)
        print(" FOUNDATIONPOSE RESULT")
        print("=" * 60)

        # 명령 기반 실행일 때만 command 정보가 채워진다
        if cid is not None:
            print(f" command_id    : {cid}")
            print(f" target        : {d.get('target_label')}")
            print(f" desired_state : {d.get('desired_state')}")
            print(f" attempt       : {self.attempts[cid]}/3")

        print(f" object_found  : {d.get('object_found')}")

        if not d.get("object_found"):
            print("\n RESULT = OBJECT NOT FOUND")
            print("=" * 60)
            return

        p = d.get("pose_6d") or {}
        t = p.get("translation") or {}
        e = p.get("rotation_euler_deg") or {}

        print(
            f"\n Translation [m] : "
            f"x={t.get('x')}  y={t.get('y')}  z={t.get('z')}"
        )
        print(
            f" Euler [deg]     : "
            f"roll={e.get('roll')}  pitch={e.get('pitch')}  yaw={e.get('yaw')}"
        )
        print(f" confidence      : {d.get('confidence')}")
        print("\n RESULT = SUCCESS")
        print("=" * 60)


def main():
    rclpy.init()
    node = VcbTopicMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
