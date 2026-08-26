#!/usr/bin/env python3
"""
VCB 토픽 모니터 (headless 터미널 출력용)

run_parallel_test.sh / run_vcb_command.sh가 공용으로 사용한다.

  - /vcb/perception        : 주기적으로 한 줄 요약 출력 (기본 2초 간격)
  - /foundation_pose/result: 수신 즉시 상세 블록 출력
  - /vcb/status_notice     : 수신 즉시 상세 블록 출력 (CLOSE 알림)
"""

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# perception은 매 프레임 발행되므로 터미널이 넘치지 않게 주기 제한
PERCEPTION_PRINT_PERIOD_SEC = 2.0


class VcbTopicMonitor(Node):

    def __init__(self, perception_log_path=None):
        super().__init__("vcb_topic_monitor")

        self._last_perception_print = 0.0
        self.attempts = {}

        # run_vcb_command.sh처럼 같은 터미널에서 operator_command.py가
        # 입력을 받는 경우, 2초마다 찍히는 [PERCEPTION] 요약이 입력 프롬프트를
        # 방해한다. --perception-log가 주어지면 [PERCEPTION]은 파일로만 쓰고,
        # 드물게 오는 FOUNDATIONPOSE RESULT는 항상 표준출력에 남긴다.
        self._perception_log_fp = None
        if perception_log_path:
            self._perception_log_fp = open(
                perception_log_path, "a", buffering=1,
            )
            print(
                f"[INFO] Perception summary redirected to: "
                f"{perception_log_path}"
            )

        self.create_subscription(
            String, "/vcb/perception", self.perception_cb, 10,
        )
        self.create_subscription(
            String, "/foundation_pose/result", self.result_cb, 10,
        )
        self.create_subscription(
            String, "/vcb/status_notice", self.status_notice_cb, 10,
        )

    def close_log(self):
        if self._perception_log_fp is not None:
            self._perception_log_fp.close()
            self._perception_log_fp = None

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

        # diagnostic HSV는 인스턴스별 결과이므로, 어느 I{idx}에 붙는
        # 정보인지 바로 보이도록 라벨 옆에 붙여서 출력한다.
        diag_by_idx = {}
        for item in d.get("diagnostic_status") or []:
            diag_by_idx.setdefault(
                item.get("vcb_instance_idx"), []
            ).append(item)

        parts = []
        for inst in d.get("instances") or []:
            idx = inst.get("index")
            labels = ",".join(inst.get("labels") or []) or "-"
            seg = f"I{idx}:{labels}"

            diag_items = diag_by_idx.get(idx)
            if diag_items:
                diag_str = ",".join(
                    f"{item.get('hsv_label')}({item.get('current_state')})"
                    for item in diag_items
                )
                seg += f"[hsv={diag_str}]"

            parts.append(seg)

        line = (
            f"[PERCEPTION] vcb={d.get('num_vcb_instances')} "
            f"{' | '.join(parts) if parts else '(no vcb)'} "
            f"state={d.get('current_state')} "
            f"decision={d.get('decision')} "
            f"fps={d.get('fps')}"
        )

        out = self._perception_log_fp or None
        print(line, file=out, flush=True)

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

    # ----- /vcb/status_notice : CLOSE 알림 상세 출력 -----

    def status_notice_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception as exc:
            print(f"\n[STATUS NOTICE] Invalid JSON: {exc}", flush=True)
            return

        print()
        print("=" * 60)
        print(" STATUS NOTICE")
        print("=" * 60)
        print(f" command_id : {d.get('command_id')}")
        print(f" target     : {d.get('target_label')}")
        print(f" status     : {d.get('status')}")
        print(f" message    : {d.get('message')}")
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perception-log",
        type=str,
        default=None,
        help=(
            "설정 시 [PERCEPTION] 요약을 표준출력 대신 이 파일에 기록한다. "
            "FOUNDATIONPOSE RESULT는 항상 표준출력에 출력된다."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = VcbTopicMonitor(perception_log_path=args.perception_log)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        node.close_log()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
