# ROS2 Humble 개발 현황 및 향후 계획

## 1. Branch 목적

`ros2-humble` 브랜치는 기존 ROS1 기반 VCB 상태 인식 및 FoundationPose 시스템을
ROS2 Humble 기반으로 확장하고, 향후 Navigation / SLAM / Robot Manipulation과
연동하기 위한 개발 브랜치이다.

최종 목표 흐름:

작업자 명령
→ SLAM / Navigation
→ YOLO Detection
→ OCR / HSV
→ Target VCB 확인
→ FoundationPose Trigger
→ 6D Pose
→ Robot Manipulation
→ 상태 재확인


## 2. 현재 반영 완료

### ROS2 확장

기존 ROS1 코드를 유지하면서 ROS2 대응 코드 추가.

- camera/pose_estimator_ros2.py
- camera/pose_streamer_ros2.py
- camera/pose_debugger_ros2.py
- yolo/src/main_infer_ros2.py

주요 변경:

- rclpy 기반 ROS2 통신
- RealSense ROS2 RGB / Depth / CameraInfo 입력
- FoundationPose ROS2 결과 publish
- YOLO ROS2 image topic 입력 지원


### Camera Rotation 보정

실제 카메라 raw image가 반시계 방향 90도 회전되어 입력되는 문제 대응.

지원 옵션:

- none
- 90_cw

추가 파일:

- rotation_utils.py

RGB-D 사용 시 다음 데이터를 동일한 기준으로 회전 보정:

- RGB
- Depth
- Camera intrinsic K

회전 보정된 영상을 기준으로:

- YOLO Detection
- OCR
- HSV
- FoundationPose

수행 가능.


## 3. 향후 시스템 구조

작업자 명령
    ↓
Target Label / Action
    ↓
SLAM / Navigation
    ↓
목표 VCB 근처 이동
    ↓
YOLO
    ↓
VCB / Label / Status Detection
    ↓
OCR / HSV
    ↓
Target VCB Verification
    ↓
필요 시 위치 미세 보정
    ↓
FoundationPose Trigger
    ↓
6D Pose
    ↓
Camera → Robot Base TF
    ↓
Robot Manipulation
    ↓
YOLO / OCR / HSV 재검증


## 4. TODO - Task Command

작업자 명령을 가정한 테스트 입력 기능 추가.

초기 입력 예:

target_label = "VCB102"
requested_action = "OPEN"

목적:

- OCR 결과와 target label 비교
- 목표 차단기 선택 테스트
- 추후 Navigation 및 작업 관리 노드와 연결

예상 구성:

- vcb_task_manager_ros2.py

초기에는 CLI 또는 config 기반 입력으로 구현 후
향후 ROS2 message/service/action으로 확장.


## 5. TODO - YOLO / OCR / HSV

### 5.1 Detection 조건 분리

현재:

vcb + label + status가 모두 검출되어야 OCR / HSV 수행

변경:

- label 검출 → OCR 독립 수행
- status 검출 → HSV 독립 수행
- vcb 검출 → VCB bbox 정보 유지

부분 Detection도 사용한다.


### 5.2 모든 Detection 유지

현재:

클래스별 confidence가 가장 높은 bbox 1개 선택

get_best_box_by_class()

변경 예정:

get_boxes_by_class()

예:

{
    "vcb": [...],
    "label": [...],
    "status": [...]
}

한 화면에 여러 VCB가 존재할 수 있으므로
각 클래스의 모든 Detection을 유지한다.


### 5.3 Target Label 검색

Navigation으로 목표 VCB 근처 이동 후
화면에 보이는 모든 label에 OCR 수행.

예:

작업 명령:
target_label = VCB102

Detection:

label1 → OCR → VCB101
label2 → OCR → VCB102
label3 → OCR → VCB103

VCB102와 일치하는 label2를 Target으로 선택.

YOLO confidence는 label 여부 판단에 사용하고,
실제 Target 식별은 OCR 결과를 이용한다.


### 5.4 VCB / Label / Status Association

Target label을 기준으로 같은 차단기에 해당하는:

- VCB
- Status

Detection을 연결.

초기 방법:

- bbox 위치
- bbox 거리
- 상대적인 배치 관계

필요 시 차단기의 고정된 2단 / 일렬 구조 정보 활용.


### 5.5 부분 Detection 처리

label만 검출:
- OCR 수행
- target label 확인 가능
- 위치 보정에 활용 가능

status만 검출:
- HSV 수행
- 상태는 확인 가능
- 어느 VCB의 상태인지는 확정하지 않음

VCB + label:
- OCR 수행
- target 후보 확인

VCB + status:
- HSV 수행
- target 확정에는 사용하지 않음

label + status:
- OCR + HSV
- 동일 VCB association 성공 시 높은 신뢰도의 정보로 사용

VCB + label + status:
- 가장 이상적인 상태
- Target / State 확인 후 FoundationPose 단계로 진행


### 5.6 ROS2 Perception 결과 전달

YOLO / OCR / HSV 결과를 ROS2로 publish.

예상 Topic:

/vcb/perception_result

초기 구현:

std_msgs/msg/String + JSON

예:

{
    "target_label": "VCB102",
    "detected_label": "VCB102",
    "target_match": true,
    "state": "OPEN",
    "vcb_detected": true,
    "label_detected": true,
    "status_detected": true
}

향후 필요 시 custom ROS2 message로 변경.


### 5.7 저장 기능 정리

배포 시 지속적인 결과 저장 비활성화.

배포 기본값:

enable_gui = false
save_video = false
save_frames = false
save_selected_frames = false
save_csv = false

최근 N frame의 OCR / HSV / Detection 상태만 RAM에 유지하여
결과 안정화에 사용.


## 6. TODO - Navigation / SLAM Integration

차단기는 일렬 + 2단 구조이며 위치가 고정되어 있다고 가정.

SLAM / Navigation 역할:

- 작업자가 지정한 차단기 위치로 이동
- 목표 차단기 중앙 근처까지 접근

YOLO 역할:

- 실제 목표 차단기가 화면에 존재하는지 확인
- OCR을 이용한 Target Verification
- 필요할 경우 마지막 위치 미세 보정

차단기 위치 DB 예:

VCB101 → row 2 / column 1 / waypoint A
VCB102 → row 2 / column 2 / waypoint B
VCB103 → row 2 / column 3 / waypoint C

고정 slot 정보는 OCR을 대체하지 않고
후보 제한 및 오류 검증을 위한 보조 정보로 사용.


## 7. TODO - FoundationPose

### 현재

pose_streamer_ros2.py는 카메라 데이터가 들어오면
FoundationPose register를 지속적으로 수행.

개발 / 연속 테스트 용도로 유지.


### 변경 방향

실제 통합 시스템에서는 FoundationPose를 항상 실행하지 않는다.

YOLO / OCR / HSV
→ Target 확인
→ State 확인
→ FoundationPose Trigger
→ Pose 1회 추정


### 구현 기반

pose_estimator_ros2.py의 RealtimePoseEstimator를 공통 FoundationPose engine으로 사용.

새 배포 노드 예정:

camera/pose_trigger_ros2.py

동작:

Node 실행
→ Model / Mesh / CUDA 초기화
→ IDLE
→ Trigger 수신
→ 현재 RGB / Depth / K 이용
→ FoundationPose inference
→ Pose publish
→ IDLE


### 기존 코드 역할

pose_estimator_ros2.py
- FoundationPose 공통 기능
- Interactive 테스트

pose_streamer_ros2.py
- 연속 자동 Pose 테스트

pose_debugger_ros2.py
- FoundationPose 내부 Debug

pose_trigger_ros2.py
- 실제 시스템 통합 / 배포용


## 8. FoundationPose ROS2 Output

기존 Topic 유지:

/foundation_pose/pose
/foundation_pose/result

/foundation_pose/pose:
- geometry_msgs/PoseStamped
- Camera frame 기준 6D Pose

향후 로봇 조작 전:

Camera Frame
→ Robot Base Frame

TF 변환 필요.


## 9. TODO - Robot Manipulation

향후 구현:

- /foundation_pose/pose subscribe
- Camera → Robot Base TF
- Object Pose → Grasp / Manipulation Pose
- Robot Arm 동작
- 작업 수행 후 YOLO / OCR / HSV로 실제 상태 재확인


## 10. 구현 우선순위

Phase 1 - YOLO 구조 수정

- OCR / HSV 독립 실행
- 모든 Detection 유지
- 모든 Label OCR
- Target Label Matching


Phase 2 - Perception 결과 통합

- Target VCB Association
- 최근 N frame 안정화
- ROS2 perception_result Topic


Phase 3 - Navigation Integration

- Target VCB ↔ waypoint mapping
- Navigation 완료 후 Target Verification
- 필요 시 미세 Alignment


Phase 4 - FoundationPose Trigger

- pose_trigger_ros2.py 구현
- IDLE / Trigger 기반 inference
- Headless 배포


Phase 5 - Robot Integration

- TF 변환
- Robot Manipulation
- 작업 완료 상태 재확인