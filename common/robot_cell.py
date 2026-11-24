"""4회차용 로봇 셀: UR5e 협동로봇(진공 흡착 툴) + 비전 카메라 + 양품·재검 트레이 + 안전 인터록.

- 로봇은 NVIDIA 에셋 서버의 UR5e 모델(assets/sim/ur5e, `./scripts/factory assets` 로 받음)이고 물리 관절체(PhysX articulation)임.
  관절 각도와 관절 토크는 물리 엔진이 계산함.
- 진공 흡착 툴은 부품을 잡은 동안 손목 끝 링크의 질량을 부품만큼 늘리는 방식으로 표현함 (부품 자체는 흡착 패드를 따라 움직임).
- 좌표는 셀 기준(로봇 받침 중심, 작업대 윗면이 z=0)이고 단위는 m, 각도는 도. 흡착 패드는 항상 아래를 향함.
"""
import csv
import datetime as dt
import json
from collections import deque
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from .scene_kit import Kit

ROBOT_USD = Path(__file__).resolve().parents[1] / "assets" / "sim" / "ur5e" / "ur5e.usd"
TABLE_Z = 0.75                       # 작업대 윗면의 높이 (세계 좌표)
BASE_H = 0.02                        # 로봇 받침판 두께
TOOL_LEN = 0.14                      # 플랜지에서 흡착 패드 끝까지
TOOL_KG = 0.35                       # 흡착 툴 무게
WRIST_KG = 0.19                      # UR5e 손목 끝 링크 무게
RATED_KG = 5.0                       # UR5e 가반하중
JOINTS = ("J1 베이스", "J2 어깨", "J3 팔꿈치", "J4 손목1", "J5 손목2", "J6 손목3")
PART = (0.10, 0.07, 0.035)           # 부품 (제어부 모듈) 크기
PART_KG = 0.50                       # 표준 부품 무게
HEAVY_KG = (1.5, 2.5)                # 무거운 부품 두 단계
FEED_X0 = 1.30                       # 부품이 공급 컨베이어에 올라오는 위치
OVERLOAD_KG = 7.0                    # 과부하 시험용 (가반하중 초과)
VACUUM_KPA = -62.0                   # 흡착 중 진공 압력
PICK_X, TRAY_X, TRAY_Y = 0.60, 0.15, 0.52
TEACH_DEFAULT = {                    # 티칭 지점: 흡착 패드 끝 위치 (x, y, z)
    "home": (0.35, 0.00, 0.45),
    "pick_above": (PICK_X, 0.00, 0.22),
    "pick": (PICK_X, 0.00, PART[2] + 0.004),
    "ok_above": (TRAY_X, TRAY_Y, 0.24),
    "ok_place": (TRAY_X, TRAY_Y, PART[2] + 0.02),
    "ng_above": (TRAY_X, -TRAY_Y, 0.24),
    "ng_place": (TRAY_X, -TRAY_Y, PART[2] + 0.02),
}
TEACH_KO = dict(home="대기", pick_above="집기 위", pick="집기", ok_above="양품 위", ok_place="양품 놓기",
                ng_above="재검 위", ng_place="재검 놓기")
REACH_MAX, Z_MIN, X_MIN = 0.80, 0.015, -0.10      # 동작 범위 제한 (인터록)
OUT_OF_RANGE = (0.92, 0.25, 0.15)                 # 인터록 시험용 범위 밖 지령
LOG_COLS = (["time_s", "cycle", "phase"] + [f"q{i}_deg" for i in range(1, 7)] + [f"tau{i}_nm" for i in range(1, 7)]
            + ["speed_pct", "gripping", "vacuum_kpa"])
C_GREEN, C_RED, C_YELLOW, C_DARK = (0.1, 0.85, 0.2), (0.95, 0.1, 0.1), (1.0, 0.75, 0.05), (0.12, 0.12, 0.12)
CAMERAS = {
    "cell": ((2.95, -1.35, 2.35), (0.3, 0.0, 0.98)),
    "robot": ((2.0, -0.85, 1.75), (0.3, 0.0, 1.05)),
    "top": ((0.35, -0.02, 4.0), (0.35, 0.0, 0.75)),
    "trays": ((-1.5, -0.3, 1.95), (0.25, 0.0, 0.9)),
}


class Arm:
    """UR5e 운동학 (Isaac Sim 의 Lula 풀이기). 흡착 패드가 아래를 향하는 자세만 씀."""
    DOWN = np.array([0.0, 1.0, 0.0, 0.0])            # 툴 축이 아래를 향하는 자세 (w, x, y, z)

    def __init__(self):
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        d = get_extension_path_from_name("isaacsim.robot_motion.motion_generation") + "/motion_policy_configs/universal_robots/ur5e/"
        self.solver = LulaKinematicsSolver(robot_description_path=d + "rmpflow/ur5e_robot_description.yaml", urdf_path=d + "ur5e.urdf")
        self.solver.set_robot_base_pose(np.array([0.0, 0.0, BASE_H]), np.array([1.0, 0.0, 0.0, 0.0]))

    def ik(self, x, y, z):
        """흡착 패드 끝 위치 → 관절 각도 (도). 닿지 않으면 None. 항상 같은 팔 자세(팔꿈치 위)의 해를 고름."""
        r = float(np.hypot(x, y))
        if r < 0.16:
            return None
        q1 = float(np.arctan2(y, x) - np.arcsin(min(0.1333 / r, 1.0)))
        seed = np.array([q1, np.radians(-75), np.radians(105), np.radians(-120), -np.pi / 2, q1 - np.pi / 2])
        q, ok = self.solver.compute_inverse_kinematics("tool0", np.array([x, y, z + TOOL_LEN]), self.DOWN, warm_start=seed)
        if not ok:
            return None
        q = seed + (np.asarray(q) - seed + np.pi) % (2 * np.pi) - np.pi          # 씨앗에서 가장 가까운 각도로
        if q[2] <= 0 or not (-np.pi < q[1] < 0) or abs(q[4] + np.pi / 2) > 0.2:   # 다른 팔 자세의 해는 쓰지 않음
            return None
        return np.degrees(q)

    def fk(self, q):
        """관절 각도 (도) → 흡착 패드 끝 위치."""
        p, R = self.solver.compute_forward_kinematics("tool0", np.radians(q))
        tip = np.asarray(p) + np.asarray(R)[:, 2] * TOOL_LEN
        return float(tip[0]), float(tip[1]), float(tip[2])


class RobotCell:
    def __init__(self, stage, log_dir, seed=4):
        self.stage, self.k = stage, Kit(stage)
        self.rng = np.random.default_rng(seed)
        self.log_dir = Path(log_dir)
        self.labels, self.labels_dirty = {}, True
        self.teach = {k: tuple(v) for k, v in TEACH_DEFAULT.items()}
        self.robot = None                      # 물리 관절체 손잡이 (attach 에서 연결)
        self.arm = Arm()
        self.ik, self.fk = self.arm.ik, self.arm.fk
        self.q_cmd = np.array(self.ik(*self.teach["home"]))
        self.q = self.q_cmd.copy()
        self.tau = np.zeros(6)
        self.vacuum = 0.0
        self.speed_pct, self.speed_limit = 60.0, 100.0
        self.torque_limit = 80.0               # J2 토크 임계 (N·m)
        self.level = 3                         # 제어 수준 1 · 2 · 3
        self.auto = False                      # 자동으로 다음 부품을 처리
        self.estop, self.trip = False, ""      # 비상 정지, 인터록 정지 사유
        self.checks = dict(range=False, speed=False, torque=False, estop=False)   # 인터록 점검표
        self.msg = ""
        self.t, self.cycle, self.phase = 0.0, 0, "대기"
        self.moves, self.move = deque(), None  # 남은 동작 목록, 진행 중인 동작
        self.part, self.gripping = None, False
        self.payload_kg, self.next_payload = PART_KG, None
        self.defect_rate = 0.3
        self.torque_ng_rate = 0.06             # 체결 이력에서 토크 부족으로 걸리는 비율
        self.force_defect = None               # 다음 부품의 상태를 지정 (교재 제작용)
        self.pending = None                    # 수준 2: 사람의 결정을 기다리는 부품
        self.judge_fn = None                   # 비전 판정 함수 (실행 파일이 연결): 부품 → 양품이면 True
        self.request_fn = None                 # 비전 영상 요청 함수
        self.vision_wait = 0
        self.counts = dict(ok=0, recheck=0, wrong=0)
        self.cycles = deque(maxlen=40)         # 사이클별 특징과 이상 점수
        self.cyc_tau = []                      # 이번 사이클의 운반 중 J2 · J3 토크
        self.base_feat = None                  # 정상 기준 (평균, 표준편차)
        self.score_th = 4.0
        self.logging, self._log, self._log_n = False, None, 0
        self.person_in, self.person_x = False, 3.2
        self.n_ok_tray = self.n_ng_tray = 0
        self._build()

    # ======================================================================
    # 장면
    # ======================================================================
    def label(self, key, text, pos, size=16, color=(1, 1, 1), detail=False):
        self.labels[key] = dict(text=text, pos=pos, size=size, color=color, detail=detail)
        self.labels_dirty = True

    def set_label(self, key, text):
        if self.labels[key]["text"] != text:
            self.labels[key]["text"] = text
            self.labels_dirty = True

    def W(self, x, y, z):
        """셀 좌표 → 세계 좌표"""
        return (x, y, z + TABLE_Z)

    def _build(self):
        k = self.k
        self.indoor = k.hall(pos=(-2.0, -4.0, 0.0))             # 공장 건물 (에셋이 있을 때)
        if self.indoor:
            k.lights(dome=120.0, sun=0.0)
            for j, (x, y) in enumerate(((0.4, 0.0), (0.4, -3.0), (-2.6, 0.0), (3.4, 0.0), (0.4, 3.0))):
                k.area_light(f"/World/Lights/Ceil{j}", (x, y, 5.6), (1.6, 1.6), 9000.0 if j == 0 else 5000.0)
        else:
            k.lights()
            k.box("/World/Floor", (14, 12, 0.1), (0.5, 0, -0.05), mat=k.mat((0.16, 0.17, 0.19), rough=0.8))
        steel = k.mat((0.62, 0.64, 0.67), rough=0.3, metal=0.9)
        alu = k.mat((0.78, 0.8, 0.82), rough=0.35, metal=0.7)
        dark = k.mat((0.07, 0.075, 0.08), rough=0.55)
        rubber = k.mat((0.03, 0.03, 0.03), rough=0.9)
        frame = k.mat((0.95, 0.72, 0.05), rough=0.45)
        top = k.mat((0.86, 0.87, 0.88), rough=0.42)
        cx = 0.42                                                # 작업대 중심
        # 작업대: 알루미늄 프로파일 뼈대 + 상판 + 아래 선반 + 제어함
        k.box("/World/Cell/Table", (1.9, 1.9, 0.04), (cx, 0, TABLE_Z - 0.02), mat=top)
        k.box("/World/Cell/TableEdge", (1.94, 1.94, 0.02), (cx, 0, TABLE_Z - 0.05), mat=alu)
        for j, (dx, dy) in enumerate(((-0.9, -0.9), (0.9, -0.9), (-0.9, 0.9), (0.9, 0.9))):
            k.box(f"/World/Cell/Leg{j}", (0.08, 0.08, TABLE_Z - 0.06), (cx + dx, dy, (TABLE_Z - 0.06) / 2), mat=alu)
            k.cyl(f"/World/Cell/Foot{j}", 0.05, 0.02, (cx + dx, dy, 0.01), mat=rubber)
        for j, (sx, sy, ox, oy) in enumerate(((1.8, 0.05, 0, -0.9), (1.8, 0.05, 0, 0.9), (0.05, 1.8, -0.9, 0), (0.05, 1.8, 0.9, 0))):
            k.box(f"/World/Cell/Brace{j}", (sx, sy, 0.05), (cx + ox, oy, 0.2), mat=alu)
        k.box("/World/Cell/Shelf", (1.8, 1.8, 0.015), (cx, 0, 0.22), mat=k.mat((0.3, 0.32, 0.34), rough=0.6))
        k.box("/World/Cell/Cabinet", (0.5, 0.42, 0.42), (cx - 0.55, 0.55, 0.44), mat=k.mat((0.83, 0.84, 0.82), rough=0.4))
        k.box("/World/Cell/CabinetDoor", (0.44, 0.01, 0.36), (cx - 0.55, 0.335, 0.44), mat=k.mat((0.72, 0.73, 0.72), rough=0.4))
        k.box("/World/Cell/VacuumPump", (0.3, 0.22, 0.2), (cx + 0.35, 0.55, 0.33), mat=k.mat((0.1, 0.3, 0.7), rough=0.4))
        k.cyl("/World/Cell/VacuumTank", 0.07, 0.3, (cx + 0.35, 0.3, 0.3), axis="X", mat=steel)
        # 부품 공급 컨베이어 (오른쪽에서 집기 위치까지)
        fx0, fx1 = PICK_X - 0.12, FEED_X0 + 0.1
        fc, fl = (fx0 + fx1) / 2, fx1 - fx0
        k.box("/World/Cell/Feeder", (fl, 0.16, 0.012), self.W(fc, 0, 0.006), mat=rubber)
        for j, dy in enumerate((-0.09, 0.09)):
            k.box(f"/World/Cell/FeederRail{j}", (fl, 0.014, 0.034), self.W(fc, dy, 0.017), mat=alu)
        for j in range(9):
            k.cyl(f"/World/Cell/FeederRoll{j}", 0.006, 0.19, self.W(fx0 + 0.05 + j * (fl - 0.1) / 8, 0, 0.004), axis="Y", mat=steel)
        k.box("/World/Cell/FeederMotor", (0.1, 0.09, 0.09), self.W(fx1 - 0.06, -0.15, 0.045), mat=k.mat((0.15, 0.35, 0.2), rough=0.5))
        # 트레이: 양품 / 재검
        for name, y, color in (("OK", TRAY_Y, (0.08, 0.38, 0.75)), ("NG", -TRAY_Y, (0.85, 0.28, 0.08))):
            m = k.mat(color, rough=0.45)
            k.box(f"/World/Cell/Tray{name}", (0.36, 0.28, 0.012), self.W(TRAY_X, y, 0.006), mat=m)
            for j, (sx, sy, ox, oy) in enumerate(((0.36, 0.012, 0, -0.14), (0.36, 0.012, 0, 0.14), (0.012, 0.28, -0.18, 0), (0.012, 0.28, 0.18, 0))):
                k.box(f"/World/Cell/Tray{name}W{j}", (sx, sy, 0.035), self.W(TRAY_X + ox, y + oy, 0.0175), mat=m)
        self.label("tray_ok", "양품 트레이", self.W(TRAY_X, TRAY_Y, 0.22), 15, (0.5, 0.8, 1.0))
        self.label("tray_ng", "재검 트레이", self.W(TRAY_X, -TRAY_Y, 0.22), 15, (1.0, 0.6, 0.45))
        # 비전 카메라 (집기 위치 위) + 링 조명
        px, cxp = PICK_X, 0.98
        k.box("/World/Cell/CamPost", (0.045, 0.045, 0.84), self.W(cxp, 0.36, 0.42), mat=alu)
        k.box("/World/Cell/CamArm0", (0.045, 0.40, 0.045), self.W(cxp, 0.18, 0.82), mat=alu)
        k.box("/World/Cell/CamArm1", (cxp - px + 0.045, 0.045, 0.045), self.W((cxp + px) / 2, 0.0, 0.82), mat=alu)
        k.box("/World/Cell/CamBody", (0.06, 0.06, 0.09), self.W(px, 0.0, 0.765), mat=dark)
        k.cyl("/World/Cell/CamLens", 0.022, 0.04, self.W(px, 0.0, 0.705), mat=steel)
        k.cyl("/World/Cell/CamRing", 0.06, 0.012, self.W(px, 0.0, 0.69), mat=k.mat((0.95, 0.95, 0.9), rough=0.3, emissive=(2.5, 2.5, 2.3)))
        cam = UsdGeom.Camera.Define(self.stage, "/World/Cell/Camera")
        cam.CreateFocalLengthAttr(50.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
        UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(*self.W(px, 0.0, 0.68)))
        self.label("cam", "비전 카메라", self.W(px, 0.0, 1.02), 15, (1, 1, 1))
        self.label("cam_res", "", self.W(px, 0.0, 0.94), 14, (0.7, 1.0, 0.7), detail=True)
        # 방호 울타리(투명 판)와 라이트 커튼, 경광등, 비상 정지 버튼
        x0, x1, yf = cx - 1.1, cx + 1.1, 1.12
        glass = k.mat((0.75, 0.9, 1.0), 0.08, opacity=0.16, name="Glass")
        for j, (x, y) in enumerate(((x0, -yf), (x1, -yf), (x0, yf), (x1, yf))):
            k.box(f"/World/Cell/FencePost{j}", (0.05, 0.05, 1.9), (x, y, 0.95), mat=frame)
        k.box("/World/Cell/FenceBack", (x1 - x0, 0.012, 1.2), (cx, yf, 1.25), mat=glass)
        k.box("/World/Cell/FenceLeft", (0.012, 2 * yf, 1.2), (x0, 0, 1.25), mat=glass)
        for j, (sx, sy, px_, py_) in enumerate(((x1 - x0, 0.05, cx, yf), (0.05, 2 * yf, x0, 0))):
            k.box(f"/World/Cell/FenceTop{j}", (sx, sy, 0.05), (px_, py_, 1.875), mat=frame)
        self.curtain = [k.box(f"/World/Cell/Curtain{j}", (x1 - x0, 0.008, 0.008), (cx, -yf, 0.35 + j * 0.22), color=C_GREEN) for j in range(7)]
        for j, x in enumerate((x0, x1)):
            k.box(f"/World/Cell/CurtainUnit{j}", (0.06, 0.04, 1.5), (x, -yf - 0.045, 1.0), mat=dark)
        k.cyl("/World/Cell/LampPole", 0.015, 0.4, (x1, yf, 2.1), mat=steel)
        self.lamp = k.cyl("/World/Cell/Lamp", 0.045, 0.14, (x1, yf, 2.37), color=C_GREEN)
        k.box("/World/Cell/EstopBox", (0.12, 0.07, 0.16), (x1, -yf - 0.09, 1.1), mat=frame)
        self.estop_pos = (x1, -yf - 0.14, 1.12)
        self.estop_btn = k.cyl("/World/Cell/EstopBtn", 0.04, 0.03, self.estop_pos, axis="Y", color=C_RED)
        self.label("estop", "비상 정지", (x1, -yf - 0.09, 1.3), 13, (1.0, 0.6, 0.5), detail=True)
        # 작업자 (방호 영역 침입 시연용)
        self.person_y0 = -yf - 0.55
        self.person = k.group("/World/Person", (self.person_x, self.person_y0, 0))
        blue, skin, vest = (0.12, 0.2, 0.42), (0.86, 0.68, 0.55), (1.0, 0.55, 0.05)
        for j, dx in enumerate((-0.09, 0.09)):
            k.cyl(f"/World/Person/Leg{j}", 0.075, 0.82, (dx, 0, 0.45), color=blue)
            k.box(f"/World/Person/Shoe{j}", (0.11, 0.26, 0.08), (dx, 0.04, 0.04), color=(0.08, 0.08, 0.08))
            k.cyl(f"/World/Person/Arm{j}", 0.05, 0.6, (dx * 2.5, 0, 1.12), color=(0.25, 0.32, 0.5))
            k.sphere(f"/World/Person/Hand{j}", 0.055, (dx * 2.5, 0, 0.8), color=skin)
        k.cyl("/World/Person/Torso", 0.17, 0.6, (0, 0, 1.15), color=(0.25, 0.32, 0.5))
        k.cyl("/World/Person/Vest", 0.178, 0.36, (0, 0, 1.2), color=vest)
        k.cyl("/World/Person/VestBand", 0.181, 0.05, (0, 0, 1.2), color=(0.9, 0.9, 0.85))
        k.cyl("/World/Person/Neck", 0.055, 0.08, (0, 0, 1.49), color=skin)
        k.sphere("/World/Person/Head", 0.115, (0, 0, 1.62), color=skin)
        k.sphere("/World/Person/Helmet", 0.125, (0, 0, 1.67), color=(1.0, 0.85, 0.1), scale=(1, 1, 0.75))
        k.cyl("/World/Person/HelmetBrim", 0.14, 0.015, (0, 0.03, 1.655), color=(1.0, 0.85, 0.1))
        # 동작 범위 (인터록) 표시: 작업대 위의 원
        ring = [(REACH_MAX * np.cos(a), REACH_MAX * np.sin(a), TABLE_Z + 0.003) for a in np.linspace(-1.9, 1.9, 70)]
        k.line("/World/Cell/ReachRing", ring, 0.008, (0.95, 0.75, 0.1))
        self.label("reach", "동작 범위 한계", self.W(0.2, -0.8, 0.05), 13, (1.0, 0.85, 0.3), detail=True)
        self._build_robot(steel, dark)
        self._build_parts()
        self.label("robot", "협동로봇 UR5e · 진공 흡착", self.W(0, 0, 1.22), 20, (1, 1, 1))
        self.label("status", "", self.W(0, 0, 1.12), 15, (0.6, 0.95, 1.0), detail=True)

    def _build_robot(self, steel, dark):
        k, p = self.k, "/World/Cell/Robot"
        if not ROBOT_USD.exists():
            raise FileNotFoundError("UR5e 에셋이 없습니다. 먼저 실행: ./scripts/factory assets   (인터넷 필요, 처음 한 번)")
        k.cyl("/World/Cell/RobotPlate", 0.11, BASE_H, self.W(0, 0, BASE_H / 2), mat=dark)
        prim = UsdGeom.Xform.Define(self.stage, p).GetPrim()
        prim.GetReferences().AddReference(str(ROBOT_USD))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*self.W(0, 0, BASE_H)))
        # 진공 흡착 툴: 손목 끝 링크에 붙인 모양 (물리 질량은 링크 질량으로 반영)
        tp = p + "/wrist_3_link/SuctionTool"
        k.group(tp)
        k.cyl(tp + "/Adapter", 0.034, 0.018, (0, 0, 0.009), mat=dark)
        k.cyl(tp + "/Body", 0.022, 0.07, (0, 0, 0.053), mat=k.mat((0.75, 0.77, 0.8), rough=0.3, metal=0.9))
        k.box(tp + "/Valve", (0.03, 0.05, 0.03), (0.0, 0.03, 0.045), mat=k.mat((0.1, 0.3, 0.75), rough=0.4))
        k.cyl(tp + "/Hose", 0.006, 0.07, (0.0, 0.075, 0.045), axis="Y", mat=k.mat((0.05, 0.4, 0.9), rough=0.5))
        for j, (r, z) in enumerate(((0.019, 0.094), (0.024, 0.104), (0.019, 0.114), (0.026, 0.124))):      # 주름 패드
            k.cyl(tp + f"/Bellows{j}", r, 0.010, (0, 0, z), mat=dark)
        self.cup = k.cyl(tp + "/Cup", 0.03, 0.012, (0, 0, TOOL_LEN - 0.006), color=(0.15, 0.15, 0.16))
        # 손목 관절의 구동 감쇠를 올림: 에셋 기본값은 빈 손목 기준이라, 흡착 툴과 부품을 달면 빠른 동작에서 손목이 흔들림
        for name, gain in (("wrist_1_joint", 12.0), ("wrist_2_joint", 12.0), ("wrist_3_joint", 12.0), ("shoulder_pan_joint", 4.0),
                           ("shoulder_lift_joint", 4.0), ("elbow_joint", 2.0)):
            d = UsdPhysics.DriveAPI.Get(self.stage.GetPrimAtPath(f"{p}/joints/{name}"), "angular")
            d.GetDampingAttr().Set(float(d.GetDampingAttr().Get()) * gain if float(d.GetDampingAttr().Get()) > 0.5 else 20.0)
        self.wrist_mass = UsdPhysics.MassAPI.Apply(self.stage.GetPrimAtPath(p + "/wrist_3_link"))
        self._set_payload(0.0)

    def _set_payload(self, kg):
        self.wrist_mass.GetMassAttr().Set(float(WRIST_KG + TOOL_KG + kg))

    def _build_parts(self):
        k = self.k
        self.parts = []
        for i in range(14):
            p = f"/World/Cell/Parts/P{i:02d}"
            g = k.group(p)
            k.box(p + "/Case", PART, (0, 0, PART[2] / 2), color=(0.75, 0.76, 0.78))
            k.box(p + "/Board", (PART[0] * 0.86, PART[1] * 0.8, 0.003), (0, 0, PART[2] + 0.0015), color=(0.1, 0.42, 0.2))
            conn = k.box(p + "/Connector", (0.026, 0.034, 0.012), (0.024, 0, PART[2] + 0.009), color=(1.0, 0.45, 0.05))
            for j, (sx, sy) in enumerate(((-0.036, -0.022), (-0.036, 0.022), (0.036, -0.022), (0.036, 0.022))):
                k.cyl(p + f"/Screw{j}", 0.004, 0.003, (sx, sy, PART[2] + 0.0045), color=(0.15, 0.15, 0.16))
            g.show(False)
            self.parts.append(dict(g=g, conn=conn, state="free"))

    # ======================================================================
    # 물리 연결과 기록
    # ======================================================================
    def attach(self, articulation):
        """물리 관절체 손잡이를 연결하고 대기 자세로 놓음."""
        from isaacsim.core.utils.types import ArticulationAction
        self._action = ArticulationAction
        self.robot = articulation
        self.robot.set_joint_positions(np.radians(self.q_cmd))
        self._set_targets()

    def _set_targets(self):
        self.robot.apply_action(self._action(joint_positions=np.radians(self.q_cmd)))

    def start_log(self):
        self.stop_log()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"robot_log_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        f = open(self.log_path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(LOG_COLS)
        self._log, self._log_n, self.logging = (f, w), 0, True

    def stop_log(self):
        if self._log:
            self._log[0].close()
        self._log, self.logging = None, False

    def save_teach(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: [round(float(c), 4) for c in v] for k, v in self.teach.items()}, indent=1))

    def load_teach(self, path):
        self.teach.update({k: tuple(v) for k, v in json.loads(path.read_text()).items()})

    # ======================================================================
    # 동작 지령과 인터록
    # ======================================================================
    def in_range(self, x, y, z):
        return np.hypot(x, y) <= REACH_MAX and z >= Z_MIN and x >= X_MIN and self.ik(x, y, z) is not None

    def goto(self, target, phase, on_done=None, speed=None):
        """손끝을 target 으로 보내는 지령을 추가. 동작 범위를 벗어나면 거부함."""
        pos = self.teach[target] if isinstance(target, str) else tuple(target)
        if not self.in_range(*pos):
            self.checks["range"] = True
            self.msg = f"인터록: 동작 범위 밖 지령 거부  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            return False
        self.moves.append(dict(q=np.array(self.ik(*pos)), p=np.array(pos, dtype=float), phase=phase, on_done=on_done, speed=speed))
        return True

    def _begin_move(self):
        m = self.moves.popleft()
        pct = self.speed_pct if m["speed"] is None else m["speed"]
        if pct > self.speed_limit:                       # 속도 상한 인터록
            pct = self.speed_limit
            self.checks["speed"] = True
            self.msg = f"인터록: 속도 상한 {self.speed_limit:.0f}% 로 제한"
        m["q0"] = self.q_cmd.copy()
        m["p0"] = np.array(self.fk(m["q0"]))            # 흡착 패드가 직선으로 움직이도록 시작 위치를 기억
        dist = float(np.linalg.norm(m["p"] - m["p0"]))
        # 100% = 관절 초당 90도 · 흡착 패드 초당 0.5 m 중 느린 쪽에 맞춤
        m["T"] = max(float(np.abs(m["q"] - m["q0"]).max()) / (90.0 * pct / 100.0), dist / (0.5 * pct / 100.0), 0.15)
        m["t"] = 0.0
        self.move, self.phase = m, m["phase"]

    def stop_all(self, reason):
        self.moves.clear()
        self.move = None
        self.trip = reason
        self.auto = False

    def press_estop(self, reason="비상 정지 버튼"):
        self.estop = True
        self.checks["estop"] = True
        self.stop_all(reason)
        self.msg = f"비상 정지: {reason}"

    def reset(self):
        """정지 사유를 확인한 뒤 복귀. 잡고 있던 부품은 사람이 꺼낸 것으로 처리함."""
        if self.person_in:
            self.msg = "작업자가 방호 영역 안에 있음 · 복귀 불가"
            return
        self.estop, self.trip, self.msg = False, "", "복귀 완료"
        self.grace = 45                                   # 복귀 직후 토크 값이 가라앉을 때까지 임계 검사를 쉼
        if self.part is not None:                         # 처리 중이던 부품은 사람이 꺼냄
            self.part["state"] = "free"
            self.part["g"].show(False)
            self.part, self.gripping, self.pending = None, False, None
            self._set_payload(0.0)
            self.msg = "복귀 완료 · 처리 중이던 부품은 사람이 꺼냄"
        self.goto("home", "대기로 복귀", on_done=lambda: setattr(self, "phase", "대기"))

    # ======================================================================
    # 부품과 사이클
    # ======================================================================
    def feed_part(self, defect=None, payload=None):
        """새 부품을 공급 컨베이어에 올림. defect: none / connector(커넥터 미삽입)."""
        part = next((p for p in self.parts if p["state"] == "free"), None)
        if part is None or self.part is not None:
            return None
        if defect is None and self.force_defect is not None:
            defect, self.force_defect = self.force_defect, None
        if defect is None:
            defect = "connector" if self.rng.random() < self.defect_rate else "none"
        kg = self.next_payload if self.next_payload is not None else payload
        self.next_payload = None
        part.update(state="feed", defect=defect, x=FEED_X0, kg=PART_KG + float(self.rng.normal(0, 0.012)) if kg is None else kg,
                    torque_ok=bool(self.rng.random() > self.torque_ng_rate))
        part["conn"].show(defect != "connector")
        part["g"].pos(*self.W(FEED_X0, 0, 0.012))
        part["g"].rot(0, 0, 0)
        part["g"].show(True)
        self.part = part
        return part

    def start_cycle(self):
        """한 사이클: 부품 공급 → 비전 판정 → 집기 → 분기 → 놓기 → 대기."""
        if self.estop or self.trip or self.move or self.moves or self.part is not None:
            return
        if self.feed_part() is not None:
            self.cycle += 1
            self.cyc_tau = []
            self.phase = "부품 공급"

    def _on_part_arrived(self):
        p = self.part
        vision_ok = self.judge_fn(p) if self.judge_fn else (p["defect"] == "none")
        p["vision_ok"] = vision_ok
        final_ok = vision_ok and p["torque_ok"]
        p["decision"] = "ok" if final_ok else "ng"
        why = [] if final_ok else (["외관: 커넥터 없음"] if not vision_ok else []) + ([] if p["torque_ok"] else ["체결 이력: 토크 부족"])
        self.set_label("cam_res", "판정: 양품" if final_ok else "판정: 재검  (" + ", ".join(why) + ")")
        self.msg = f"{self.cycle}번 부품  " + ("양품 판정" if final_ok else "재검 판정: " + ", ".join(why))
        if self.level == 1:                    # 읽기 전용: 판정은 화면에만 · 전부 다음 공정으로
            p["dest"] = "ok"
        elif self.level == 2 and not final_ok:  # 권고: 사람이 결정
            self.pending = p
            self.phase = "사람의 결정을 기다림"
            return
        else:
            p["dest"] = p["decision"]
        self._pick_sequence()

    def decide(self, dest):
        """수준 2 에서 사람이 내린 결정 (ok / ng)."""
        if self.pending is not None:
            self.pending["dest"] = dest
            self.pending = None
            self._pick_sequence()

    def _pick_sequence(self):
        self.goto("pick_above", "집기 위로")
        self.goto("pick", "집기", on_done=self._grip, speed=min(self.speed_pct, 50.0))
        self.goto("pick_above", "들어 올림")
        self._place_sequence(self.part["dest"], "운반")

    def _place_sequence(self, dest, phase):
        n = self.n_ok_tray if dest == "ok" else self.n_ng_tray
        col, row = n % 3, (n // 3) % 2
        above, place = np.array(self.teach[f"{dest}_above"]), np.array(self.teach[f"{dest}_place"])
        off = np.array([(col - 1) * 0.105, (row - 0.5) * 0.085, 0.0])
        self.goto(tuple(above + off), phase)
        self.goto(tuple(place + off), "놓기", on_done=lambda d=dest: self._release(d), speed=min(self.speed_pct, 50.0))
        self.goto(tuple(above + off), "올라옴")
        self.goto("home", "대기로", on_done=self._end_cycle)

    def _grip(self):
        if self.part is None:
            return
        self.gripping, self.part["state"] = True, "grip"
        self.payload_kg = self.part["kg"]
        self._set_payload(self.payload_kg)

    def _release(self, dest):
        p = self.part
        self.gripping = False
        self._set_payload(0.0)
        if p is None:
            return
        x, y, z = self.fk(self.q_cmd)
        p["g"].pos(*self.W(x, y, 0.014))
        p["state"] = "tray"
        if dest == "ok":
            self.n_ok_tray += 1
        else:
            self.n_ng_tray += 1
        truth_ok = p["defect"] == "none" and p["torque_ok"]
        self.counts["ok" if dest == "ok" else "recheck"] += 1
        if (dest == "ok") != truth_ok:
            self.counts["wrong"] += 1
        self.part = None
        for tray, n in (("ok", self.n_ok_tray), ("ng", self.n_ng_tray)):       # 트레이가 차면 비움
            if n >= 6:
                for q in self.parts:
                    if q["state"] == "tray":
                        q["state"] = "free"; q["g"].show(False)
                self.n_ok_tray = self.n_ng_tray = 0

    def _end_cycle(self):
        if len(self.cyc_tau) > 5:
            a = np.abs(np.array(self.cyc_tau))
            feat = np.array([a[:, 0].mean(), a[:, 0].max(), a[:, 1].mean()])
            score = 0.0 if self.base_feat is None else float(np.max(np.abs(feat - self.base_feat[0]) / self.base_feat[1]))
            self.cycles.append(dict(cycle=self.cycle, feat=feat, score=score, kg=self.payload_kg))
        self.phase = "대기"

    def learn_baseline(self, n=5):
        """최근 n 사이클을 정상으로 보고 기준을 잡음."""
        recent = [c["feat"] for c in list(self.cycles)[-n:]]
        if len(recent) < 3:
            self.msg = "정상 기준 학습: 사이클이 3회 이상 필요"
            return False
        f = np.array(recent)
        self.base_feat = (f.mean(axis=0), f.std(axis=0) + 0.02)
        for c in self.cycles:
            c["score"] = float(np.max(np.abs(c["feat"] - self.base_feat[0]) / self.base_feat[1]))
        self.msg = f"정상 기준 학습 완료 (최근 {len(recent)}회)"
        return True

    # ======================================================================
    # 매 프레임
    # ======================================================================
    def step(self, dt_):
        self.t += dt_
        blink = int(self.t * 3) % 2 == 0
        # 작업자 이동 · 라이트 커튼
        goal = 0.35 if self.person_in else 3.2
        self.person_x += float(np.clip(goal - self.person_x, -0.9 * dt_, 0.9 * dt_))
        py = self.person_y0 + (0.62 * min((1.7 - self.person_x) / 0.5, 1.0) if self.person_x < 1.7 else 0.0)
        self.person.pos(self.person_x, py, 0)
        intruding = py > self.person_y0 + 0.4
        if intruding and not self.estop:
            self.press_estop("방호 영역에 작업자 진입")
        for c in self.curtain:
            c.color(C_RED if intruding else C_GREEN)
        # 부품 공급
        p = self.part
        if p is not None and p["state"] == "feed" and not (self.estop or self.trip):
            p["x"] = max(p["x"] - 0.35 * dt_, self.teach["pick"][0])
            p["g"].pos(*self.W(p["x"], 0, 0.012))
            if p["x"] <= self.teach["pick"][0]:
                p["state"], self.vision_wait, self.phase = "vision", 24, "비전 판정"
        if p is not None and p["state"] == "vision" and not (self.estop or self.trip):
            self.vision_wait -= 1
            if self.vision_wait == 10 and self.request_fn:
                self.request_fn()                    # 영상 요청 → 몇 프레임 뒤 도착
            if self.vision_wait <= 0:
                p["state"] = "ready"
                self._on_part_arrived()
        # 동작 진행
        if not (self.estop or self.trip):
            if self.move is None and self.moves:
                self._begin_move()
            if self.move is not None:
                m = self.move
                m["t"] += dt_
                u = min(m["t"] / m["T"], 1.0)
                s = u * u * (3 - 2 * u)
                # 흡착 패드 끝을 직선으로 옮기며 매 순간 역기구학을 풂 → 패드가 항상 아래를 향함
                q = self.ik(*(m["p0"] + (m["p"] - m["p0"]) * s)) if u < 1.0 else m["q"]
                self.q_cmd = np.array(q) if q is not None else m["q0"] + (m["q"] - m["q0"]) * s
                if u >= 1.0:
                    self.move = None
                    if m["on_done"]:
                        m["on_done"]()
            if self.auto and self.move is None and not self.moves and self.part is None and self.pending is None:
                self.start_cycle()
        if self.robot is not None:
            self._set_targets()
            self.q = np.degrees(self.robot.get_joint_positions())
            self.track_err = np.maximum(getattr(self, 'track_err', np.zeros(6)), np.abs(self.q - self.q_cmd))
            tau = self.robot.get_measured_joint_efforts()
            if tau is not None:
                self.tau = 0.7 * self.tau + 0.3 * np.asarray(tau, dtype=float)       # 표시용으로 조금 평활
            self.grace = max(getattr(self, "grace", 0) - 1, 0)
            if abs(self.tau[1]) > self.torque_limit and not self.trip and not self.estop and self.grace == 0:
                self.checks["torque"] = True
                self.stop_all(f"J2 토크 {abs(self.tau[1]):.1f} N·m 가 임계 {self.torque_limit:.1f} 초과")
                self.msg = "인터록: 토크 임계 초과 → 보호 정지"
        if self.gripping and self.part is not None:
            x, y, z = self.fk(self.q)
            self.part["g"].pos(*self.W(x, y, z - PART[2] - 0.002))
            if self.phase in ("운반", "들어 올림"):
                self.cyc_tau.append((self.tau[1], self.tau[2]))
        if self._log and int(self.t * 60) % 3 == 0:
            self._log[1].writerow([f"{self.t:.3f}", self.cycle, self.phase, *[f"{v:.2f}" for v in self.q],
                                   *[f"{v:.3f}" for v in self.tau], f"{min(self.speed_pct, self.speed_limit):.0f}", int(self.gripping), f"{self.vacuum:.1f}"])
            self._log_n += 1
            if self._log_n % 60 == 0:
                self._log[0].flush()
        stopped = self.estop or bool(self.trip)
        self.lamp.color((C_RED if blink else C_DARK) if stopped else (C_YELLOW if self.pending is not None else C_GREEN))
        self.estop_btn.pos(self.estop_pos[0], self.estop_pos[1] + (0.015 if self.estop else 0.0), self.estop_pos[2])
        target = VACUUM_KPA if self.gripping else 0.0
        self.vacuum += (target - self.vacuum) * 0.25 + (float(self.rng.normal(0, 0.4)) if self.gripping else 0.0)
        self.cup.color((0.1, 0.55, 0.95) if self.gripping else (0.15, 0.15, 0.16))
        self.set_label("status", f"{self.phase}   J2 {self.tau[1]:+.1f} N·m   J3 {self.tau[2]:+.1f} N·m   진공 {self.vacuum:+.0f} kPa")

    def close(self):
        self.stop_log()
