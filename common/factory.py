"""가상 가전공장: 사출 셀 3대 → 컨베이어 → 외관 검사기, 데이터 수집 경로, 도장 부스.

- 물리 엔진을 쓰지 않고 위치를 직접 움직임(재현성 확보). 결함과 데이터는 common/process_model.py 가 만듦.
- step(dt) 를 매 프레임 불러 주면 됨. 화면 패널은 회차별 실행 파일에서 따로 만듦.
"""
import csv
import datetime as dt
from collections import deque
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom

from . import process_model as pm
from .scene_kit import Kit, Path3

# ---- 배치 (m) ----------------------------------------------------------------
MACH_ID = tuple(pm.MACHINE_X)
MACH_X = tuple(pm.MACHINE_X.values())
BELT_Z = 0.83                   # 벨트 위 부품 중심 높이
BELT_V = pm.BELT_SPEED          # 벨트 속도 (m/s, 공정 시간 기준)
INSPECT_X, DIVERT_X, END_X = pm.INSPECT_X, 3.6, 5.9
Y_OPEN, Y_CLOSED = 2.15, 2.75   # 가동측 형판 위치
TAKEOUT_S = pm.TAKEOUT_S        # 취출에 걸리는 공정 시간
BOOTH_X0, BOOTH_X1, RAIL_Y = 11.5, 15.5, 1.0
HALL_POS, HALL_SCALE = (0.0, 1.5, 0.0), 1.6          # 공장 건물(창고 에셋)의 위치와 배율
PANEL_START_X, PANEL_END_X = 9.6, 19.6
PANEL_V, PANEL_TAKT_S = 0.06, 48.0
SPRAY_X = (12.4, 13.4)
CLOCK_OFFSET_S = 432            # 시각 오차 재현 값 (7분 12초)

C_OK = (0.93, 0.93, 0.90)
C_GREEN, C_RED, C_YELLOW, C_DARK = (0.1, 0.85, 0.2), (0.95, 0.1, 0.1), (1.0, 0.75, 0.05), (0.12, 0.12, 0.12)
C_CYAN, C_ORANGE, C_MAGENTA = (0.1, 0.8, 1.0), (1.0, 0.5, 0.05), (1.0, 0.1, 0.9)

CAMERAS = {   # 이름: (눈 위치, 보는 곳)
    "overview": ((4.5, -21.5, 9.5), (4.5, 2.8, 2.8)),
    "injection": ((-14.2, -1.2, 3.9), (-9.6, 3.4, 1.5)),
    "dataflow": ((-3.0, -11.0, 9.5), (-1.5, 5.0, 1.5)),
    "inspection": ((5.2, -6.0, 4.6), (1.9, 0.3, 0.9)),
    "paint": ((15.0, -10.5, 3.4), (15.0, 1.0, 2.0)),
    "server": ((4.8, 0.2, 3.6), (4.8, 6.5, 1.6)),
    "samples": ((-6.0, -7.4, 2.7), (-6.0, -3.4, 1.4)),         # 결함 견본대
}

ERRORS = {   # 키: (표시 이름, 설명)
    "missing": ("결측", "IMM-01 금형온도 센서 단선"),
    "outlier": ("이상치", "통신 오류 값 0 · 999.9"),
    "unit": ("단위 혼재", "IMM-03 압력을 bar 로 전송"),
    "clock": ("시각 오차", "IMM-02 PLC 시계 +7분 12초"),
    "dup": ("중복", "게이트웨이 재전송"),
}
SHOT_COLS = ["timestamp", "machine_id", "lot_id", "shot_no", *pm.INJ_NOMINAL, "cycle_time_s", "ambient_temp_c"]
INSPECT_COLS = ["inspect_id", "inspect_time", "machine_id", "result", "defect_type"]
PAINT_COLS = ["timestamp", "booth_id", "lot_id", *pm.PAINT_NOMINAL, "defect_type", "is_defect"]


class Machine:
    def __init__(self, idx):
        self.idx, self.id, self.x = idx, MACH_ID[idx], MACH_X[idx]
        self.model = pm.InjectionMachine(seed=100 + idx)
        self.t = idx * 24.0          # 사이클 안에서 흐른 시간 (설비마다 어긋나게 시작)
        self.shot = self.model.next_shot()
        self.published = False
        self.tags = {}               # 화면에 보이는 마지막 전송 값
        self.flash = 0.0             # 이상치 발생 표시 남은 시간


class VirtualFactory:
    def __init__(self, stage, log_dir, seed=7):
        self.stage = stage
        self.k = Kit(stage)
        self.rng = np.random.default_rng(seed)
        self.log_dir = Path(log_dir)
        self.speed = 12.0            # 공정 시간 배속
        self.paused = False
        self.errors = {key: False for key in ERRORS}
        self.machines = [Machine(i) for i in range(3)]
        self.booth = pm.PaintBooth(seed=5)
        self.clock = dt.datetime(2026, 12, 8, 13, 30, 0)
        self.wall = 0.0              # 실제 흐른 시간 (깜박임·패킷용)
        self.labels = {}             # 키: dict(pos, text, size, color)
        self.labels_dirty = True
        self.packets = []            # 이동 중인 데이터 패킷
        self.parts, self.panels = [], []
        self.paint_wait = 0.0
        self.inspect_flash = 0.0
        self.pusher_t = 0.0
        self._files = {}
        self.reset_stats(new_files=False)
        self._build()

    # ======================================================================
    # 통계·기록
    # ======================================================================
    def reset_stats(self, new_files=True):
        self.n_shot = self.n_ng = self.n_inspect = 0
        self.by_type = {key: 0 for key in ("none", *pm.INJ_DEFECTS)}
        self.recent = deque(maxlen=30)          # 최근 30샷 양·불
        self.rate_hist = deque([0.0] * 120, maxlen=120)
        self.p_n = self.p_ng = 0
        self.p_by_type = {key: 0 for key in ("none", *pm.PAINT_DEFECTS)}
        self.p_recent = deque(maxlen=20)
        self.rows_sent = 0
        self.last_rows = deque(maxlen=8)        # 패널에 보여 줄 최근 전송 행
        if new_files:
            self.open_logs()

    def open_logs(self):
        for f, _ in self._files.values():
            f.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._files = {}
        for name, cols in (("shots", SHOT_COLS), ("inspection", INSPECT_COLS), ("paint", PAINT_COLS)):
            f = open(self.log_dir / f"{name}_{stamp}.csv", "w", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            self._files[name] = (f, w)
        self.log_stamp = stamp

    def _write(self, name, row):
        if name in self._files:
            f, w = self._files[name]
            w.writerow(row)
            f.flush()

    def close(self):
        for f, _ in self._files.values():
            f.close()
        self._files = {}

    @property
    def defect_rate(self):
        return self.n_ng / self.n_inspect if self.n_inspect else 0.0

    @property
    def recent_rate(self):
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def set_setpoint(self, key, value, machine=None):
        for m in self.machines:
            if machine is None or m.idx == machine:
                m.model.setpoint[key] = float(value)

    def expected_rate(self):
        return float(np.mean([m.model.expected_defect_rate() for m in self.machines]))

    # ======================================================================
    # 장면 만들기
    # ======================================================================
    def label(self, key, text, pos, size=16, color=(1, 1, 1), detail=False):
        """detail=True 인 라벨은 공장 전체 화면에서는 숨김(겹침 방지)."""
        self.labels[key] = dict(text=text, pos=pos, size=size, color=color, detail=detail)
        self.labels_dirty = True

    def set_label(self, key, text):
        if self.labels[key]["text"] != text:
            self.labels[key]["text"] = text
            self.labels_dirty = True

    def _build(self):
        k = self.k
        # 공장 건물 (에셋이 있을 때). 카메라 쪽(앞) 벽과 그쪽 기둥은 떼어 내서 안이 들여다보이게 함
        self.indoor = k.hall(pos=HALL_POS, yaw=-90.0, scale=HALL_SCALE,
                             hide=lambda name, lo, hi: lo[0] > 8.5 or ("Pillar" in name and hi[0] > 9.0))
        if self.indoor:
            k.lights(dome=140.0, sun=0.0)
            n = 0
            for x in np.arange(-15.0, 26.0, 5.0):
                for y in (-3.5, 3.0, 9.0):
                    k.area_light(f"/World/Lights/Ceil{n}", (x, y, 9.5), (2.2, 1.2), 14000.0)
                    n += 1
        else:
            k.lights()
        self.m_steel = k.mat((0.6, 0.62, 0.65), rough=0.28, metal=0.9)
        self.m_body = k.mat((0.84, 0.86, 0.85), rough=0.32)                # 설비 몸체: 밝은 회색 도장
        self.m_accent = k.mat((0.04, 0.27, 0.52), rough=0.35)              # 설비 강조색: 파랑
        self.m_dark = k.mat((0.06, 0.065, 0.07), rough=0.5)
        self.m_frame = k.mat((0.95, 0.7, 0.04), rough=0.4)
        self.m_cab = k.mat((0.78, 0.8, 0.8), rough=0.38)
        self.m_glass = k.mat((0.75, 0.9, 1.0), rough=0.1, opacity=0.16, name="Glass")
        self.m_spray = k.mat((1.0, 1.0, 1.0), rough=1.0, opacity=0.4, emissive=(0.9, 0.9, 0.9), name="Spray")
        self.m_fog = k.mat((0.75, 0.85, 1.0), rough=1.0, opacity=0.0, name="Fog")
        if not self.indoor:
            k.box("/World/Floor", (44, 26, 0.1), (4, 2, -0.05), mat=k.mat((0.16, 0.17, 0.19), rough=0.8))
        for i, (y0, y1) in enumerate(((-1.6, -1.5), (8.4, 8.5))):      # 안전 통로 선
            k.box(f"/World/Lane{i}", (30, 0.1, 0.005), (-1, (y0 + y1) / 2, 0.003), color=(0.95, 0.8, 0.1))
        self._build_conveyor()
        for m in self.machines:
            self._build_machine(m)
        self._build_inspection()
        self._build_datapath()
        self._build_paint()
        self._build_samples()
        self._build_pools()

    def _build_conveyor(self):
        k, x0, x1 = self.k, -11.6, END_X + 0.3
        k.box("/World/Conveyor/Belt", (x1 - x0, 0.8, 0.06), ((x0 + x1) / 2, 0, 0.77), mat=self.m_dark)
        for j, y in enumerate((-0.43, 0.43)):
            k.box(f"/World/Conveyor/Rail{j}", (x1 - x0, 0.06, 0.16), ((x0 + x1) / 2, y, 0.76), mat=self.m_steel)
        for j, x in enumerate(np.arange(x0 + 0.3, x1, 1.9)):
            for s, y in enumerate((-0.4, 0.4)):
                k.box(f"/World/Conveyor/Leg{j}_{s}", (0.07, 0.07, 0.7), (x, y, 0.35), mat=self.m_steel)
        # 벨트가 움직이는 것처럼 보이게 하는 줄무늬
        self.stripes = [k.box(f"/World/Conveyor/Stripe{j}", (0.04, 0.74, 0.005), (0, 0, 0.802), color=(0.22, 0.22, 0.24))
                        for j in range(24)]
        self.belt_x0, self.belt_x1, self.belt_shift = x0, x1, 0.0
        # 불량 배출 푸셔·슈트·불량 박스, 양품 적재 박스
        self.pusher = k.box("/World/Divert/Pusher", (0.5, 0.12, 0.2), (DIVERT_X, 0.62, 0.95), color=C_ORANGE)
        k.box("/World/Divert/Cyl", (0.12, 0.7, 0.12), (DIVERT_X, 1.05, 0.95), mat=self.m_steel)
        k.box("/World/Divert/Chute", (1.1, 1.0, 0.04), (DIVERT_X, -0.9, 0.6), mat=self.m_steel, rot=(28, 0, 0))
        self._bin("/World/Divert/NGBin", (DIVERT_X, -1.9, 0.0), (1.3, 0.9, 0.55), (0.75, 0.12, 0.1))
        self._bin("/World/OKBox", (END_X + 1.0, 0.0, 0.0), (1.3, 1.0, 0.6), (0.1, 0.45, 0.8))
        self.label("ng_bin", "불량 박스", (DIVERT_X, -1.9, 1.1), 15, (1.0, 0.5, 0.45), detail=True)
        self.label("ok_box", "양품 적재", (END_X + 1.0, 0.0, 1.2), 15, (0.5, 0.8, 1.0), detail=True)

    def _bin(self, path, pos, size, color):
        k, (x, y, z), (sx, sy, sz) = self.k, pos, size
        mat = k.mat(color, rough=0.6)
        k.box(path + "/Bottom", (sx, sy, 0.04), (x, y, z + 0.02), mat=mat)
        k.box(path + "/W0", (sx, 0.04, sz), (x, y - sy / 2, z + sz / 2), mat=mat)
        k.box(path + "/W1", (sx, 0.04, sz), (x, y + sy / 2, z + sz / 2), mat=mat)
        k.box(path + "/W2", (0.04, sy, sz), (x - sx / 2, y, z + sz / 2), mat=mat)
        k.box(path + "/W3", (0.04, sy, sz), (x + sx / 2, y, z + sz / 2), mat=mat)

    def _build_machine(self, m):
        k, x, p = self.k, m.x, f"/World/{m.id.replace('-', '_')}"
        k.box(p + "/Base", (1.7, 5.6, 0.9), (x, 4.3, 0.45), mat=self.m_body)
        k.box(p + "/Plinth", (1.76, 5.66, 0.22), (x, 4.3, 0.11), mat=self.m_accent)
        k.box(p + "/BaseStripe", (1.72, 5.62, 0.06), (x, 4.3, 0.86), mat=self.m_accent)
        for j in range(5):                                                   # 몸체 옆면의 점검문
            k.box(p + f"/Door{j}", (0.02, 0.86, 0.5), (x - 0.86, 2.2 + j * 1.05, 0.5), mat=self.m_cab)
            k.box(p + f"/DoorHandle{j}", (0.03, 0.04, 0.16), (x - 0.88, 2.55 + j * 1.05, 0.5), mat=self.m_dark)
        k.box(p + "/EndPlaten", (1.5, 0.3, 1.3), (x, 1.75, 1.55), mat=self.m_steel)
        k.box(p + "/FixedPlaten", (1.5, 0.3, 1.3), (x, 3.53, 1.55), mat=self.m_steel)
        k.box(p + "/FixedMold", (1.05, 0.25, 0.8), (x, 3.25, 1.55), mat=k.mat((0.75, 0.6, 0.35), rough=0.3, metal=0.9))
        m.platen = k.box(p + "/MovingPlaten", (1.4, 0.25, 1.2), (x, Y_OPEN, 1.55), mat=self.m_steel)
        m.mold = k.box(p + "/MovingMold", (1.05, 0.25, 0.8), (x, Y_OPEN + 0.25, 1.55),
                       mat=k.mat((0.75, 0.6, 0.35), rough=0.3, metal=0.9))
        for j, (dx, z) in enumerate(((-0.6, 1.05), (0.6, 1.05), (-0.6, 2.05), (0.6, 2.05))):
            k.cyl(p + f"/TieBar{j}", 0.045, 1.8, (x + dx, 2.64, z), axis="Y", mat=k.mat((0.8, 0.8, 0.82), 0.2, 1.0))
        k.cyl(p + "/Barrel", 0.17, 1.9, (x, 4.6, 1.55), axis="Y", mat=self.m_steel)
        for j in range(4):
            k.cyl(p + f"/Heater{j}", 0.2, 0.26, (x, 3.95 + j * 0.42, 1.55), axis="Y", mat=k.mat((0.85, 0.4, 0.12), 0.5))
        m.unit = k.box(p + "/InjUnit", (1.0, 1.3, 0.9), (x, 6.2, 1.5), mat=self.m_accent)
        k.cyl(p + "/Hopper", 0.3, 0.42, (x, 5.35, 2.36), mat=k.mat((0.8, 0.82, 0.85), 0.25, 0.8))
        k.cone(p + "/HopperCone", 0.3, 0.4, (x, 5.35, 1.95), mat=k.mat((0.8, 0.82, 0.85), 0.25, 0.8), rot=(180, 0, 0))
        k.cyl(p + "/HopperLid", 0.32, 0.04, (x, 5.35, 2.59), mat=self.m_accent)
        k.box(p + "/PurgeGuard", (0.9, 0.5, 0.5), (x, 3.95, 1.95), mat=self.m_glass)
        for j, dx in enumerate((-0.5, 0.5)):                                 # 냉각수 호스
            k.cyl(p + f"/Hose{j}", 0.025, 1.0, (x + dx, 2.9, 0.95), axis="Y", mat=k.mat((0.05, 0.25, 0.75) if j else (0.75, 0.1, 0.08), rough=0.5))
        k.box(p + "/HMI", (0.08, 0.7, 0.9), (x + 0.92, 2.6, 1.55), mat=self.m_cab)
        k.box(p + "/HMIScreen", (0.02, 0.5, 0.35), (x + 0.97, 2.6, 1.7),
              mat=k.mat((0.05, 0.2, 0.5), emissive=(0.1, 0.5, 1.2), name="Screen"))
        k.cyl(p + "/LampPole", 0.03, 0.7, (x + 0.6, 3.53, 2.55), mat=self.m_steel)
        m.lamp = k.cyl(p + "/Lamp", 0.07, 0.22, (x + 0.6, 3.53, 3.0), color=C_GREEN)
        # 취출 로봇 (직교형): 기둥 · 빔 · 수직축
        for j, y in enumerate((-0.75, 3.9)):
            k.box(p + f"/TakeoutPost{j}", (0.12, 0.12, 3.1), (x - 0.95, y, 1.55), mat=self.m_frame)
        k.box(p + "/TakeoutBeam", (0.14, 4.9, 0.14), (x - 0.95, 1.57, 3.1), mat=self.m_frame)
        m.carriage = k.box(p + "/TakeoutCarriage", (1.1, 0.22, 0.16), (x - 0.4, 2.9, 3.1), mat=self.m_frame)
        m.arm = k.box(p + "/TakeoutArm", (0.07, 0.07, 1.0), (x, 2.9, 2.6), mat=self.m_steel)
        # 센서 (금형온도 · 수지온도 · 압력)
        m.s_mold = k.sphere(p + "/SensorMold", 0.08, (x, 3.25, 2.0), color=C_GREEN)
        m.s_melt = k.sphere(p + "/SensorMelt", 0.08, (x, 4.6, 1.78), color=C_GREEN)
        m.s_press = k.sphere(p + "/SensorPress", 0.08, (x, 6.2, 2.0), color=C_GREEN)
        # PLC 반
        k.box(p + "/PLC", (0.6, 0.45, 1.6), (x + 1.45, 5.0, 0.8), mat=self.m_cab)
        k.box(p + "/PLCLamp", (0.2, 0.02, 0.08), (x + 1.45, 4.77, 1.4), color=C_GREEN)
        self.label(m.id, m.id, (x, 4.2, 3.9), 22, (1, 1, 1))
        self.label(m.id + "_tag", "", (x, 4.2, 3.45 - 0.45 * (m.idx % 2)), 14, (0.6, 0.95, 1.0), detail=True)
        self.label(m.id + "_plc", "PLC", (x + 1.45, 5.0, 1.85), 14, (0.9, 0.9, 0.9), detail=True)
        for a, b, c in ((m.s_mold, (x, 3.25, 2.0), 0), (m.s_melt, (x, 4.6, 1.78), 1), (m.s_press, (x, 6.2, 2.0), 2)):
            k.line(p + f"/Wire{c}", [b, (x + 1.45, b[1], b[2]), (x + 1.45, 5.0, 1.62)], 0.02, (0.2, 0.75, 0.3))

    def _build_inspection(self):
        k, x = self.k, INSPECT_X
        for j, (dx, dy) in enumerate(((-0.7, -0.6), (0.7, -0.6), (-0.7, 0.6), (0.7, 0.6))):
            k.box(f"/World/Inspect/Post{j}", (0.08, 0.08, 2.5), (x + dx, dy, 1.25), mat=self.m_frame)
        k.box("/World/Inspect/Top", (1.6, 1.4, 0.08), (x, 0, 2.5), mat=self.m_frame)
        k.box("/World/Inspect/CamBody", (0.18, 0.18, 0.3), (x, 0, 2.3), mat=self.m_dark)
        k.cyl("/World/Inspect/Lens", 0.06, 0.12, (x, 0, 2.1), mat=self.m_dark)
        for j, (dx, sx, sy) in enumerate(((-0.45, 0.08, 0.9), (0.45, 0.08, 0.9))):
            k.box(f"/World/Inspect/Light{j}", (sx, sy, 0.05), (x + dx, 0, 2.0),
                  mat=k.mat((1, 1, 1), emissive=(6, 6, 5.6), name="LightBar"))
        self.lamp_ok = k.sphere("/World/Inspect/LampOK", 0.09, (x + 0.45, -0.62, 2.72), color=C_DARK)
        self.lamp_ng = k.sphere("/World/Inspect/LampNG", 0.09, (x + 0.75, -0.62, 2.72), color=C_DARK)
        k.box("/World/Inspect/PC", (0.5, 0.4, 1.3), (x + 1.2, 1.1, 0.65), mat=self.m_cab)
        cam = UsdGeom.Camera.Define(self.stage, "/World/Inspect/Camera")
        cam.CreateFocalLengthAttr(16.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
        xf = UsdGeom.Xformable(cam)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, 1.9))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))     # USD 카메라는 -Z(아래)를 봄 → 벨트가 가로로 보임
        self.label("inspect", "외관 검사기", (x, 0, 3.2), 20, (1, 1, 1))
        self.label("inspect_res", "", (x, 0, 2.85), 16, (0.7, 1.0, 0.7), detail=True)

    def _build_datapath(self):
        k, TZ, TY = self.k, 3.6, 7.6           # 케이블 트레이 높이·위치
        gw, sv, mes = (2.6, 6.6), (4.6, 6.6), (7.4, 6.6)
        k.box("/World/IT/Tray", (15.6, 0.3, 0.05), (-3.6, TY, TZ), mat=self.m_steel)
        for j, x in enumerate(np.arange(-11.0, 4.5, 3.0)):
            k.box(f"/World/IT/TrayPost{j}", (0.07, 0.07, TZ), (x, TY + 0.2, TZ / 2), mat=self.m_steel)
        k.box("/World/IT/Gateway", (0.7, 0.45, 1.5), (gw[0], gw[1], 0.75), mat=self.m_cab)
        k.box("/World/IT/GatewayFace", (0.5, 0.02, 0.25), (gw[0], gw[1] - 0.23, 1.2),
              mat=k.mat((0.05, 0.3, 0.1), emissive=(0.1, 1.5, 0.3), name="GwFace"))
        k.box("/World/IT/Server", (0.8, 0.9, 2.0), (sv[0], sv[1], 1.0), mat=self.m_dark)
        for j in range(6):
            k.box(f"/World/IT/ServerSlot{j}", (0.66, 0.02, 0.1), (sv[0], sv[1] - 0.455, 0.45 + j * 0.25),
                  mat=k.mat((0.05, 0.1, 0.3), emissive=(0.1, 0.4, 1.6), name="SrvSlot"))
        k.box("/World/IT/MesStand", (0.12, 0.12, 1.3), (mes[0], mes[1], 0.65), mat=self.m_steel)
        k.box("/World/IT/MesScreen", (2.2, 0.08, 1.3), (mes[0], mes[1], 1.9), mat=self.m_dark)
        k.box("/World/IT/MesFace", (2.05, 0.02, 1.15), (mes[0], mes[1] - 0.05, 1.9),
              mat=k.mat((0.03, 0.08, 0.16), emissive=(0.05, 0.16, 0.34), name="MesFace"))
        self.label("gw", "게이트웨이", (gw[0], gw[1], 1.85), 17, (0.7, 1.0, 0.75))
        self.label("sv", "수집 서버 (DB)", (sv[0], sv[1], 2.35), 17, (0.7, 0.85, 1.0))
        self.label("mes", "생산관리시스템 (MES)", (mes[0], mes[1], 2.85), 17, (1.0, 0.9, 0.6))
        self.label("mes_val", "", (mes[0], mes[1] - 0.2, 1.9), 17, (1, 1, 1), detail=True)
        tail = [(gw[0], TY, TZ), (gw[0], gw[1], 1.5), (sv[0], sv[1], 1.7), (mes[0], mes[1], 1.9)]
        for m in self.machines:
            pts = [(m.x, 3.25, 2.0), (m.x + 1.45, 3.25, 2.0), (m.x + 1.45, 5.0, 1.62),
                   (m.x + 1.45, TY, TZ), *tail]
            m.path = Path3(pts)
            k.line(f"/World/IT/Cable{m.idx}", pts[2:], 0.035, (0.15, 0.55, 0.95))
        ipts = [(INSPECT_X + 1.2, 1.1, 1.3), (INSPECT_X + 1.2, gw[1], 1.3), (sv[0], sv[1], 1.7), (mes[0], mes[1], 1.9)]
        self.inspect_path = Path3(ipts)
        k.line("/World/IT/CableInspect", ipts, 0.035, (0.95, 0.6, 0.15))

    def _build_paint(self):
        k, xc, y = self.k, (BOOTH_X0 + BOOTH_X1) / 2, RAIL_Y
        L = BOOTH_X1 - BOOTH_X0
        k.box("/World/Paint/FloorPlate", (L, 3.2, 0.04), (xc, y, 0.02), mat=k.mat((0.3, 0.32, 0.34), 0.7))
        k.box("/World/Paint/Roof", (L, 3.2, 0.1), (xc, y, 3.45), mat=self.m_cab)
        k.box("/World/Paint/Back", (L, 0.06, 3.4), (xc, y + 1.6, 1.7), mat=self.m_cab)
        for j, x in enumerate((BOOTH_X0, BOOTH_X1)):
            for s, dy in enumerate((-1.6, 1.6)):
                k.box(f"/World/Paint/Corner{j}{s}", (0.1, 0.1, 3.4), (x, y + dy, 1.7), mat=self.m_frame)
        # 부스 온도·습도 게이지 (입구 옆 기둥)
        self.gauges = {}
        for j, (key, name) in enumerate((("booth_temp_c", "온도"), ("booth_humidity_pct", "습도"))):
            gx = BOOTH_X0 - 0.9 + j * 0.45
            k.box(f"/World/Paint/Gauge{j}Back", (0.3, 0.06, 2.1), (gx, y - 1.6, 1.25), mat=self.m_dark)
            self.gauges[key] = k.box(f"/World/Paint/Gauge{j}Bar", (0.2, 0.08, 1.0), (gx, y - 1.63, 0.75), color=C_GREEN)
            self.label(f"gauge{j}", name, (gx, y - 1.63, 2.5), 14, (1, 1, 1), detail=True)
        k.box("/World/Paint/Rail", (PANEL_END_X - PANEL_START_X + 0.6, 0.1, 0.1),
              ((PANEL_START_X + PANEL_END_X) / 2, y, 3.2), mat=self.m_steel)
        for j, x in enumerate((PANEL_START_X, 10.9, 16.0, PANEL_END_X)):
            k.box(f"/World/Paint/RailPost{j}", (0.08, 0.08, 3.2), (x, y + 0.5, 1.6), mat=self.m_steel)
            k.box(f"/World/Paint/RailArm{j}", (0.08, 0.5, 0.08), (x, y + 0.25, 3.2), mat=self.m_steel)
        self.sprays = []
        for j, x in enumerate(SPRAY_X):
            k.box(f"/World/Paint/GunPost{j}", (0.1, 0.1, 1.5), (x, y - 1.1, 0.75), mat=self.m_dark)
            k.box(f"/World/Paint/Gun{j}", (0.1, 0.3, 0.12), (x, y - 0.95, 1.55), mat=self.m_frame)
            self.sprays.append(k.cone(f"/World/Paint/Spray{j}", 0.3, 0.8, (x, y - 0.4, 1.6), axis="Y",
                                      mat=self.m_spray, rot=(0, 0, 180)))
        # 건조로
        ox0, ox1 = 16.2, 18.4
        k.box("/World/Paint/OvenTop", (ox1 - ox0, 1.4, 0.1), ((ox0 + ox1) / 2, y, 3.05), mat=self.m_cab)
        k.box("/World/Paint/OvenBack", (ox1 - ox0, 0.08, 2.9), ((ox0 + ox1) / 2, y + 0.7, 1.55),
              mat=k.mat((0.5, 0.2, 0.08), emissive=(1.6, 0.5, 0.1), name="OvenBack"))
        k.box("/World/Paint/EnvSensor", (0.25, 0.08, 0.35), (BOOTH_X0 + 0.5, y + 1.55, 2.3),
              mat=k.mat((0.9, 0.9, 0.9), emissive=(0.2, 0.8, 0.3), name="EnvSensor"))
        self.label("booth", "도장 부스 PB-01", (xc, y, 4.3), 22, (1, 1, 1))
        self.label("booth_tag", "", (xc, y, 3.85), 15, (0.6, 0.95, 1.0), detail=True)
        self.label("oven", "건조로", ((ox0 + ox1) / 2, y, 3.5), 16, (1.0, 0.8, 0.6), detail=True)
        self.label("booth_res", "", (19.0, y, 3.6), 16, (0.7, 1.0, 0.7), detail=True)

    def _make_part(self, p):
        """사출품 하나: 몸체 + 흐름자국(줄무늬) + 수축(움푹한 자국). 결함 표시는 켜고 끌 수 있음."""
        k = self.k
        g = k.group(p)
        body = k.box(p + "/Body", (0.9, 0.5, 0.06), color=C_OK)
        flow = k.group(p + "/Flow")
        for j in range(5):
            k.box(p + f"/Flow/S{j}", (0.035, 0.44 - 0.03 * (j % 2), 0.004), (-0.3 + j * 0.13, 0, 0.032),
                  color=(0.3, 0.3, 0.28), rot=(0, 0, 14 - j * 4))
        sink = k.group(p + "/Sink")
        for j, (sx, sy) in enumerate(((-0.18, 0.06), (0.2, -0.08))):
            k.sphere(p + f"/Sink/D{j}", 0.085, (sx, sy, 0.028), color=(0.28, 0.28, 0.26), scale=(1, 0.8, 0.12))
        return dict(g=g, body=body, flow=flow, sink=sink, state="free")

    def _set_defect(self, part, d):
        part["flow"].show(d == "flow_mark")
        part["sink"].show(d == "sink")
        if d == "short_shot":
            part["body"].scale(0.58, 0.5, 0.06)
            part["body"].pos(-0.19, 0, 0)
        else:
            part["body"].scale(0.9, 0.5, 0.06)
            part["body"].pos(0, 0, 0)

    def _build_samples(self):
        """결함 견본대: 양품과 결함 세 종을 나란히 세워 둠."""
        k, y, z, xc = self.k, -3.4, 1.25, -6.0          # 사출기 앞 통로 쪽
        k.box("/World/Samples/Board", (4.6, 0.06, 0.95), (xc, y + 0.05, z), mat=self.m_dark, rot=(-55, 0, 0))
        for j, x in enumerate((xc - 2.1, xc + 2.1)):
            k.box(f"/World/Samples/Leg{j}", (0.08, 0.08, 1.0), (x, y + 0.25, 0.5), mat=self.m_steel)
        for j, d in enumerate(("none", "flow_mark", "sink", "short_shot")):
            x = xc - 1.65 + j * 1.1
            part = self._make_part(f"/World/Samples/S{j}")
            self._set_defect(part, d)
            part["g"].pos(x, y, z + 0.03)
            part["g"].rot(35, 0, 0)
            self.label(f"sample{j}", pm.DEFECT_KO[d], (x, y + 0.3, z + 0.5), 17,
                       (0.6, 1.0, 0.6) if d == "none" else (1.0, 0.6, 0.5), detail=True)
        self.label("samples", "결함 견본", (xc, y + 0.3, z + 0.95), 18, (1, 1, 1), detail=True)

    def _build_pools(self):
        k = self.k
        for i in range(28):                              # 사출품 (도어 트림)
            part = self._make_part(f"/World/Parts/P{i:02d}")
            part["g"].show(False)
            self.parts.append(part)
        for i in range(10):                              # 도장 패널 (외판)
            p = f"/World/Panels/N{i:02d}"
            g = k.group(p)
            body = k.box(p + "/Body", (0.9, 0.04, 1.3), (0, 0, -1.25), color=(0.55, 0.56, 0.58))
            k.box(p + "/Hanger", (0.03, 0.03, 0.6), (0, 0, -0.3), mat=self.m_steel)
            sag = k.group(p + "/Sag")
            for j, (dx, h) in enumerate(((-0.25, 0.42), (0.02, 0.6), (0.28, 0.34))):
                k.box(p + f"/Sag/D{j}", (0.05, 0.012, h), (dx, -0.022, -1.55 + h / 2 - 0.1), color=(0.72, 0.7, 0.6))
                k.sphere(p + f"/Sag/B{j}", 0.04, (dx, -0.022, -1.65 + 0.0), color=(0.72, 0.7, 0.6), scale=(1, 0.4, 1.3))
            pin = k.group(p + "/Pin")
            for j in range(9):
                k.sphere(p + f"/Pin/H{j}", 0.022, (-0.33 + 0.085 * j + 0.05 * ((j * 7) % 3 - 1), -0.022,
                                                    -1.25 + 0.5 * (((j * 5) % 7) / 3.0 - 1.0)),
                         color=(0.18, 0.18, 0.2), scale=(1, 0.3, 1))
            g.show(False)
            self.panels.append(dict(g=g, body=body, sag=sag, pin=pin, state="free"))
        for i in range(24):                              # 데이터 패킷
            n = k.sphere(f"/World/Packets/K{i:02d}", 0.11, (0, 0, -5), color=C_CYAN)
            n.show(False)
            self.packets.append(dict(n=n, path=None, d=0.0, on=False))

    # ======================================================================
    # 매 프레임 갱신
    # ======================================================================
    def step(self, real_dt):
        self.wall += real_dt
        blink = int(self.wall * 3) % 2 == 0
        pdt = 0.0 if self.paused else real_dt * self.speed
        with Sdf.ChangeBlock():
            if pdt > 0:
                self.clock += dt.timedelta(seconds=pdt)
                for m in self.machines:
                    self._step_machine(m, pdt)
                self._step_parts(pdt)
                self._step_paint(pdt)
                self.belt_shift = (self.belt_shift + BELT_V * pdt) % 0.75
                for j, s in enumerate(self.stripes):
                    s.pos(self.belt_x0 + 0.1 + (j * 0.75 + self.belt_shift) % (self.belt_x1 - self.belt_x0 - 0.2), 0, 0.802)
            self._step_packets(real_dt)
            self._step_indicators(real_dt, blink)

    # ---- 사출기 ----------------------------------------------------------------
    def _step_machine(self, m, pdt):
        if getattr(m, "stopped", False):          # 정지 알람 등으로 멈춘 설비
            return
        m.t += pdt
        T = m.shot.cycle_time_s
        f = m.t / T
        if f < 0.06:
            y = Y_OPEN + (Y_CLOSED - Y_OPEN) * (f / 0.06)
        elif f < 0.80:
            y = Y_CLOSED
        elif f < 0.88:
            y = Y_CLOSED - (Y_CLOSED - Y_OPEN) * ((f - 0.80) / 0.08)
        else:
            y = Y_OPEN
        m.platen.pos(m.x, y, 1.55)
        m.mold.pos(m.x, y + 0.25, 1.55)
        m.unit.pos(m.x, 6.2 - (0.18 if 0.06 <= f < 0.30 else 0.0), 1.5)
        if f >= 0.88 and not m.published:
            m.published = True
            self._publish_shot(m)
            self._spawn_part(m)             # 금형이 열리면 바로 취출 시작
        if f >= 1.0:
            m.t -= T
            m.shot = m.model.next_shot()
            m.published = False

    def _plc_time(self, m):
        t = self.clock
        if self.errors["clock"] and m.idx == 1:
            t += dt.timedelta(seconds=CLOCK_OFFSET_S)
        return t

    def _publish_shot(self, m):
        s = m.shot
        row = dict(timestamp=self._plc_time(m).strftime("%Y-%m-%d %H:%M:%S"), machine_id=m.id,
                   lot_id=f"L{self.clock:%m%d}-{m.id[-2:]}", shot_no=s.shot_no,
                   **{key: getattr(s, key) for key in pm.INJ_NOMINAL},
                   cycle_time_s=s.cycle_time_s, ambient_temp_c=round(22.0 + float(self.rng.normal(0, 0.3)), 1))
        units = dict(mold_temp_c="℃", melt_temp_c="℃", inj_pressure_mpa="MPa", hold_pressure_mpa="MPa",
                     hold_time_s="s", cooling_time_s="s", cycle_time_s="s")
        marks = {}
        if self.errors["unit"] and m.idx == 2:
            for key in ("inj_pressure_mpa", "hold_pressure_mpa"):
                row[key] = round(row[key] * 10.0, 1)
                units[key] = "bar"
                marks[key] = "unit"
        if self.errors["missing"] and m.idx == 0:
            row["mold_temp_c"] = ""
            marks["mold_temp_c"] = "missing"
        if self.errors["outlier"] and self.rng.random() < 0.25:
            key, bad = [("inj_pressure_mpa", 999.9), ("melt_temp_c", 0.0), ("mold_temp_c", 999.9)][self.rng.integers(3)]
            if row[key] != "":
                row[key] = bad
                marks[key] = "outlier"
                m.flash = 2.5
        if self.errors["clock"] and m.idx == 1:
            marks["timestamp"] = "clock"
        dup = self.errors["dup"] and self.rng.random() < 0.35
        m.tags = dict(row=row, units=units, marks=marks, dup=dup)
        for n in range(2 if dup else 1):
            self._write("shots", row)
            self.rows_sent += 1
            self.last_rows.append((row, marks, n == 1))
            self._send_packet(m.path, C_ORANGE if n == 1 else (C_MAGENTA if marks else C_CYAN), delay=0.35 * n)
        self.n_shot += 1
        mt = "  --  " if row["mold_temp_c"] == "" else f"{row['mold_temp_c']:.1f}"
        self.set_label(m.id + "_tag", f"금형 {mt}℃  보압 {row['hold_pressure_mpa']:.1f}{units['hold_pressure_mpa']}"
                                      f"  사이클 {row['cycle_time_s']:.1f}s")

    # ---- 부품 ------------------------------------------------------------------
    def _spawn_part(self, m):
        part = next((p for p in self.parts if p["state"] == "free"), None)
        if part is None:
            return None
        d = m.shot.defect
        part.update(state="takeout", m=m, defect=d, x=m.x, y=0.0, u=0.0, inspected=False)
        self._set_defect(part, d)
        part["g"].pos(m.x, 2.95, 1.55)
        part["g"].rot(90, 0, 0)
        part["g"].show(True)
        return part

    def _step_parts(self, pdt):
        belt = [p for p in self.parts if p["state"] == "belt"]
        for p in self.parts:
            st = p["state"]
            if st == "takeout":
                m = p["m"]
                u = p["u"] + pdt / TAKEOUT_S
                if u > 0.72 and p["u"] <= 0.72 and any(abs(b["x"] - m.x) < 1.15 for b in belt):
                    u = 0.72                               # 벨트 자리가 빌 때까지 위에서 대기
                p["u"] = u
                if u < 0.25:                               # 위로
                    y, z, rx = 2.95, 1.55 + (2.6 - 1.55) * (u / 0.25), 90.0
                elif u < 0.72:                             # 벨트 위로 이동하며 눕힘
                    v = (u - 0.25) / 0.47
                    y, z, rx = 2.95 * (1 - v), 2.6, 90.0 * (1 - v)
                else:                                      # 아래로
                    v = min((u - 0.72) / 0.28, 1.0)
                    y, z, rx = 0.0, 2.6 - (2.6 - BELT_Z) * v, 0.0
                p["g"].pos(m.x, y, z)
                p["g"].rot(rx, 0, 0)
                m.carriage.pos(m.x - 0.4, y, 3.1)
                m.arm.pos(m.x, y, (z + 0.3 + 3.05) / 2)
                m.arm.scale(0.07, 0.07, max(3.05 - (z + 0.3), 0.05))
                if u >= 1.0:
                    p["state"], p["x"] = "belt", m.x
            elif st == "belt":
                p["x"] += BELT_V * pdt
                if not p["inspected"] and p["x"] >= INSPECT_X:
                    p["inspected"] = True
                    self._inspect(p)
                if p["defect"] != "none" and p["x"] >= DIVERT_X:
                    p["state"], p["u"] = "divert", 0.0
                    self.pusher_t = 1.0
                elif p["x"] >= END_X + 0.5:
                    p["state"], p["u"] = "drop", 0.0
                p["g"].pos(p["x"], 0, BELT_Z)
            elif st == "divert":
                p["u"] += pdt / 14.0
                u = min(p["u"], 1.0)
                p["g"].pos(DIVERT_X, -2.0 * u, BELT_Z - 0.55 * max(u - 0.25, 0) / 0.75)
                p["g"].rot(28 * min(u * 3, 1.0), 0, 0)
                if u >= 1.0:
                    self._free(p)
            elif st == "drop":
                p["u"] += pdt / 6.0
                u = min(p["u"], 1.0)
                p["g"].pos(END_X + 0.5 + 0.5 * u, 0, BELT_Z - 0.4 * u)
                if u >= 1.0:
                    self._free(p)

    def _free(self, p):
        p["state"] = "free"
        p["g"].show(False)
        p["g"].rot(0, 0, 0)

    def _inspect(self, p):
        ng = p["defect"] != "none"
        self.n_inspect += 1
        self.n_ng += int(ng)
        self.by_type[p["defect"]] += 1
        self.recent.append(int(ng))
        self.rate_hist.append(self.recent_rate)
        self.inspect_flash, self.inspect_ng = 6.0, ng
        self._write("inspection", dict(inspect_id=f"V{self.n_inspect:05d}",
                                       inspect_time=self.clock.strftime("%Y-%m-%d %H:%M:%S"),
                                       machine_id=p["m"].id, result="NG" if ng else "OK", defect_type=p["defect"]))
        self._send_packet(self.inspect_path, C_RED if ng else C_GREEN)
        self.set_label("inspect_res", f"{p['m'].id}  {'불량: ' + pm.DEFECT_KO[p['defect']] if ng else '양품'}")
        self.set_label("mes_val", f"검사 {self.n_inspect}   불량 {self.n_ng}   불량률 {self.defect_rate * 100:.1f}%")

    # ---- 도장 부스 ---------------------------------------------------------------
    def _step_paint(self, pdt):
        self.paint_wait -= pdt
        if self.paint_wait <= 0:
            self.paint_wait += PANEL_TAKT_S
            n = next((p for p in self.panels if p["state"] == "free"), None)
            if n is not None:
                n.update(state="run", x=PANEL_START_X, painted=False, done=False, rec=None)
                n["body"].color((0.55, 0.56, 0.58))
                n["sag"].show(False)
                n["pin"].show(False)
                n["g"].show(True)
        spraying = [False, False]
        for n in self.panels:
            if n["state"] != "run":
                continue
            n["x"] += PANEL_V * pdt
            for j, sx in enumerate(SPRAY_X):
                if abs(n["x"] - sx) < 0.55:
                    spraying[j] = True
            if not n["painted"] and n["x"] >= SPRAY_X[1]:
                n["painted"] = True
                n["rec"] = self.booth.next_panel()
                n["body"].color((0.96, 0.96, 0.94))
                n["sag"].show(n["rec"]["defect"] == "sag")
                n["pin"].show(n["rec"]["defect"] == "pinhole")
                r = n["rec"]
                self.set_label("booth_tag", f"온도 {r['booth_temp_c']:.1f}℃  습도 {r['booth_humidity_pct']:.0f}%"
                                            f"  점도 {r['viscosity_s']:.1f}s  도막 {r['film_thickness_um']:.0f}㎛")
            if not n["done"] and n["x"] >= 18.9:
                n["done"] = True
                r = n["rec"]
                ng = r["defect"] != "none"
                self.p_n += 1
                self.p_ng += int(ng)
                self.p_by_type[r["defect"]] += 1
                self.p_recent.append(int(ng))
                self._write("paint", dict(timestamp=self.clock.strftime("%Y-%m-%d %H:%M:%S"), booth_id="PB-01",
                                          lot_id=f"P{self.clock:%m%d}",
                                          **{key: r[key] for key in pm.PAINT_NOMINAL},
                                          defect_type=r["defect"], is_defect=int(ng)))
                self.set_label("booth_res", f"{'불량: ' + pm.DEFECT_KO[r['defect']] if ng else '양품'}"
                                            f"   ({self.p_ng}/{self.p_n})")
            n["g"].pos(n["x"], RAIL_Y, 3.15)
            if n["x"] >= PANEL_END_X:
                n["state"] = "free"
                n["g"].show(False)
        for j, s in enumerate(self.sprays):
            s.show(spraying[j])
        for key, (warn, bad) in (("booth_temp_c", (26.0, 28.0)), ("booth_humidity_pct", (62.0, 70.0))):
            lo, hi = pm.PAINT_RANGE[key]
            v = self.booth.setpoint[key]
            h = 0.1 + 1.9 * float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
            g = self.gauges[key]
            g.scale(0.2, 0.08, h)
            g.pos(g_x := (BOOTH_X0 - 0.9 + (0.45 if key == "booth_humidity_pct" else 0.0)), RAIL_Y - 1.63, 0.25 + h / 2)
            g.color(C_RED if v >= bad else (C_YELLOW if v >= warn else C_GREEN))

    # ---- 패킷·표시등 -------------------------------------------------------------
    def _send_packet(self, path, color, delay=0.0):
        k = next((p for p in self.packets if not p["on"]), None)
        if k is None:
            return
        k.update(on=True, path=path, d=-delay * 9.0)
        k["n"].color(color)

    def _step_packets(self, real_dt):
        for k in self.packets:
            if not k["on"]:
                continue
            k["d"] += 9.0 * real_dt
            if k["d"] >= k["path"].length:
                k["on"] = False
                k["n"].show(False)
            elif k["d"] >= 0:
                k["n"].pos(*k["path"].at(k["d"]))
                k["n"].show(True)

    def _step_indicators(self, real_dt, blink):
        e = self.errors
        for m in self.machines:
            m.flash = max(m.flash - real_dt, 0.0)
            bad_mold = e["missing"] and m.idx == 0
            m.s_mold.color((C_RED if blink else C_DARK) if bad_mold else C_GREEN)
            hot = C_MAGENTA if m.flash > 0 and blink else C_GREEN
            m.s_melt.color(hot)
            m.s_press.color(C_YELLOW if (e["unit"] and m.idx == 2) else hot)
            warn = bad_mold or (e["unit"] and m.idx == 2) or (e["clock"] and m.idx == 1) or m.flash > 0
            m.lamp.color(C_YELLOW if warn else C_GREEN)
        if self.inspect_flash > 0:
            self.inspect_flash -= real_dt * self.speed * (0 if self.paused else 1)
            self.lamp_ok.color(C_DARK if self.inspect_ng else C_GREEN)
            self.lamp_ng.color(C_RED if self.inspect_ng else C_DARK)
        else:
            self.lamp_ok.color(C_DARK)
            self.lamp_ng.color(C_DARK)
        self.pusher_t = max(self.pusher_t - real_dt * 0.8, 0.0)
        ext = 1.0 - abs(self.pusher_t * 2 - 1.0) if self.pusher_t > 0 else 0.0
        self.pusher.pos(DIVERT_X, 0.62 - 0.75 * ext, 0.95)
