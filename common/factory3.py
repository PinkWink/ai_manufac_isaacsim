"""3회차용 가상 공장: 2회차 공장 + 도장 외관 검사 셀 + 유압 유닛(예지보전)."""
from collections import deque

import numpy as np
import csv

from pxr import Gf, UsdGeom, UsdLux

from . import process_model as pm
from .factory import C_DARK, C_GREEN, C_RED, C_YELLOW, CAMERAS, VirtualFactory

# ---- 도장 외관 검사 셀 -------------------------------------------------------
CX, PY, PZ = 22.8, 1.0, 1.55                 # 시편(패널) 중심
CAM_Y = -0.75                                # 검사 카메라 위치 (패널 앞 1.75 m)
SAMPLE_KINDS = ("none", "pinhole", "sag", "dust", "tint")
SAMPLE_KO = dict(none="양품", pinhole="핀홀", sag="흐름", dust="이물", tint="색차")
LIGHT_NOMINAL = dict(angle=45.0, level=1.0, backdrop=True)
LIGHT_POWER = 9000.0                          # 기준 밝기에서의 조명 세기
C_PANEL, C_TINT = (0.45, 0.45, 0.44), (0.421, 0.457, 0.393)
# ---- 유압 유닛 ----------------------------------------------------------------
HX, HY = -12.6, 5.2
PUMP_TICK = 0.2                              # 진동을 한 번 재는 간격 (실제 시간, 초)

PUMP_COLS = ["run_hours", "sensor", "rms", "kurtosis", "crest", "peak", "current_a", "score", "alarm"]   # 유압 유닛 기록
CAMERAS.update({
    "overview": ((5.5, -26.0, 10.5), (5.5, 2.8, 2.2)),
    "paint_inspect": ((CX + 1.5, -5.4, 2.1), (CX, 0.6, 1.4)),
    "pump": ((-16.2, 1.6, 3.0), (HX, HY, 0.9)),
})


class Factory3(VirtualFactory):
    def __init__(self, stage, log_dir, seed=7):
        self.light = dict(LIGHT_NOMINAL)
        self.sample_kind, self.sample_seed = "none", 0
        self.pump = pm.PumpUnit(seed=11, life_h=1500.0, age_h=1080.0)
        self.pump_rate = 6.0                 # 유압 유닛 시계: 실제 1초에 흐르는 시간 (h)
        self.pump_sensor = "bearing"
        self.warn_th, self.stop_th, self.need_n = 5.0, 26.0, 3
        self.pump_alarm = 0                  # 0 정상, 1 경고, 2 정지
        self.pump_hist = deque([0.0] * 150, maxlen=150)
        self.pump_wave = np.zeros(pm.PUMP_N, dtype=np.float32)
        self.pump_feat = dict(rms=0.0, kurtosis=0.0, crest=0.0, peak=0.0, current=12.0, score=0.0)
        self.pump_over = [0, 0]              # 경고·정지 기준을 연속으로 넘은 횟수
        self.pump_events = deque(maxlen=6)   # 최근 알람 기록
        self._pump_acc, self._shock = 0.0, False
        self.health = {}
        for sensor in ("bearing", "base"):   # 정상 운전 구간으로 기준을 잡음
            u = pm.PumpUnit(seed=3, life_h=None)
            self.health[sensor] = pm.HealthScore().fit(pm.vib_features(np.stack([u.window(sensor) for _ in range(300)])))
        super().__init__(stage, log_dir, seed)

    def open_logs(self):
        super().open_logs()
        f = open(self.log_dir / f"pump_{self.log_stamp}.csv", "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=PUMP_COLS)
        w.writeheader()
        self._files["pump"] = (f, w)

    def _build(self):
        super()._build()
        # 검사 셀의 조명이 영상 밝기를 좌우하도록 공장 전체 조명을 낮춤
        if self.indoor:                      # 건물 안: 검사 셀 근처의 천장 조명을 끔
            for prim in self.stage.GetPrimAtPath("/World/Lights").GetChildren():
                if prim.GetName().startswith("Ceil") and UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()[0] > 7.0:
                    UsdLux.RectLight(prim).GetIntensityAttr().Set(0.0)
            print("hall lights:", len(self.k.hall_lights), [str(q.GetTypeName()) for q in self.k.hall_lights[:5]], flush=True)
            for q in self.k.hall_lights:        # 건물에 딸린 조명 중 검사 셀 가까이 있는 것도 끔
                x = UsdGeom.Xformable(q).ComputeLocalToWorldTransform(0).ExtractTranslation()[0]
                if x > 7.0:
                    UsdGeom.Imageable(q).GetVisibilityAttr().Set("invisible")
        else:
            UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/Lights/Dome")).GetIntensityAttr().Set(260.0)
        self._build_paint_inspect()
        self._build_pump()
        for key, text, pos in (("type1", "① 품질검사: 도장 외관 검사 셀", (CX - 7.5, PY, 6.4)),
                               ("type2", "② 이상탐지: 사출 조건 · 외관 검사기", (3.5, 0.0, 7.6)),
                               ("type3", "③ 예지보전: 유압 유닛", (HX + 5.5, HY, 5.4)),
                               ("type4", "④ 공정 최적화: 성형 조건 추천", (-6.0, 4.2, 9.2))):
            self.label(key, text, pos, 22, (1.0, 0.85, 0.3))
            self.labels[key]["group"] = "types"

    # ======================================================================
    # 도장 외관 검사 셀
    # ======================================================================
    def _build_paint_inspect(self):
        k, p = self.k, "/World/PaintInspect"
        k.box(p + "/Base", (2.6, 3.4, 0.06), (CX, 0.3, 0.03), mat=k.mat((0.3, 0.32, 0.34), 0.7))
        for j, dx in enumerate((-0.62, 0.62)):
            k.box(p + f"/JigPost{j}", (0.06, 0.06, 2.3), (CX + dx, PY + 0.06, 1.15), mat=self.m_frame)
        k.box(p + "/JigBar", (1.3, 0.06, 0.06), (CX, PY + 0.06, 2.27), mat=self.m_frame)
        self.backdrop = k.box(p + "/Backdrop", (2.3, 0.04, 2.3), (CX, PY + 0.4, 1.45), color=(0.22, 0.23, 0.25))
        # 배경판을 치웠을 때 뒤에 보이는 것들 (지나가는 대차 · 자재)
        k.box(p + "/Clutter0", (0.7, 0.5, 1.1), (CX - 0.55, PY + 1.6, 0.9), color=(0.85, 0.55, 0.1))
        k.box(p + "/Clutter1", (0.5, 0.5, 1.9), (CX + 0.6, PY + 1.9, 1.2), color=(0.15, 0.4, 0.75))
        k.box(p + "/Clutter2", (2.6, 0.1, 0.1), (CX, PY + 1.3, 2.05), color=(0.9, 0.9, 0.2))
        # 덮개: 공장 조명이 시편에 직접 닿지 않게 가림 (검사 셀 조명만으로 촬영)
        k.box(p + "/Hood", (3.4, 3.6, 0.05), (CX, 0.0, 2.95), mat=self.m_cab)
        # 암막: 주변 조명이 시편에 닿지 않도록 뒤 · 옆 · 앞 위쪽을 가림 (실제 검사 셀도 외광을 막음)
        blk = k.mat((0.03, 0.03, 0.035), rough=0.9, name="Blackout")
        k.box(p + "/HoodBack", (3.4, 1.5, 0.05), (CX, 2.55, 2.95), mat=self.m_cab)
        k.box(p + "/CurtainBack", (3.4, 0.03, 2.9), (CX, 3.3, 1.475), mat=blk)
        for j, dx in enumerate((-1.7, 1.7)):
            k.box(p + f"/CurtainSide{j}", (0.03, 3.9, 2.9), (CX + dx, 1.35, 1.475), mat=blk)
            k.box(p + f"/CurtainSideTop{j}", (0.03, 1.2, 0.75), (CX + dx, -1.2, 2.575), mat=blk)
        k.box(p + "/CurtainFront", (3.4, 0.03, 0.75), (CX, -1.8, 2.575), mat=blk)
        for j, (dx, dy) in enumerate(((-1.65, -1.75), (1.65, -1.75), (-1.65, 1.75), (1.65, 1.75))):
            k.box(p + f"/HoodPost{j}", (0.07, 0.07, 2.95), (CX + dx, dy, 1.475), mat=self.m_steel)
        k.box(p + "/CamPost", (0.07, 0.07, 1.5), (CX, CAM_Y - 0.12, 0.75), mat=self.m_steel)
        k.box(p + "/CamBody", (0.16, 0.26, 0.16), (CX, CAM_Y - 0.05, PZ), mat=self.m_dark)
        cam = UsdGeom.Camera.Define(self.stage, p + "/Camera")
        cam.CreateFocalLengthAttr(17.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        xf = UsdGeom.Xformable(cam)
        xf.AddTranslateOp().Set(Gf.Vec3d(CX, CAM_Y + 0.1, PZ))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))            # 아래(-Z)를 보는 카메라를 +Y 를 보게 세움
        # 시편 (도장 패널) 과 결함 표시
        self.sample = k.group(p + "/Sample", (CX, PY, PZ))
        self.sample_body = k.box(p + "/Sample/Body", (0.8, 0.02, 0.8), color=C_PANEL)
        self.s_pin = [k.cyl(p + f"/Sample/Pin{j}", 0.008, 0.004, (0, -0.012, 0), axis="Y", color=(0.1, 0.1, 0.11)) for j in range(8)]
        self.s_sag = [(k.box(p + f"/Sample/Sag{j}", (0.03, 0.006, 0.2), (0, -0.012, 0), color=(0.385, 0.38, 0.335)),
                       k.sphere(p + f"/Sample/SagTip{j}", 0.02, (0, -0.012, 0), color=(0.385, 0.38, 0.335), scale=(1, 0.3, 1.4)))
                      for j in range(3)]
        self.s_dust = [k.sphere(p + f"/Sample/Dust{j}", 0.012, (0, -0.012, 0), color=(0.27, 0.255, 0.23), scale=(1, 0.5, 1))
                       for j in range(2)]
        # 조명 두 개 (각도 · 밝기를 바꿀 수 있음)
        self.lights = []
        for j in range(2):
            lt = UsdLux.RectLight.Define(self.stage, p + f"/Light{j}")
            lt.CreateWidthAttr(0.25)
            lt.CreateHeightAttr(0.9)
            lt.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
            op = UsdGeom.Xformable(lt).AddTransformOp()
            bar = k.box(p + f"/LightBar{j}", (0.3, 0.08, 0.95), (0, 0, 0), mat=k.mat((0.9, 0.9, 0.9), emissive=(3, 3, 2.8), name="InspBar"))
            stand = k.box(p + f"/LightStand{j}", (0.05, 0.05, 1.1), (0, 0, 0), mat=self.m_steel)
            self.lights.append((lt, op, bar, stand))
        self.label("pinspect", "도장 외관 검사 셀", (CX, PY, 3.1), 20, (1, 1, 1))
        self.label("pinspect_res", "", (CX, PY, 2.72), 16, (0.7, 1.0, 0.7), detail=True)
        self.set_sample("none", 0, jitter=False)
        self.apply_lighting()

    def apply_lighting(self, flicker=0.0):
        """조명 각도(패널 법선 기준) · 밝기 · 배경판을 장면에 반영."""
        a, R = np.radians(self.light["angle"]), 1.35
        level = self.light["level"] * (1.0 + flicker)
        target = Gf.Vec3d(CX, PY, PZ)
        for j, (lt, op, bar, stand) in enumerate(self.lights):
            s = -1.0 if j == 0 else 1.0
            eye = Gf.Vec3d(CX + s * R * np.sin(a), PY - R * np.cos(a), PZ + 0.1)
            op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
            lt.GetIntensityAttr().Set(float(LIGHT_POWER * level)) if lt.GetIntensityAttr() else lt.CreateIntensityAttr(float(LIGHT_POWER * level))
            back = eye + (eye - target).GetNormalized() * 0.07
            bar.pos(back[0], back[1], back[2])
            bar.rot(0, 0, float(np.degrees(np.arctan2(target[1] - eye[1], target[0] - eye[0])) - 90.0))
            stand.pos(back[0], back[1], 0.55)
        self.backdrop.show(self.light["backdrop"])

    def set_sample(self, kind, seed=None, jitter=True):
        """시편 하나를 검사 위치에 놓음. 같은 seed 면 같은 시편."""
        self.sample_kind = kind
        self.sample_seed = seed = int(self.rng.integers(1 << 30)) if seed is None else seed
        r = np.random.default_rng(seed)
        dx, dz, tilt = (r.normal(0, 0.0025), r.normal(0, 0.0025), r.normal(0, 0.25)) if jitter else (0, 0, 0)
        self.sample.pos(CX + dx, PY, PZ + dz)
        self.sample.rot(0, tilt, 0)
        self.sample_body.color(C_TINT if kind == "tint" else C_PANEL)
        n_pin = int(r.integers(3, 8)) if kind == "pinhole" else 0
        for j, n in enumerate(self.s_pin):
            n.show(j < n_pin)
            if j < n_pin:
                rad = r.uniform(0.008, 0.014)
                n.pos(r.uniform(-0.33, 0.33), -0.012, r.uniform(-0.33, 0.33))
                n.scale(rad, 0.004, rad)
        n_sag = int(r.integers(1, 4)) if kind == "sag" else 0
        for j, (bar, tip) in enumerate(self.s_sag):
            bar.show(j < n_sag); tip.show(j < n_sag)
            if j < n_sag:
                x, h, w = r.uniform(-0.3, 0.3), r.uniform(0.12, 0.28), r.uniform(0.024, 0.042)
                top = r.uniform(-0.05, 0.33)
                bar.scale(w, 0.006, h); bar.pos(x, -0.012, top - h / 2)
                tip.scale(w * 0.8, 0.006, w * 1.1); tip.pos(x, -0.012, top - h)
        n_dust = int(r.integers(1, 3)) if kind == "dust" else 0
        for j, n in enumerate(self.s_dust):
            n.show(j < n_dust)
            if j < n_dust:
                rad = r.uniform(0.012, 0.02)
                n.pos(r.uniform(-0.33, 0.33), -0.012, r.uniform(-0.33, 0.33))
                n.scale(rad, rad * 0.5, rad * r.uniform(0.7, 1.4))

    # ======================================================================
    # 유압 유닛 (예지보전)
    # ======================================================================
    def _build_pump(self):
        k, p = self.k, "/World/Hydraulic"
        k.box(p + "/Tank", (1.5, 1.0, 0.8), (HX, HY, 0.4), mat=k.mat((0.2, 0.32, 0.5), 0.5))
        k.box(p + "/BasePlate", (1.4, 0.7, 0.06), (HX, HY, 0.83), mat=self.m_steel)
        self.motor = k.group(p + "/Motor", (HX, HY, 1.12))
        k.cyl(p + "/Motor/Body", 0.24, 0.62, (-0.22, 0, 0), axis="X", mat=k.mat((0.25, 0.45, 0.3), 0.4))
        for j in range(5):
            k.cyl(p + f"/Motor/Fin{j}", 0.265, 0.03, (-0.44 + j * 0.11, 0, 0), axis="X", mat=k.mat((0.2, 0.38, 0.25), 0.4))
        k.cyl(p + "/Motor/Coupling", 0.09, 0.16, (0.17, 0, 0), axis="X", mat=self.m_steel)
        k.box(p + "/Motor/Pump", (0.34, 0.36, 0.36), (0.42, 0, 0), mat=k.mat((0.55, 0.3, 0.12), 0.4))
        k.cyl(p + "/Motor/BearingHousing", 0.14, 0.1, (0.07, 0, 0), axis="X", mat=self.m_steel)
        self.s_bearing = k.sphere(p + "/Motor/SensorBearing", 0.06, (0.07, 0, 0.17), color=C_GREEN)
        self.s_base = k.sphere(p + "/SensorBase", 0.06, (HX - 0.55, HY - 0.3, 0.9), color=C_DARK)
        k.cyl(p + "/Pipe", 0.04, 2.0, (HX + 1.35, HY, 1.0), axis="X", mat=self.m_steel)
        k.cyl(p + "/LampPole", 0.03, 0.9, (HX + 0.6, HY + 0.42, 1.3), mat=self.m_steel)
        self.pump_lamp = k.cyl(p + "/Lamp", 0.08, 0.26, (HX + 0.6, HY + 0.42, 1.88), color=C_GREEN)
        self.label("pump", "유압 유닛 HP-01", (HX, HY, 2.75), 20, (1, 1, 1))
        self.label("pump_tag", "", (HX, HY, 2.35), 15, (0.6, 0.95, 1.0), detail=True)
        self.label("pump_s1", "베어링 하우징", (HX + 0.07, HY - 0.25, 1.5), 13, (0.7, 1.0, 0.7), detail=True)
        self.label("pump_s2", "베이스 프레임", (HX - 0.55, HY - 0.55, 1.0), 13, (0.8, 0.8, 0.8), detail=True)

    def pump_reset(self, age_h):
        """베어링 교체(0) 또는 수명 말기로 이동."""
        self.pump = pm.PumpUnit(seed=int(self.rng.integers(1 << 30)), life_h=1500.0, age_h=age_h)
        self.pump_alarm, self.pump_over = 0, [0, 0]
        self.pump_hist.extend([0.0] * self.pump_hist.maxlen)
        self.machines[0].stopped = False

    def pump_shock(self):
        self._shock = True

    def _step_pump(self, real_dt, blink):
        if not self.paused:
            self._pump_acc += real_dt
        while self._pump_acc >= PUMP_TICK:
            self._pump_acc -= PUMP_TICK
            if self.pump_alarm < 2 and not self.pump.failed:
                self.pump.age_h += self.pump_rate * PUMP_TICK
            w = self.pump.window(self.pump_sensor, shock=self._shock)
            self._shock = False
            f = pm.vib_features(w)
            score = float(self.health[self.pump_sensor].score(f))
            self.pump_wave = w
            self.pump_feat = dict({key: float(f[key]) for key in pm.PUMP_FEATURES}, current=self.pump.current_rms(), score=score)
            self.pump_hist.append(score)
            for i, th in enumerate((self.warn_th, self.stop_th)):
                self.pump_over[i] = self.pump_over[i] + 1 if score >= th else 0
            level = 2 if self.pump_over[1] >= self.need_n else (1 if self.pump_over[0] >= self.need_n else 0)
            self._write("pump", dict(run_hours=round(self.pump.age_h, 1), sensor=self.pump_sensor, rms=round(float(f["rms"]), 4),
                                     kurtosis=round(float(f["kurtosis"]), 3), crest=round(float(f["crest"]), 3), peak=round(float(f["peak"]), 3),
                                     current_a=round(self.pump_feat["current"], 2), score=round(score, 2), alarm=max(level, self.pump_alarm)))
            if level > self.pump_alarm:
                self.pump_alarm = level
                left = self.pump.hours_to_failure
                self.pump_events.append(f"{'경고' if level == 1 else '정지'}  점수 {score:.1f}  (고장까지 {left:.0f}시간)")
                if level == 2:
                    self.machines[0].stopped = True          # 정지 알람 → 사출기 IMM-01 을 세움
            elif level == 0 and self.pump_alarm == 1:
                self.pump_alarm = 0
            if self.pump.failed and self.pump_alarm < 2:
                self.pump_alarm = 2
                self.machines[0].stopped = True
                self.pump_events.append("고장 발생 (알람 없이 멈춤)")
            left = self.pump.hours_to_failure
            self.set_label("pump_tag", f"실효값 {f['rms']:.2f}g  첨도 {f['kurtosis']:.1f}  전류 {self.pump_feat['current']:.1f}A"
                                       f"  이상 점수 {score:.1f}")
        sev = self.pump.severity()
        if self.pump_alarm < 2 and not self.paused:
            amp = 0.0015 + 0.012 * min(sev, 1.0)
            self.motor.pos(HX + float(self.rng.normal(0, amp)), HY + float(self.rng.normal(0, amp)), 1.12 + float(self.rng.normal(0, amp)))
        self.pump_lamp.color((C_RED if blink else C_DARK) if self.pump_alarm == 2 else (C_YELLOW if self.pump_alarm == 1 else C_GREEN))
        self.s_bearing.color(C_GREEN if self.pump_sensor == "bearing" else C_DARK)
        self.s_base.color(C_GREEN if self.pump_sensor == "base" else C_DARK)

    def step(self, real_dt):
        super().step(real_dt)
        self._step_pump(real_dt, int(self.wall * 3) % 2 == 0)
