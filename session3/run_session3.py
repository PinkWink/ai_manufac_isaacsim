"""3회차 (AI 품질검사·이상탐지·예지보전·공정 최적화) 가상 공장 실행 파일.

실행:  ./scripts/factory session3
옵션:  --capture <폴더>   교재용 화면 촬영        --record <폴더>   교재용 영상 프레임 촬영
       --dataset <폴더>   도장 외관 검사 이미지 촬영 (실습 데이터 제작용)
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
for name in ("--capture", "--record", "--dataset", "--look"):
    parser.add_argument(name, default=None)
parser.add_argument("--clips", default="all")
parser.add_argument("--no-web", action="store_true", help="수강생 연결(웹 조작 · 데이터 받기)을 끔")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--web-allow", action="store_true", help="처음부터 조별 조작을 허용한 상태로 시작")
args, _ = parser.parse_known_args()

from common import app_kit  # noqa: E402
app, stage = app_kit.start()

import numpy as np  # noqa: E402
import omni.ui as ui  # noqa: E402

from common import process_model as pm  # noqa: E402
from common.anomaly import PixelModel, prepare  # noqa: E402
from common.factory import CAMERAS  # noqa: E402
from common.factory3 import LIGHT_NOMINAL, SAMPLE_KINDS, SAMPLE_KO, Factory3  # noqa: E402

DATA = ROOT / "data" / "session3"
factory = Factory3(stage, DATA / "live")
factory.open_logs()

WHITE, GRAY, CYAN, RED, GREEN, YELLOW = 0xFFFFFFFF, 0xFFAAAAAA, 0xFFFFDD66, 0xFF5555FF, 0xFF66DD66, 0xFF33CCFF
IMG = 192                                       # 검사 카메라 이미지 크기
import carb  # noqa: E402
carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)      # 자동 노출을 끔 → 조명 밝기가 영상에 그대로 반영
INJ_KO = dict(mold_temp_c="금형온도 (℃)", melt_temp_c="수지온도 (℃)", inj_pressure_mpa="사출압 (MPa)",
              hold_pressure_mpa="보압 (MPa)", hold_time_s="보압 시간 (s)", cooling_time_s="냉각 시간 (s)")


def first_existing(*paths):
    return next((p for p in paths if p.exists()), None)


# ============================================================================
# 검사 카메라 영상과 실시간 판정
# ============================================================================
class Judge:
    """검사 카메라 창(뷰포트)의 영상을 읽어 이상 점수를 매김."""

    def __init__(self):
        from omni.kit.viewport.utility import create_viewport_window
        self.vp = create_viewport_window("검사 카메라", width=430, height=470, camera_path="/World/PaintInspect/Camera")
        self.vp.viewport_api.resolution = (IMG * 2, IMG * 2)
        self.image, self.n_image = None, 0
        self.model, self.model_name = None, "없음"
        self.threshold = 2.5
        self.auto, self.wait, self.seed = False, 0, 1000
        self.truth = None
        self.reset()
        self.load_model()

    def request(self):
        """검사 카메라 영상 한 장을 요청함. 몇 프레임 뒤 self.image 에 들어오고 self.n_image 가 1 늘어남."""
        from omni.kit.viewport.utility import capture_viewport_to_buffer

        def done(buffer, size, width, height, fmt):
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_byte * size)
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            raw = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
            a = np.frombuffer(raw.contents, dtype=np.uint8).reshape(height, width, 4)[..., :3]
            f = min(height, width) // IMG                     # 가운데 정사각형을 잘라 IMG 크기로 줄임
            y0, x0 = (height - IMG * f) // 2, (width - IMG * f) // 2
            self.image = a[y0:y0 + IMG * f, x0:x0 + IMG * f].reshape(IMG, f, IMG, f, 3).mean(axis=(1, 3)).astype(np.uint8)
            self.n_image += 1
        capture_viewport_to_buffer(self.vp.viewport_api, done)

    def grab(self):
        """새 영상을 받아 올 때까지 화면을 돌림 (데이터 촬영용)."""
        n = self.n_image
        self.request()
        for _ in range(60):
            app.update()
            if self.n_image > n:
                return self.image
        return None

    def reset(self):
        self.count = dict(ok=0, over=0, miss=0, hit=0)       # 양품 통과, 과검, 미검, 불량 검출
        self.last = "-"

    def load_model(self):
        path = first_existing(DATA / "work" / "pixel_model.npz", DATA / "pixel_model_provided.npz")
        if path is not None:
            self.model, self.model_name = PixelModel.load(path), ("내 모델" if "work" in str(path) else "제공 모델")

    def score_now(self):
        img = self.image
        if img is None or self.model is None:
            return img, None, None
        x = prepare(img)
        return img, self.model.anomaly_map(x), self.model.score(x)

    def step(self):
        """자동 판정: 시편을 바꿔 놓고 몇 프레임 뒤에 찍어 판정."""
        if not self.auto or factory.paused:
            return None
        self.wait -= 1
        if self.wait == 20:
            self.request()                                    # 영상 요청 → 몇 프레임 뒤 도착
        if self.wait == 14:                                   # 판정
            img, amap, score = self.score_now()
            if score is not None:
                ng = score >= self.threshold
                bad = self.truth != "none"
                key = ("hit" if ng else "miss") if bad else ("over" if ng else "ok")
                self.count[key] += 1
                self.last = f"{SAMPLE_KO[self.truth]} → {'불량' if ng else '양품'} 판정 (점수 {score:.1f})" + \
                            ("  ← 과검" if key == "over" else "  ← 미검" if key == "miss" else "")
                factory.set_label("pinspect_res", f"이상 점수 {score:.1f}  →  {'불량' if ng else '양품'}")
                return img, amap, score
        if self.wait <= 0:                                    # 다음 시편
            self.seed += 1
            r = np.random.default_rng(self.seed)
            self.truth = "none" if r.random() < 0.55 else SAMPLE_KINDS[1 + int(r.integers(4))]
            factory.set_sample(self.truth, self.seed)
            factory.apply_lighting(flicker=float(r.normal(0, 0.02)))
            self.wait = 50
        return None


def heat_bytes(img, amap, th):
    """카메라 영상 위에 이상 지도를 빨갛게 겹친 RGBA 바이트."""
    a = img.astype(np.float32)
    if amap is not None:
        m = np.kron(amap, np.ones((IMG // amap.shape[0], IMG // amap.shape[1]), dtype=np.float32))
        w = np.clip((m - th * 0.6) / (th * 0.8), 0, 1)[..., None]
        a = a * (1 - 0.75 * w) + np.array([255, 30, 30], dtype=np.float32) * 0.75 * w
    out = np.concatenate([a.astype(np.uint8), np.full((IMG, IMG, 1), 255, dtype=np.uint8)], axis=-1)
    return out.tobytes()


# ============================================================================
# 진행 패널
# ============================================================================
class Panel:
    PERIODS = {1: ("overview", ("run",)), 2: ("paint_inspect", ("run", "shoot")),
               3: ("paint_inspect", ("run", "judge")), 4: ("pump", ("run", "pump"))}

    def __init__(self):
        self.sec, self.sliders = {}, {}
        self.rate_dirty = True
        self.inspect_vp = None
        self.win = ui.Window("3회차 진행 패널", width=470, height=1000)
        with self.win.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    self._sec_run(); self._sec_shoot(); self._sec_judge(); self._sec_pump(); self._sec_reco()
                    web.build_section(ui)

    def section(self, key, title, collapsed=False):
        self.sec[key] = ui.CollapsableFrame(title, height=0, collapsed=collapsed)
        return self.sec[key]

    def period(self, n):
        view, opened = self.PERIODS[n]
        for name, frame in self.sec.items():
            frame.collapsed = name not in opened
        self.types_cb.model.set_value(n == 1)
        judge.auto = (n == 3)
        judge.vp.visible = n in (2, 3)                       # 검사 카메라 창은 도장 검사 교시에만 띄움
        self.auto_cb.model.set_value(n == 3)
        set_view(view)

    # ---- 화면과 진행 ------------------------------------------------------------
    def _sec_run(self):
        with self.section("run", "화면과 진행"):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=34, spacing=4):
                    for n in (1, 2, 3, 4):
                        ui.Button(f"{n}교시", clicked_fn=lambda n=n: self.period(n), style={"background_color": 0xFF7A4A1E})
                names = [("overview", "공장 전체"), ("paint", "도장 부스"), ("paint_inspect", "도장 검사 셀"),
                         ("pump", "유압 유닛"), ("injection", "사출기"), ("inspection", "외관 검사기")]
                for row in (names[:3], names[3:]):
                    with ui.HStack(height=30, spacing=4):
                        for key, text in row:
                            ui.Button(text, clicked_fn=lambda k=key: set_view(k))
                with ui.HStack(height=24, spacing=6):
                    ui.Label("배속", width=40)
                    s = ui.FloatSlider(min=1, max=40, step=1)
                    s.model.set_value(factory.speed)
                    s.model.add_value_changed_fn(lambda m: setattr(factory, "speed", m.get_value_as_float()))
                with ui.HStack(height=30, spacing=4):
                    self.pause_btn = ui.Button("일시 정지", clicked_fn=self._toggle_pause)
                    ui.Button("검사 카메라 창", clicked_fn=self._toggle_inspect_view)
                with ui.HStack(height=24, spacing=8):
                    self.types_cb = ui.CheckBox(width=22)
                    self.types_cb.model.add_value_changed_fn(lambda m: show_types(m.get_value_as_bool()))
                    ui.Label("AI 적용 네 유형의 위치 표시")

    def _toggle_pause(self):
        factory.paused = not factory.paused
        self.pause_btn.text = "다시 진행" if factory.paused else "일시 정지"

    def _toggle_inspect_view(self):
        judge.vp.visible = not judge.vp.visible

    # ---- 촬영 조건 (2교시) -------------------------------------------------------
    def _sec_shoot(self):
        with self.section("shoot", "도장 외관 검사: 촬영 조건", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                for key, text, lo, hi in (("angle", "조명 각도 (도)", 10.0, 80.0), ("level", "조명 밝기 (배)", 0.3, 2.0)):
                    with ui.HStack(height=24, spacing=6):
                        ui.Label(text, width=120)
                        s = ui.FloatSlider(min=lo, max=hi, step=1.0 if key == "angle" else 0.05)
                        s.model.set_value(LIGHT_NOMINAL[key])
                        s.model.add_value_changed_fn(lambda m, k=key: self._on_light(k, m.get_value_as_float()))
                        self.sliders[key] = s
                with ui.HStack(height=24, spacing=8):
                    self.back_cb = ui.CheckBox(width=22)
                    self.back_cb.model.set_value(True)
                    self.back_cb.model.add_value_changed_fn(lambda m: self._on_light("backdrop", m.get_value_as_bool()))
                    ui.Label("배경판 사용 (배경 고정)")
                ui.Button("기준 촬영 조건으로", height=28, clicked_fn=self.light_nominal)
                ui.Label("검사 위치에 놓을 시편", height=20, style={"color": GRAY})
                with ui.HStack(height=28, spacing=4):
                    for kind in SAMPLE_KINDS:
                        ui.Button(SAMPLE_KO[kind], clicked_fn=lambda k=kind: self.put_sample(k))

    def _on_light(self, key, value):
        factory.light[key] = value
        factory.apply_lighting()

    def light_nominal(self):
        self.sliders["angle"].model.set_value(LIGHT_NOMINAL["angle"])
        self.sliders["level"].model.set_value(LIGHT_NOMINAL["level"])
        self.back_cb.model.set_value(True)

    def put_sample(self, kind):
        judge.seed += 1
        factory.set_sample(kind, judge.seed)
        self.show_wait = 12                                   # 몇 프레임 뒤에 영상·점수 갱신

    # ---- 실시간 판정 (3교시) -----------------------------------------------------
    def _sec_judge(self):
        with self.section("judge", "외관 이상탐지: 임계값과 과검·미검", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                with ui.HStack(height=IMG + 4, spacing=8):
                    self.provider = ui.ByteImageProvider()
                    ui.ImageWithProvider(self.provider, width=IMG, height=IMG)
                    with ui.VStack(spacing=4):
                        self.score_lb = ui.Label("", height=26, style={"font_size": 20})
                        self.verdict_lb = ui.Label("", height=26, style={"font_size": 20})
                        self.model_lb = ui.Label("", height=20, style={"color": GRAY})
                        ui.Label("빨간 부분: 정상과 다른 곳", height=20, style={"color": GRAY})
                with ui.HStack(height=24, spacing=6):
                    ui.Label("임계값", width=120)
                    s = ui.FloatSlider(min=0.5, max=12.0, step=0.1)
                    s.model.set_value(judge.threshold)
                    s.model.add_value_changed_fn(lambda m: setattr(judge, "threshold", m.get_value_as_float()))
                    self.sliders["threshold"] = s
                with ui.HStack(height=24, spacing=8):
                    self.auto_cb = ui.CheckBox(width=22)
                    self.auto_cb.model.add_value_changed_fn(lambda m: setattr(judge, "auto", m.get_value_as_bool()))
                    ui.Label("시편을 계속 흘려보내며 자동 판정")
                self.count_lb = ui.Label("", height=24, style={"font_size": 17})
                self.rate_lb = ui.Label("", height=22, style={"color": CYAN})
                self.last_lb = ui.Label("", height=22)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("집계 초기화", clicked_fn=judge.reset)
                    ui.Button("모델 다시 읽기", clicked_fn=judge.load_model)
        self.show_wait = 12

    def show_image(self, img, amap, score):
        if img is None:
            return
        self.provider.set_bytes_data(list(heat_bytes(img, amap, judge.threshold)), [IMG, IMG])
        if score is not None:
            ng = score >= judge.threshold
            self.score_lb.text = f"이상 점수  {score:.1f}"
            self.verdict_lb.text = "판정  불량" if ng else "판정  양품"
            self.verdict_lb.style = {"font_size": 20, "color": RED if ng else GREEN}

    # ---- 예지보전 (4교시) --------------------------------------------------------
    def _sec_pump(self):
        with self.section("pump", "예지보전: 유압 유닛 HP-01", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                ui.Label("진동 파형 (가속도, ±4 g)", height=18, style={"color": GRAY})
                self.wave_plot = ui.Plot(ui.Type.LINE, -4.0, 4.0, *([0.0] * 256), height=70,
                                         style={"color": 0xFF66DD66, "background_color": 0xFF202020})
                self.feat_lb = ui.Label("", height=22)
                ui.Label("이상 점수 추이 (0 ~ 40)", height=18, style={"color": GRAY})
                self.score_plot = ui.Plot(ui.Type.LINE, 0.0, 40.0, *factory.pump_hist, height=70,
                                          style={"color": 0xFF44CCFF, "background_color": 0xFF202020})
                self.alarm_lb = ui.Label("", height=26, style={"font_size": 19})
                self.life_lb = ui.Label("", height=20, style={"color": GRAY})
                for key, text, lo, hi in (("warn_th", "경고 기준", 1.0, 20.0), ("stop_th", "정지 기준", 5.0, 40.0),
                                          ("pump_rate", "시간 흐름 (h/초)", 0.0, 24.0)):
                    with ui.HStack(height=24, spacing=6):
                        ui.Label(text, width=120)
                        s = ui.FloatSlider(min=lo, max=hi, step=0.5)
                        s.model.set_value(getattr(factory, key))
                        s.model.add_value_changed_fn(lambda m, k=key: setattr(factory, k, m.get_value_as_float()))
                        self.sliders[key] = s
                with ui.HStack(height=24, spacing=8):
                    cb = ui.CheckBox(width=22)
                    cb.model.set_value(True)
                    cb.model.add_value_changed_fn(lambda m: setattr(factory, "need_n", 3 if m.get_value_as_bool() else 1))
                    self.need_cb = cb
                    ui.Label("3회 연속 넘을 때만 알람 (오경보 억제)")
                with ui.HStack(height=24, spacing=8):
                    cb = ui.CheckBox(width=22)
                    cb.model.add_value_changed_fn(lambda m: setattr(factory, "pump_sensor", "base" if m.get_value_as_bool() else "bearing"))
                    self.base_cb = cb
                    ui.Label("센서를 베이스 프레임으로 옮김")
                with ui.HStack(height=28, spacing=4):
                    ui.Button("충격 주입 (외란)", clicked_fn=factory.pump_shock)
                    ui.Button("수명 말기로", clicked_fn=lambda: factory.pump_reset(1150.0))
                    ui.Button("베어링 교체", clicked_fn=lambda: factory.pump_reset(0.0))
                self.event_lbs = [ui.Label("", height=19, style={"font_size": 14}) for _ in range(4)]

    # ---- 조건 추천 (공정 최적화) --------------------------------------------------
    def _sec_reco(self):
        with self.section("reco", "공정 최적화: 성형 조건 추천", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                for key, text in INJ_KO.items():
                    with ui.HStack(height=24, spacing=6):
                        ui.Label(text, width=120)
                        lo, hi = pm.INJ_RANGE[key]
                        s = ui.FloatSlider(min=lo, max=hi, step=0.5)
                        s.model.set_value(pm.INJ_NOMINAL[key])
                        s.model.add_value_changed_fn(lambda m, k=key: self._on_setpoint(k, m.get_value_as_float()))
                        self.sliders[key] = s
                self.exp_lb = ui.Label("", height=24, style={"font_size": 18, "color": CYAN})
                self.stat_lb = ui.Label("", height=22)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("현장 변동 (금형온도 · 보압 저하)", clicked_fn=lambda: self.set_cond(dict(mold_temp_c=54.0, hold_pressure_mpa=50.0)))
                    ui.Button("추천 조건 적용", clicked_fn=self.apply_reco)
                self.reco_lb = ui.Label("", height=58, word_wrap=True, style={"color": GRAY})

    def _on_setpoint(self, key, value):
        factory.set_setpoint(key, value)
        self.rate_dirty = True

    def set_cond(self, change):
        for key in INJ_KO:
            self.sliders[key].model.set_value(change.get(key, pm.INJ_NOMINAL[key]))
        factory.reset_stats(new_files=False)

    def apply_reco(self):
        path = first_existing(DATA / "work" / "recommend.json", DATA / "recommend_provided.json")
        if path is None:
            self.reco_lb.text = "추천 파일 없음: 노트북 3-0 을 먼저 실행"
            return
        reco = json.loads(path.read_text())
        for key in INJ_KO:
            if key in reco:
                self.sliders[key].model.set_value(float(reco[key]["recommend"]))
        factory.reset_stats(new_files=False)
        self.reco_lb.text = ("내 추천" if "work" in str(path) else "제공 추천") + ":  " + "   ".join(
            f"{INJ_KO[k].split(' (')[0]} {v['low']:.0f}~{v['high']:.0f}" for k, v in reco.items() if k in INJ_KO)

    # ---- 주기 갱신 ---------------------------------------------------------------
    def refresh(self):
        f = factory
        c = judge.count
        n_ok, n_bad = c["ok"] + c["over"], c["miss"] + c["hit"]
        self.count_lb.text = f"양품 통과 {c['ok']}   과검 {c['over']}   미검 {c['miss']}   불량 검출 {c['hit']}"
        self.rate_lb.text = (f"과검률 {c['over'] / n_ok * 100:.0f}%" if n_ok else "과검률 -") + "    " + \
                            (f"미검률 {c['miss'] / n_bad * 100:.0f}%" if n_bad else "미검률 -")
        self.last_lb.text = judge.last
        self.model_lb.text = f"모델: {judge.model_name}"
        w = f.pump_wave
        self.wave_plot.set_data(*[float(v) for v in w[:512:2]])
        self.score_plot.set_data(*f.pump_hist)
        p = f.pump_feat
        self.feat_lb.text = f"실효값 {p['rms']:.2f} g   첨도 {p['kurtosis']:.1f}   파고율 {p['crest']:.1f}   전류 {p['current']:.1f} A"
        name, color = (("정상", GREEN), ("경고: 점검 계획", YELLOW), ("정지: 설비 세움", RED))[f.pump_alarm]
        self.alarm_lb.text = f"이상 점수 {p['score']:.1f}    알람  {name}"
        self.alarm_lb.style = {"font_size": 19, "color": color}
        left = f.pump.hours_to_failure
        self.life_lb.text = f"운전 {f.pump.age_h:.0f}시간   (실제 고장까지 {max(left, 0):.0f}시간 · 현장에서는 알 수 없는 값)"
        ev = list(f.pump_events)[-4:]
        for i, lb in enumerate(self.event_lbs):
            lb.text = ev[i] if i < len(ev) else ""
        self.stat_lb.text = f"외관 검사  {f.n_inspect}개 중 불량 {f.n_ng}개   ({f.defect_rate * 100:.1f}%)"
        if self.rate_dirty:
            self.rate_dirty = False
            self.exp_lb.text = f"이 조건의 예상 불량률  {f.expected_rate() * 100:.1f}%"
        if self.show_wait > 0:                                # 시편을 바꾼 뒤 영상·점수 갱신
            self.show_wait -= 1
            if self.show_wait == 2 and not judge.auto:
                judge.request()
            if self.show_wait == 0:
                img, amap, score = judge.score_now()
                self.show_image(img, amap, score)
                if score is not None:
                    factory.set_label("pinspect_res", f"이상 점수 {score:.1f}  →  {'불량' if score >= judge.threshold else '양품'}")
                if not judge.auto:
                    self.show_wait = 6                        # 조명을 바꾸면 바로 반영되도록 계속 갱신


def set_view(name):
    eye, target = CAMERAS[name]
    app_kit.set_camera(eye, target)
    labels.set_detail(name != "overview")
    if name == "pump":
        for n, fr in panel.sec.items():
            fr.collapsed = n not in ("run", "pump")


def show_types(on):
    for key, lb in factory.labels.items():
        if lb.get("group") == "types":
            labels.items[key].visible = on


judge = Judge()
from common.web_link import WebLink  # noqa: E402
OFFLINE_RUN = any(getattr(args, k, None) for k in ("capture", "record", "dataset", "look", "probe"))     # 교재 제작 · 데이터 생성 중
web = WebLink(3, "3회차 가상 공장 · AI 품질검사 · 예지보전", DATA / "live", port=args.port)
if OFFLINE_RUN:
    web.url = "http://<강사 PC 주소>:8080"
with web.record(ui):
    panel = Panel()
if not (args.no_web or OFFLINE_RUN):
    web.start()
    web.url_lb.text = f"주소  {web.url}"
    if args.web_allow:
        web.allow_cb.model.set_value(True)
_step = factory.step


def _step_web(dt):
    _step(dt)
    web.pump()


factory.step = _step_web
labels = app_kit.Labels(factory)
show_types(False)
set_view("overview")
for _ in range(30):
    app.update()
app_kit.arrange_windows(panel.win)


def tick(i, dt=1 / 30.0):
    factory.step(dt)
    res = judge.step()
    if res:
        panel.show_image(*res)
    labels.sync()
    if i % 4 == 0:
        panel.refresh()


runner = app_kit.Runner(tick)

if args.look:
    Path(args.look).mkdir(parents=True, exist_ok=True)
    runner.run(150)
    import os
    for name in (os.environ.get("VIEWS", "").split(",") if os.environ.get("VIEWS") else list(CAMERAS)):
        set_view(name); runner.run(50); runner.still(Path(args.look) / f"look3_{name}.png")
elif args.dataset:                                   # 도장 외관 검사 이미지 촬영
    out = Path(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    plan = [("train", "none", 240, {}), ("test", "none", 60, {})]
    plan += [("test", kind, 15, {}) for kind in SAMPLE_KINDS[1:]]
    plan += [("shift", "none", 10, dict(angle=25.0)), ("shift", "none", 10, dict(level=1.5)), ("shift", "none", 10, dict(backdrop=False))]
    images, rows, seed = [], [], 5000
    rng = np.random.default_rng(0)
    factory.paused = True
    runner.run(60)
    for lv in (0.3, 1.0, 2.0):                     # 조명 밝기에 따른 패널 밝기 (확인용 출력)
        factory.light["level"] = lv; factory.apply_lighting(); runner.run(20)
        im = judge.grab()
        print("level", lv, "panel", im[60:130, 60:130].mean().round(1), "all", im.mean().round(1), flush=True)
    factory.light.update(LIGHT_NOMINAL)
    if __import__("os").environ.get("PROBE"):
        from PIL import Image
        factory.set_sample("pinhole", 1); factory.apply_lighting(); runner.run(20)
        Image.fromarray(judge.grab()).save(out / "probe.png")
        factory.close(); app.close(); sys.exit(0)
    for split, kind, n, cond in plan:
        factory.light.update(LIGHT_NOMINAL); factory.light.update(cond)
        for _ in range(n):
            seed += 1
            factory.set_sample(kind, seed)
            factory.apply_lighting(flicker=float(rng.normal(0, 0.02)))
            for _ in range(24):                    # 앞 시편의 잔상이 사라질 때까지 기다림
                app.update()
            images.append(judge.grab().copy())
            rows.append((split, kind, seed, json.dumps(cond, ensure_ascii=False)))
        print(split, kind, n, cond, flush=True)
    np.savez_compressed(out / "paint_images_raw.npz", images=np.stack(images),
                        split=[r[0] for r in rows], kind=[r[1] for r in rows], seed=[r[2] for r in rows], cond=[r[3] for r in rows])
elif args.capture:
    out = Path(args.capture)
    out.mkdir(parents=True, exist_ok=True)
    runner.run(300)
    panel.period(1); runner.run(150); runner.still(out / "s3_p1_types.png")
    panel.period(2); panel.put_sample("pinhole"); runner.run(120); runner.still(out / "s3_p2_cell.png")
    runner.run(90); runner.still(out / "s3_p2_camera.png")
    panel.period(3); runner.run(900); runner.still(out / "s3_p3_judge.png")
    judge.auto = False
    panel.period(4); factory.pump_reset(1150.0); factory.pump_rate = 12.0; runner.run(560); runner.still(out / "s3_p4_pump.png")
    for n, fr in panel.sec.items():
        fr.collapsed = n not in ("run", "reco")
    set_view("inspection"); panel.set_cond(dict(mold_temp_c=54.0, hold_pressure_mpa=50.0)); runner.run(600)
    runner.still(out / "s3_reco_before.png")
elif args.record:
    out = Path(args.record)
    want = lambda name: args.clips == "all" or name in args.clips.split(",")
    runner.run(300)
    if want("types"):
        panel.period(1); runner.run(60); runner.rec(out / "types", 240, 3)
    if want("shoot"):
        panel.period(2)
        for kind in ("pinhole", "sag"):
            panel.put_sample(kind); runner.run(60)
            def sweep(i):
                if i % 6 == 0:
                    panel.sliders["angle"].model.set_value(round(45.0 + 30.0 * np.sin(i / 300 * 2 * np.pi)))
            runner.rec(out / f"angle_{kind}", 300, 3, sweep)
            panel.light_nominal()
        panel.put_sample("none"); runner.run(40)
        def level(i):
            if i % 6 == 0:
                panel.sliders["level"].model.set_value(round(1.0 + 0.7 * np.sin(i / 240 * 2 * np.pi), 2))
        runner.rec(out / "level", 240, 3, level); panel.light_nominal()
        def back(i):
            if i in (60, 180):
                panel.back_cb.model.set_value(i == 180)
        runner.rec(out / "backdrop", 240, 3, back); panel.light_nominal()
        def kinds(i):
            if i % 50 == 0:
                panel.put_sample(SAMPLE_KINDS[(i // 50) % 5])
        runner.rec(out / "samples", 250, 3, kinds)
    if want("judge"):
        panel.period(3); runner.run(200)
        runner.rec(out / "judge", 420, 3)
        def th(i):
            if i == 40:
                panel.sliders["threshold"].model.set_value(1.2); judge.reset()
        runner.rec(out / "judge_low", 480, 4, th)
        def th2(i):
            if i == 40:
                panel.sliders["threshold"].model.set_value(8.0); judge.reset()
        runner.rec(out / "judge_high", 480, 4, th2)
        panel.sliders["threshold"].model.set_value(2.5)
        def cond(i):
            if i == 40:
                panel.sliders["angle"].model.set_value(25.0); judge.reset()
        runner.rec(out / "judge_shift", 480, 4, cond)
        panel.light_nominal(); judge.auto = False
    if want("pump"):
        panel.period(4); factory.pump_reset(1150.0); panel.sliders["pump_rate"].model.set_value(12.0)
        runner.rec(out / "pump_degrade", 900, 6)
        factory.pump_reset(600.0); panel.sliders["pump_rate"].model.set_value(2.0); panel.need_cb.model.set_value(False); runner.run(60)
        runner.rec(out / "pump_shock1", 240, 3, lambda i: factory.pump_shock() if i in (40, 120) else None)
        factory.pump_reset(600.0); panel.need_cb.model.set_value(True); runner.run(60)
        runner.rec(out / "pump_shock3", 240, 3, lambda i: factory.pump_shock() if i in (40, 120) else None)
        factory.pump_reset(1150.0); panel.base_cb.model.set_value(True); panel.sliders["pump_rate"].model.set_value(10.0)
        runner.rec(out / "pump_base", 900, 6)
        panel.base_cb.model.set_value(False); factory.pump_reset(1080.0)
    if want("reco"):
        for n, fr in panel.sec.items():
            fr.collapsed = n not in ("run", "reco")
        set_view("inspection"); factory.speed = 20
        panel.set_cond(dict(mold_temp_c=54.0, hold_pressure_mpa=50.0)); runner.run(500)
        runner.rec(out / "reco", 640, 4, lambda i: panel.apply_reco() if i == 120 else None)
else:
    last, i = time.perf_counter(), 0
    while app.is_running():
        now = time.perf_counter()
        tick(i, min(now - last, 0.1))
        last = now
        i += 1
        app.update()

web.close()
factory.close()
app.close()
