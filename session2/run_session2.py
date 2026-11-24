"""2회차 (제조데이터 수집·정제·분석·활용) 가상 공장 실행 파일.

실행:  ~/isaacsim/python.sh session2/run_session2.py
옵션:  --capture <폴더>   교재용 화면을 자동으로 찍고 종료 (강의 때는 쓰지 않음)
       --record <폴더>    교재용 움직이는 그림의 프레임을 찍고 종료 (scripts/make_gifs_s2.sh 로 GIF 변환)
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--capture", default=None)
parser.add_argument("--look", default=None, help="개발용: 화면 전환 위치마다 한 장씩 찍고 종료")
parser.add_argument("--record", default=None, help="교재용 움직이는 그림의 프레임을 폴더에 저장")
parser.add_argument("--clips", default="all", help="--record 에서 찍을 묶음: all 또는 tour,inspect,paint,errors,cond")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--no-web", action="store_true", help="수강생 연결(웹 조작 · 데이터 받기)을 끔")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--web-allow", action="store_true", help="처음부터 조별 조작을 허용한 상태로 시작")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

FONT_DIR = ROOT / "assets" / "fonts"
app = SimulationApp({
    "headless": False, "width": args.width, "height": args.height,
    "window_width": args.width, "window_height": args.height,
    "extra_args": [   # 패널·라벨에 한글을 표시하기 위한 글꼴 설정
        f"--/app/font/customFontPath={FONT_DIR / 'NanumSquareRoundB.ttf'}",
        f"--/app/font/customRegionFiles=[{FONT_DIR / 'korean_region.txt'}]",
        "--/app/viewport/grid/enabled=false",
        "--/persistent/app/viewport/displayOptions=0",
    ],
})

import carb  # noqa: E402
from common import app_kit  # noqa: E402
app_kit.quality()
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from omni.ui import scene as sc  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport_window  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

from common import process_model as pm  # noqa: E402
from common.factory import CAMERAS, ERRORS, VirtualFactory  # noqa: E402

omni.usd.get_context().new_stage()
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()
factory = VirtualFactory(stage, ROOT / "data" / "session2" / "live")
factory.open_logs()

WHITE, GRAY, CYAN = 0xFFFFFFFF, 0xFFAAAAAA, 0xFFFFDD66          # ABGR
MARK_COLOR = dict(missing=0xFF4444FF, outlier=0xFFFF44FF, unit=0xFF33CCFF, clock=0xFF3399FF)
TAG_ROWS = [("timestamp", "PLC 시각"), ("mold_temp_c", "금형온도"), ("melt_temp_c", "수지온도"),
            ("inj_pressure_mpa", "사출압"), ("hold_pressure_mpa", "보압"), ("hold_time_s", "보압 시간"),
            ("cooling_time_s", "냉각 시간"), ("cycle_time_s", "사이클타임")]
INJ_KO = dict(mold_temp_c="금형온도 (℃)", melt_temp_c="수지온도 (℃)", inj_pressure_mpa="사출압 (MPa)",
              hold_pressure_mpa="보압 (MPa)", hold_time_s="보압 시간 (s)", cooling_time_s="냉각 시간 (s)")
PAINT_KO = dict(booth_temp_c="부스 온도 (℃)", booth_humidity_pct="부스 습도 (%)",
                viscosity_s="도료 점도 (s)", film_thickness_um="도막 두께 (㎛)")
SCENARIOS = [   # 현장에서 생기는 변동 (실습 데이터에 넣은 것과 같은 변동)
    ("기준 조건", {}),
    ("금형 온조기 이상", dict(mold_temp_c=52.0)),
    ("냉각 시간 단축", dict(cooling_time_s=23.5)),
    ("보압 낮음", dict(hold_pressure_mpa=49.0)),
    ("사출압 저하", dict(inj_pressure_mpa=79.0)),
]
PAINT_SCENARIOS = [
    ("기준 조건", {}),
    ("장마철 (습도↑)", dict(booth_humidity_pct=74.0, booth_temp_c=26.0)),
    ("도료 과희석 (점도↓)", dict(viscosity_s=17.0)),
    ("도막 과다", dict(film_thickness_um=50.0)),
]


def abgr(rgb, a=1.0):
    r, g, b = (int(max(0.0, min(1.0, c)) * 255) for c in rgb)
    return (int(a * 255) << 24) | (b << 16) | (g << 8) | r


# ============================================================================
# 3차원 라벨 (뷰포트 위에 겹쳐 그림)
# ============================================================================
class Labels:
    def __init__(self):
        self.vw = get_active_viewport_window()
        self.items = {}
        with self.vw.get_frame("factory_labels"):
            self.view = sc.SceneView()
            with self.view.scene:
                for key, lb in factory.labels.items():
                    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(*lb["pos"])):
                        with sc.Transform(look_at=sc.Transform.LookAt.CAMERA):
                            self.items[key] = sc.Label(lb["text"] or " ", size=lb["size"], color=abgr(lb["color"]),
                                                       alignment=ui.Alignment.CENTER)
        self.vw.viewport_api.add_scene_view(self.view)

    def sync(self):
        if factory.labels_dirty:
            for key, lb in factory.labels.items():
                self.items[key].text = lb["text"] or " "
            factory.labels_dirty = False

    def set_detail(self, on):
        for key, lb in factory.labels.items():
            if lb["detail"]:
                self.items[key].visible = on


# ============================================================================
# 진행 패널
# ============================================================================
class Panel:
    def __init__(self):
        self.sliders, self.psliders = {}, {}
        self.sec = {}
        self.cells = {}
        self.err_boxes = {}
        self.rate_dirty = True
        self.inspect_vp = None
        self.win = ui.Window("2회차 진행 패널", width=470, height=1000)
        with self.win.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    self._sec_run()
                    self._sec_tags()
                    self._sec_errors()
                    self._sec_setpoint()
                    self._sec_stats()
                    self._sec_paint()
                    self._sec_log()
                    web.build_section(ui)

    # ---- 1. 진행 ---------------------------------------------------------------
    def _sec_run(self):
        self.sec["run"] = ui.CollapsableFrame("화면과 진행", height=0)
        with self.sec["run"]:
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=34, spacing=4):          # 교시별로 화면과 패널을 한 번에 맞춤
                    for n in (1, 2, 3, 4):
                        ui.Button(f"{n}교시", clicked_fn=lambda n=n: self.period(n),
                                  style={"background_color": 0xFF7A4A1E})
                names = [("overview", "공장 전체"), ("injection", "사출기"), ("dataflow", "수집 경로"),
                         ("inspection", "외관 검사기"), ("server", "서버·MES"), ("paint", "도장 부스"),
                         ("samples", "결함 견본")]
                for row in (names[:3], names[3:6], names[6:]):
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
                self.clock_lb = ui.Label("", height=22, style={"color": CYAN, "font_size": 18})

    PERIODS = {   # 교시: (화면, 펼칠 패널)
        1: ("dataflow", ("run", "tags")),
        2: ("inspection", ("run", "tags", "stats")),
        3: ("dataflow", ("run", "tags", "errors")),
        4: ("inspection", ("run", "setpoint", "stats")),
    }

    def period(self, n):
        view, opened = self.PERIODS[n]
        self.show_sections(opened)
        set_view(view)

    def show_sections(self, opened):
        for name, frame in self.sec.items():
            frame.collapsed = name not in opened

    def _toggle_pause(self):
        factory.paused = not factory.paused
        self.pause_btn.text = "다시 진행" if factory.paused else "일시 정지"

    def _toggle_inspect_view(self):
        if self.inspect_vp is None:
            from omni.kit.viewport.utility import create_viewport_window
            self.inspect_vp = create_viewport_window("검사 카메라", width=520, height=420,
                                                     camera_path="/World/Inspect/Camera")
        else:
            self.inspect_vp.visible = not self.inspect_vp.visible

    # ---- 2. 실시간 태그 ----------------------------------------------------------
    def _sec_tags(self):
        self.sec["tags"] = ui.CollapsableFrame("PLC 가 올린 값 (마지막 샷)", height=0)
        with self.sec["tags"]:
            with ui.VStack(spacing=2, height=0):
                with ui.HStack(height=22):
                    ui.Label("항목", width=90, style={"color": GRAY})
                    for m in factory.machines:
                        ui.Label(m.id, style={"color": GRAY})
                for key, text in TAG_ROWS:
                    with ui.HStack(height=22):
                        ui.Label(text, width=90)
                        for m in factory.machines:
                            self.cells[(m.idx, key)] = ui.Label("-")
                ui.Spacer(height=4)
                ui.Label("서버에 들어온 행 (최근)", height=20, style={"color": GRAY})
                self.row_lbs = [ui.Label("", height=19, style={"font_size": 14}) for _ in range(6)]

    # ---- 3. 데이터 오류 재현 ------------------------------------------------------
    def _sec_errors(self):
        self.sec["errors"] = ui.CollapsableFrame("데이터 오류 5종 재현", height=0)
        with self.sec["errors"]:
            with ui.VStack(spacing=3, height=0):
                for key, (name, desc) in ERRORS.items():
                    with ui.HStack(height=24, spacing=8):
                        cb = ui.CheckBox(width=22)
                        cb.model.add_value_changed_fn(lambda m, k=key: factory.errors.__setitem__(k, m.get_value_as_bool()))
                        self.err_boxes[key] = cb
                        ui.Label(name, width=80)
                        ui.Label(desc, style={"color": GRAY})
                with ui.HStack(height=28, spacing=4):
                    ui.Button("모두 켜기", clicked_fn=lambda: self._all_errors(True))
                    ui.Button("모두 끄기", clicked_fn=lambda: self._all_errors(False))

    def _all_errors(self, on):
        for cb in self.err_boxes.values():
            cb.model.set_value(on)

    # ---- 4. 성형 조건 ------------------------------------------------------------
    def _sec_setpoint(self):
        self.sec["setpoint"] = ui.CollapsableFrame("성형 조건 설정 (3대 공통)", height=0)
        with self.sec["setpoint"]:
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
                ui.Label("현장 변동 재현", height=20, style={"color": GRAY})
                for row in (SCENARIOS[:3], SCENARIOS[3:]):
                    with ui.HStack(height=28, spacing=4):
                        for name, change in row:
                            ui.Button(name, clicked_fn=lambda c=change: self._scenario(c))

    def _on_setpoint(self, key, value):
        factory.set_setpoint(key, value)
        self.rate_dirty = True

    def _scenario(self, change):
        for key, s in self.sliders.items():
            s.model.set_value(change.get(key, pm.INJ_NOMINAL[key]))

    # ---- 5. 생산 현황 ------------------------------------------------------------
    def _sec_stats(self):
        self.sec["stats"] = ui.CollapsableFrame("외관 검사 결과", height=0)
        with self.sec["stats"]:
            with ui.VStack(spacing=3, height=0):
                self.stat_lb = ui.Label("", height=24, style={"font_size": 18})
                self.type_lb = ui.Label("", height=22)
                ui.Label("최근 30개 불량률 추이 (0 ~ 100%)", height=20, style={"color": GRAY})
                self.plot = ui.Plot(ui.Type.LINE, 0.0, 1.0, *factory.rate_hist, height=70,
                                    style={"color": 0xFF44CCFF, "background_color": 0xFF202020})
                ui.Button("집계 초기화 · 새 파일로 기록", height=28, clicked_fn=self._reset)

    def _reset(self):
        factory.reset_stats(new_files=True)

    # ---- 6. 도장 부스 ------------------------------------------------------------
    def _sec_paint(self):
        self.sec["paint"] = ui.CollapsableFrame("도장 부스 PB-01", height=0, collapsed=True)
        with self.sec["paint"]:
            with ui.VStack(spacing=3, height=0):
                for key, text in PAINT_KO.items():
                    with ui.HStack(height=24, spacing=6):
                        ui.Label(text, width=120)
                        lo, hi = pm.PAINT_RANGE[key]
                        s = ui.FloatSlider(min=lo, max=hi, step=0.5)
                        s.model.set_value(pm.PAINT_NOMINAL[key])
                        s.model.add_value_changed_fn(lambda m, k=key: self._on_paint(k, m.get_value_as_float()))
                        self.psliders[key] = s
                self.pexp_lb = ui.Label("", height=24, style={"font_size": 18, "color": CYAN})
                self.pstat_lb = ui.Label("", height=22)
                for row in (PAINT_SCENARIOS[:2], PAINT_SCENARIOS[2:]):
                    with ui.HStack(height=28, spacing=4):
                        for name, change in row:
                            ui.Button(name, clicked_fn=lambda c=change: self._paint_scenario(c))

    def _on_paint(self, key, value):
        factory.booth.setpoint[key] = float(value)
        self.rate_dirty = True

    def _paint_scenario(self, change):
        for key, s in self.psliders.items():
            s.model.set_value(change.get(key, pm.PAINT_NOMINAL[key]))

    # ---- 7. 기록 ----------------------------------------------------------------
    def _sec_log(self):
        self.sec["log"] = ui.CollapsableFrame("기록 파일", height=0, collapsed=True)
        with self.sec["log"]:
            with ui.VStack(spacing=3, height=0):
                self.log_lb = ui.Label("", height=40, word_wrap=True, style={"color": GRAY})

    # ---- 주기 갱신 ---------------------------------------------------------------
    def refresh(self):
        f = factory
        self.clock_lb.text = f"서버 시각  {f.clock:%Y-%m-%d %H:%M:%S}    (x{f.speed:.0f})"
        for m in f.machines:
            if not m.tags:
                continue
            row, units, marks = m.tags["row"], m.tags["units"], m.tags["marks"]
            for key, _ in TAG_ROWS:
                v = row[key]
                if key == "timestamp":
                    text = v[11:]
                elif v == "":
                    text = "(값 없음)"
                else:
                    text = f"{v:.1f} {units.get(key, '')}"
                cell = self.cells[(m.idx, key)]
                cell.text = text
                cell.style = {"color": MARK_COLOR.get(marks.get(key), WHITE)}
        rows = list(f.last_rows)[-6:]
        for i, lb in enumerate(self.row_lbs):
            if i < len(rows):
                row, marks, resent = rows[i]
                mt = "    " if row["mold_temp_c"] == "" else f"{row['mold_temp_c']:.1f}"
                lb.text = (f"{row['timestamp'][11:]}  {row['machine_id'][-2:]}호기  금형 {mt}  "
                           f"사출압 {row['inj_pressure_mpa']:.1f}  보압 {row['hold_pressure_mpa']:.1f}"
                           + ("  ← 재전송" if resent else ""))
                lb.style = {"font_size": 14, "color": 0xFF33AAFF if resent else (0xFFCC88FF if marks else WHITE)}
            else:
                lb.text = ""
        self.stat_lb.text = f"검사 {f.n_inspect}개   불량 {f.n_ng}개   불량률 {f.defect_rate * 100:.1f}%"
        self.type_lb.text = "   ".join(f"{pm.DEFECT_KO[k]} {v}" for k, v in f.by_type.items())
        self.plot.set_data(*f.rate_hist)
        self.pstat_lb.text = (f"검사 {f.p_n}장   불량 {f.p_ng}장   "
                              + "   ".join(f"{pm.DEFECT_KO[k]} {v}" for k, v in f.p_by_type.items() if k != "none"))
        if self.rate_dirty:
            self.rate_dirty = False
            self.exp_lb.text = f"이 조건의 예상 불량률  {f.expected_rate() * 100:.1f}%"
            self.pexp_lb.text = f"이 조건의 예상 불량률  {f.booth.expected_defect_rate() * 100:.1f}%"
        self.log_lb.text = f"data/session2/live/  shots · inspection · paint _{f.log_stamp}.csv   (전송 {f.rows_sent}행)"


def set_view(name):
    eye, target = CAMERAS[name]
    set_camera_view(eye=list(eye), target=list(target), camera_prim_path="/OmniverseKit_Persp")
    if labels is not None:
        labels.set_detail(name != "overview")
    if name == "paint" and panel is not None:
        panel.show_sections(("run", "paint"))


def arrange_windows(panel):
    for name in ("Stage", "Property", "Content", "Console", "Render Settings", "Layer", "Semantics Schema Editor",
                 "Simulation Settings", "Physics Authoring Toolbar"):
        ui.Workspace.show_window(name, False)
    vp = ui.Workspace.get_window("Viewport")
    if vp is not None:
        panel.win.dock_in(vp, ui.DockPosition.RIGHT, 0.31)


# ============================================================================
# 실행
# ============================================================================
labels = panel = None
from common.web_link import WebLink  # noqa: E402
OFFLINE_RUN = any(getattr(args, k, None) for k in ("capture", "record", "dataset", "look", "probe"))     # 교재 제작 · 데이터 생성 중
web = WebLink(2, "2회차 가상 공장 · 제조데이터", ROOT / "data" / "session2" / "live", port=args.port)
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
labels = Labels()
labels.vw.viewport_api.fill_frame = True
set_view("overview")
for _ in range(30):
    app.update()
arrange_windows(panel)


def capture(path):
    import omni.renderer_capture
    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(path))
    for _ in range(4):
        app.update()


def run_frames(n, dt_=1 / 30.0):
    for i in range(n):
        factory.step(dt_)
        labels.sync()
        if i % 8 == 0:
            panel.refresh()
        app.update()


if args.look:
    Path(args.look).mkdir(parents=True, exist_ok=True)
    run_frames(200)
    for name in (os.environ.get("VIEWS", "").split(",") if os.environ.get("VIEWS") else list(CAMERAS)):
        set_view(name); run_frames(50); capture(str(Path(args.look) / f"look_{name}.png"))
elif args.capture:                                   # 교재용 화면 자동 촬영
    out = Path(args.capture)
    out.mkdir(parents=True, exist_ok=True)
    run_frames(900)
    for name in ("overview", "injection", "server", "samples"):
        set_view(name)
        run_frames(150)
        capture(out / f"s2_{name}.png")
    panel.period(1)
    run_frames(200)
    capture(out / "s2_p1_dataflow.png")
    panel.period(2)
    panel._toggle_inspect_view()
    run_frames(260)
    capture(out / "s2_p2_inspection.png")
    panel._toggle_inspect_view()
    set_view("paint")
    panel._paint_scenario(dict(booth_humidity_pct=74.0, booth_temp_c=26.0))
    run_frames(1100)
    capture(out / "s2_p2_paint.png")
    panel._paint_scenario({})
    panel.period(3)
    panel._all_errors(True)
    run_frames(700)
    capture(out / "s2_p3_errors.png")
    panel._all_errors(False)
    panel.period(4)
    panel._reset()
    panel._scenario(dict(mold_temp_c=52.0))
    run_frames(1700)
    capture(out / "s2_p4_mold_low.png")
elif args.record:                                  # 교재용 움직이는 그림 프레임 촬영
    import math
    import omni.renderer_capture
    cap_if = omni.renderer_capture.acquire_renderer_capture_interface()
    out = Path(args.record)

    def rec(name, frames, every=3, each=None):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for i in range(frames):
            if each:
                each(i)
            factory.step(1 / 30.0)
            labels.sync()
            if i % 8 == 0:
                panel.refresh()
            if i % every == 0:
                cap_if.capture_next_frame_swapchain(str(d / f"{n:04d}.png"))
                n += 1
            app.update()
        for _ in range(6):
            app.update()

    def speed(v):
        factory.speed = float(v)

    def settle(frames=450):                         # 기준 조건으로 되돌리고 라인을 비움
        panel._scenario({})
        panel._all_errors(False)
        speed(30)
        run_frames(frames)
        speed(12)

    want = lambda name: args.clips == "all" or name in args.clips.split(",")
    run_frames(600)
    if want("tour"):
        # A. 공장 전체를 돌아보는 화면
        def orbit(i):
            a = math.radians(-90 + 28 * math.sin(i / 240 * 2 * math.pi))
            set_camera_view(eye=[4.5 + 25 * math.cos(a), 2.8 + 25 * math.sin(a), 10.5], target=[4.5, 2.8, 0.8],
                            camera_prim_path="/OmniverseKit_Persp")
        set_view("overview")
        rec("overview", 240, 3, orbit)
        # B. 한 샷: 금형 닫힘 → 열림 → 취출
        set_view("injection"); speed(5)
        rec("shot_cycle", 450, 4)
        # C. 수집 경로
        speed(12); panel.period(1)
        rec("dataflow", 300, 3)
        set_view("server")
        rec("server", 210, 3)
    if want("inspect"):
        # D. 외관 검사와 불량 배출 (금형온도를 조금 낮춰 양품·불량이 섞이게 함)
        panel.period(2); panel._toggle_inspect_view()
        panel.sliders["mold_temp_c"].model.set_value(54.5)
        run_frames(520)
        rec("inspection", 360, 3)
        panel._toggle_inspect_view()
        settle()
    if want("paint"):
        # E. 도장 부스
        set_view("paint"); speed(30)
        run_frames(200)
        rec("paint_humid", 480, 4, lambda i: panel._paint_scenario(dict(booth_humidity_pct=74.0, booth_temp_c=26.0)) if i == 50 else None)
        panel._paint_scenario({}); run_frames(420)
        rec("paint_thin", 480, 4, lambda i: panel._paint_scenario(dict(viscosity_s=17.0)) if i == 50 else None)
        panel._paint_scenario({}); speed(12)
    if want("errors"):
        # F. 데이터 오류 다섯 종
        panel.period(3)
        for key in ("missing", "outlier", "unit", "clock", "dup"):
            panel._all_errors(False)
            run_frames(120)
            rec(f"err_{key}", 300, 3, lambda i, k=key: panel.err_boxes[k].model.set_value(True) if i == 30 else None)
        panel._all_errors(False)
    if want("cond"):
        # G. 성형 조건을 바꾸면
        for name, change in (("mold_low", dict(mold_temp_c=52.0)), ("cooling_short", dict(cooling_time_s=23.5)),
                             ("hold_low", dict(hold_pressure_mpa=49.0)), ("inj_low", dict(inj_pressure_mpa=79.0))):
            settle()
            panel.period(4); panel._reset(); speed(20)
            run_frames(60)
            rec(f"cond_{name}", 520, 4, lambda i, c=change: panel._scenario(c) if i == 40 else None)
        settle()
else:
    last, tick = time.perf_counter(), 0
    while app.is_running():
        now = time.perf_counter()
        factory.step(min(now - last, 0.1))
        last = now
        labels.sync()
        tick += 1
        if tick % 8 == 0:
            panel.refresh()
        app.update()

web.close()
factory.close()
app.close()
