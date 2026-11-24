"""4회차 (설비 연계·장비 제어·자율제조 운영 실무) 로봇 셀 실행 파일.

실행:  ./scripts/factory session4
옵션:  --capture <폴더>   교재용 화면 촬영        --record <폴더>   교재용 영상 프레임 촬영
       --dataset <폴더>   실습용 로봇 로그 생성
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
for name in ("--capture", "--record", "--dataset"):
    parser.add_argument(name, default=None)
parser.add_argument("--clips", default="all")
parser.add_argument("--probe", action="store_true")
parser.add_argument("--no-web", action="store_true", help="수강생 연결(웹 조작 · 데이터 받기)을 끔")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--web-allow", action="store_true", help="처음부터 조별 조작을 허용한 상태로 시작")
args, _ = parser.parse_known_args()

from common import app_kit  # noqa: E402
app, stage = app_kit.start()

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.ui as ui  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402

from common import robot_cell as rc  # noqa: E402

DATA = ROOT / "data" / "session4"
cell = rc.RobotCell(stage, DATA / "live")
for _ in range(5):
    app.update()
omni.timeline.get_timeline_interface().play()
for _ in range(10):
    app.update()
robot = SingleArticulation("/World/Cell/Robot")
robot.initialize()
cell.attach(robot)
print("robot dof:", robot.num_dof, robot.dof_names, flush=True)

WHITE, GRAY, CYAN, RED, GREEN, YELLOW = 0xFFFFFFFF, 0xFFAAAAAA, 0xFFFFDD66, 0xFF5555FF, 0xFF66DD66, 0xFF33CCFF
IMG = 160


# ============================================================================
# 비전 판정: 카메라 영상에서 커넥터(주황)가 보이는지 확인
# ============================================================================
class Vision:
    def __init__(self):
        from omni.kit.viewport.utility import create_viewport_window
        self.vp = create_viewport_window("비전 카메라", width=400, height=440, camera_path="/World/Cell/Camera")
        self.image, self.ratio = None, 0.0
        self.threshold = 1.0                  # 주황 화소 비율 (%) 이 이 값 이상이면 커넥터 있음

    def request(self):
        from omni.kit.viewport.utility import capture_viewport_to_buffer

        def done(buffer, size, width, height, fmt):
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_byte * size)
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            raw = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
            a = np.frombuffer(raw.contents, dtype=np.uint8).reshape(height, width, 4)[..., :3]
            f = max(min(height, width) // IMG, 1)
            y0, x0 = (height - IMG * f) // 2, (width - IMG * f) // 2
            self.image = a[y0:y0 + IMG * f, x0:x0 + IMG * f].reshape(IMG, f, IMG, f, 3).mean(axis=(1, 3)).astype(np.uint8)
            r, g, b = (self.image[..., i].astype(int) for i in range(3))
            self.ratio = float(((r > 160) & (r - b > 70)).mean() * 100.0)       # 커넥터 색(주황·노랑 계열) 화소의 비율
        capture_viewport_to_buffer(self.vp.viewport_api, done)

    def judge(self, part):
        """영상으로 판정. 영상이 아직 없으면 실제 상태를 그대로 씀."""
        if self.image is None:
            return part["defect"] == "none"
        return self.ratio >= self.threshold


vision = Vision()
cell.judge_fn, cell.request_fn = vision.judge, vision.request


# ============================================================================
# 진행 패널
# ============================================================================
class Panel:
    PERIODS = {1: ("cell", ("run", "control")), 2: ("robot", ("run", "teach")),
               3: ("cell", ("run", "judge", "control")), 4: ("cell", ("run", "interlock"))}

    def __init__(self):
        self.sec, self.sliders = {}, {}
        self.sel = "pick_above"
        self.tau_hist = [0.0] * 200
        self.win = ui.Window("4회차 진행 패널", width=470, height=1000)
        with self.win.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    self._sec_run(); self._sec_teach(); self._sec_judge(); self._sec_control(); self._sec_interlock()
                    web.build_section(ui)

    def section(self, key, title, collapsed=False):
        self.sec[key] = ui.CollapsableFrame(title, height=0, collapsed=collapsed)
        return self.sec[key]

    def period(self, n):
        view, opened = self.PERIODS[n]
        for name, frame in self.sec.items():
            frame.collapsed = name not in opened
        set_view(view)

    def slider(self, key, text, lo, hi, value, fn, step=1.0, width=130):
        with ui.HStack(height=24, spacing=6):
            ui.Label(text, width=width)
            s = ui.FloatSlider(min=lo, max=hi, step=step)
            s.model.set_value(value)
            s.model.add_value_changed_fn(lambda m: fn(m.get_value_as_float()))
            self.sliders[key] = s

    # ---- 화면과 진행 ------------------------------------------------------------
    def _sec_run(self):
        with self.section("run", "화면과 진행"):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=34, spacing=4):
                    for n in (1, 2, 3, 4):
                        ui.Button(f"{n}교시", clicked_fn=lambda n=n: self.period(n), style={"background_color": 0xFF7A4A1E})
                with ui.HStack(height=30, spacing=4):
                    for key, text in (("cell", "셀 전체"), ("robot", "로봇"), ("top", "위에서"), ("trays", "트레이 쪽")):
                        ui.Button(text, clicked_fn=lambda k=key: set_view(k))
                with ui.HStack(height=30, spacing=4):
                    ui.Button("한 사이클 실행", clicked_fn=cell.start_cycle)
                    ui.Button("비전 카메라 창", clicked_fn=lambda: setattr(vision.vp, "visible", not vision.vp.visible))
                self.msg_lb = ui.Label("", height=40, word_wrap=True, style={"color": CYAN, "font_size": 16})
                with ui.HStack(height=20):
                    ui.Label("관절", width=90, style={"color": GRAY})
                    for name in rc.JOINTS:
                        ui.Label(name.split()[0], style={"color": GRAY})
                with ui.HStack(height=22):
                    ui.Label("각도 (도)", width=90)
                    self.q_lbs = [ui.Label("-") for _ in range(6)]
                with ui.HStack(height=22):
                    ui.Label("토크 (N·m)", width=90)
                    self.t_lbs = [ui.Label("-") for _ in range(6)]
                ui.Label("관절 토크는 물리 엔진이 계산한 값 (실물 장비의 전류에 해당)", height=18, style={"color": GRAY, "font_size": 13})

    # ---- 1단계 인출: 티칭과 로그 --------------------------------------------------
    def _sec_teach(self):
        with self.section("teach", "1단계 인출: 티칭 · 관절 각도와 토크 로그", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                ui.Label("티칭 지점 (눌러서 그 지점으로 이동)", height=20, style={"color": GRAY})
                keys = list(rc.TEACH_KO)
                for row in (keys[:4], keys[4:]):
                    with ui.HStack(height=28, spacing=4):
                        for key in row:
                            ui.Button(rc.TEACH_KO[key], clicked_fn=lambda k=key: self.go_teach(k))
                self.sel_lb = ui.Label("", height=22)
                for i, axis in enumerate("XYZ"):
                    self.slider(f"jog{i}", f"흡착 패드 {axis} (m)", -0.9 if i < 2 else 0.0, 0.9 if i < 2 else 0.7, 0.0,
                                lambda v, i=i: None, step=0.005)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("이 위치로 이동", clicked_fn=self.jog_move)
                    ui.Button("현재 위치를 지점으로 저장", clicked_fn=self.save_point)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("티칭 파일 저장", clicked_fn=lambda: (cell.save_teach(DATA / "work" / "teach_points.json"),
                                                               setattr(cell, "msg", "티칭 지점 저장: data/session4/work/teach_points.json")))
                    ui.Button("기본 티칭으로", clicked_fn=lambda: cell.teach.update(rc.TEACH_DEFAULT))
                with ui.HStack(height=28, spacing=4):
                    self.log_btn = ui.Button("로그 기록 시작", clicked_fn=self.toggle_log)
                self.log_lb = ui.Label("", height=36, word_wrap=True, style={"color": GRAY})

    def go_teach(self, key):
        self.sel = key
        x, y, z = cell.teach[key]
        for i, v in enumerate((x, y, z)):
            self.sliders[f"jog{i}"].model.set_value(v)
        if not (cell.move or cell.moves):
            cell.goto(key, f"티칭: {rc.TEACH_KO[key]}")

    def jog_move(self):
        if not (cell.move or cell.moves):
            cell.goto(tuple(self.sliders[f"jog{i}"].model.get_value_as_float() for i in range(3)), "수동 이동")

    def save_point(self):
        cell.teach[self.sel] = tuple(round(v, 4) for v in cell.fk(cell.q_cmd))
        cell.msg = f"'{rc.TEACH_KO[self.sel]}' 지점을 현재 위치로 저장"

    def toggle_log(self):
        if cell.logging:
            cell.stop_log()
        else:
            cell.start_log()
        self.log_btn.text = "로그 기록 정지" if cell.logging else "로그 기록 시작"

    # ---- 2단계 판단: 부하와 이상 점수 ----------------------------------------------
    def _sec_judge(self):
        with self.section("judge", "2단계 판단: 부하 변화와 토크 이상 점수", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                ui.Label("J2 어깨 토크 (N·m, 최근 약 13초)", height=18, style={"color": GRAY})
                self.tau_plot = ui.Plot(ui.Type.LINE, -100.0, 10.0, *self.tau_hist, height=70,
                                        style={"color": 0xFF66DD66, "background_color": 0xFF202020})
                ui.Label("다음 부품의 무게", height=20, style={"color": GRAY})
                with ui.HStack(height=28, spacing=4):
                    for text, kg in ((f"표준 {rc.PART_KG:.1f} kg", None), (f"무거움 {rc.HEAVY_KG[0]:.1f} kg", rc.HEAVY_KG[0]), (f"아주 무거움 {rc.HEAVY_KG[1]:.1f} kg", rc.HEAVY_KG[1])):
                        ui.Button(text, clicked_fn=lambda kg=kg: self.set_next(kg))
                self.next_lb = ui.Label("", height=20)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("정상 기준 학습 (최근 5회)", clicked_fn=cell.learn_baseline)
                self.slider("score_th", "이상 판정 기준", 1.0, 15.0, cell.score_th, lambda v: setattr(cell, "score_th", v), step=0.5)
                ui.Label("사이클별 이상 점수 (0 ~ 30)", height=18, style={"color": GRAY})
                self.score_plot = ui.Plot(ui.Type.HISTOGRAM, 0.0, 30.0, *([0.0] * 40), height=60,
                                          style={"color": 0xFF44CCFF, "background_color": 0xFF202020})
                self.cyc_lbs = [ui.Label("", height=19, style={"font_size": 14}) for _ in range(5)]

    def set_next(self, kg):
        cell.next_payload = kg

    # ---- 3단계 제어: 판정에 따른 분기와 제어 수준 -------------------------------------
    def _sec_control(self):
        with self.section("control", "3단계 제어: 비전 판정으로 분기 · 제어 3수준", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                with ui.HStack(height=30, spacing=4):
                    self.level_btns = [ui.Button(t, clicked_fn=lambda n=n: self.set_level(n))
                                       for n, t in ((1, "수준 1 모니터링"), (2, "수준 2 권고"), (3, "수준 3 자동"))]
                self.level_lb = ui.Label("", height=38, word_wrap=True, style={"color": GRAY})
                with ui.HStack(height=24, spacing=8):
                    self.auto_cb = ui.CheckBox(width=22)
                    self.auto_cb.model.add_value_changed_fn(lambda m: setattr(cell, "auto", m.get_value_as_bool()))
                    ui.Label("자동 운전 (부품을 계속 공급)")
                self.slider("defect", "불량 부품 비율", 0.0, 1.0, cell.defect_rate, lambda v: setattr(cell, "defect_rate", v), step=0.05)
                self.slider("vis_th", "비전 기준 (주황 %)", 0.2, 6.0, vision.threshold, lambda v: setattr(vision, "threshold", v), step=0.1)
                self.vis_lb = ui.Label("", height=22)
                with ui.HStack(height=30, spacing=4):
                    self.ok_btn = ui.Button("사람 결정: 양품으로", clicked_fn=lambda: cell.decide("ok"))
                    self.ng_btn = ui.Button("사람 결정: 재검으로", clicked_fn=lambda: cell.decide("ng"))
                self.count_lb = ui.Label("", height=24, style={"font_size": 17})
                ui.Button("집계 초기화", height=26, clicked_fn=lambda: cell.counts.update(ok=0, recheck=0, wrong=0))

    LEVEL_TEXT = {1: "읽기 전용 모니터링: 판정은 화면에만 표시. 로봇은 모든 부품을 다음 공정(양품 트레이)으로 보냄",
                  2: "권고 · 사람 조치: 재검 판정이 나오면 멈추고 사람의 결정을 기다림",
                  3: "범위 제한 자동 제어: 판정대로 자동 분기. 인터록 범위 안에서만 동작"}

    def set_level(self, n):
        cell.level = n
        if n != 2 and cell.pending is not None:
            cell.decide("ng")

    # ---- 안전 인터록 --------------------------------------------------------------
    def _sec_interlock(self):
        with self.section("interlock", "안전 인터록 4종 점검", collapsed=True):
            with ui.VStack(spacing=3, height=0):
                self.slider("speed", "속도 지령 (%)", 10.0, 200.0, cell.speed_pct, lambda v: setattr(cell, "speed_pct", v), step=5.0)
                self.slider("speed_limit", "속도 상한 (%)", 30.0, 150.0, cell.speed_limit, lambda v: setattr(cell, "speed_limit", v), step=5.0)
                self.slider("torque_limit", "J2 토크 임계 (N·m)", 30.0, 140.0, cell.torque_limit, lambda v: setattr(cell, "torque_limit", v), step=0.5)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("범위 밖 지령 보내기", clicked_fn=lambda: cell.goto(rc.OUT_OF_RANGE, "범위 밖 지령"))
                    ui.Button(f"과부하 부품 투입 ({rc.OVERLOAD_KG:.0f} kg)", clicked_fn=lambda: self.set_next(rc.OVERLOAD_KG))
                with ui.HStack(height=28, spacing=4):
                    ui.Button("비상 정지", clicked_fn=cell.press_estop, style={"background_color": 0xFF2222AA})
                    self.person_btn = ui.Button("작업자 진입", clicked_fn=self.toggle_person)
                    ui.Button("복귀", clicked_fn=cell.reset)
                self.state_lb = ui.Label("", height=26, style={"font_size": 18})
                self.check_lbs = {key: ui.Label("", height=21) for key in cell.checks}
                ui.Button("점검표 초기화", height=26, clicked_fn=lambda: cell.checks.update({k: False for k in cell.checks}))

    CHECK_KO = dict(range="동작 범위 제한", speed="속도 상한", torque="토크(전류) 임계", estop="비상 정지 · 방호 영역")

    def toggle_person(self):
        cell.person_in = not cell.person_in
        self.person_btn.text = "작업자 퇴장" if cell.person_in else "작업자 진입"

    # ---- 주기 갱신 ---------------------------------------------------------------
    def refresh(self):
        c = cell
        self.msg_lb.text = c.msg
        for i in range(6):
            self.q_lbs[i].text = f"{c.q[i]:.0f}"
            self.t_lbs[i].text = f"{c.tau[i]:+.1f}"
        self.sel_lb.text = f"선택한 지점: {rc.TEACH_KO[self.sel]}   흡착 패드 현재 위치 ({', '.join(f'{v:.3f}' for v in cell.fk(c.q))})"
        self.log_lb.text = (f"기록 중: {c.log_path.name}  ({c._log_n}행)" if c.logging else
                            "기록 정지" + (f"  · 마지막 파일 data/session4/live/{c.log_path.name}" if hasattr(c, "log_path") else ""))
        self.tau_hist = self.tau_hist[1:] + [float(c.tau[1])]
        self.tau_plot.set_data(*self.tau_hist)
        self.next_lb.text = "다음 부품: " + ("표준" if c.next_payload is None else f"{c.next_payload:.2f} kg") + \
                            ("     정상 기준: 학습됨" if c.base_feat is not None else "     정상 기준: 없음")
        cyc = list(c.cycles)
        self.score_plot.set_data(*([0.0] * (40 - len(cyc)) + [min(x["score"], 30.0) for x in cyc]))
        for i, lb in enumerate(self.cyc_lbs):
            if i < len(cyc[-5:]):
                x = cyc[-5:][i]
                bad = x["score"] >= c.score_th and c.base_feat is not None
                lb.text = f"{x['cycle']}번  J2 평균 {x['feat'][0]:.1f}  최대 {x['feat'][1]:.1f} N·m   점수 {x['score']:.1f}" + ("   ← 이상" if bad else "")
                lb.style = {"font_size": 14, "color": RED if bad else WHITE}
            else:
                lb.text = ""
        for n, b in enumerate(self.level_btns, start=1):
            b.style = {"background_color": 0xFF7A4A1E if n == c.level else 0xFF333333}
        self.level_lb.text = self.LEVEL_TEXT[c.level]
        self.vis_lb.text = f"마지막 영상의 주황 화소 {vision.ratio:.2f}%  →  " + ("커넥터 있음" if vision.ratio >= vision.threshold else "커넥터 없음")
        wait = c.pending is not None
        self.ok_btn.style = {"background_color": 0xFF2E7D32 if wait else 0xFF333333}
        self.ng_btn.style = {"background_color": 0xFF1E4A9A if wait else 0xFF333333}
        self.count_lb.text = f"양품 트레이 {c.counts['ok']}   재검 트레이 {c.counts['recheck']}   잘못 보낸 것 {c.counts['wrong']}"
        stopped = c.estop or bool(c.trip)
        self.state_lb.text = ("정지: " + (c.trip or "비상 정지")) if stopped else "운전 가능"
        self.state_lb.style = {"font_size": 18, "color": RED if stopped else GREEN}
        for key, lb in self.check_lbs.items():
            lb.text = ("[확인] " if c.checks[key] else "[  -  ] ") + self.CHECK_KO[key]
            lb.style = {"color": GREEN if c.checks[key] else GRAY}


def set_view(name):
    eye, target = rc.CAMERAS[name]
    app_kit.set_camera(eye, target)
    labels.set_detail(True)


from common.web_link import WebLink  # noqa: E402
OFFLINE_RUN = any(getattr(args, k, None) for k in ("capture", "record", "dataset", "look", "probe"))     # 교재 제작 · 데이터 생성 중
web = WebLink(4, "4회차 로봇 셀 · 설비 연계 · 제어", DATA / "live", port=args.port)
if OFFLINE_RUN:
    web.url = "http://<강사 PC 주소>:8080"
with web.record(ui):
    panel = Panel()
if not (args.no_web or OFFLINE_RUN):
    web.start()
    web.url_lb.text = f"주소  {web.url}"
    if args.web_allow:
        web.allow_cb.model.set_value(True)
_step = cell.step


def _step_web(dt):
    _step(dt)
    web.pump()


cell.step = _step_web
labels = app_kit.Labels(cell)
set_view("cell")
for _ in range(30):
    app.update()
app_kit.arrange_windows(panel.win)
panel.period(1)


def tick(i, dt=1 / 60.0):
    cell.step(dt)
    labels.sync()
    if i % 4 == 0:
        panel.refresh()


runner = app_kit.Runner(tick)

def idle():
    return cell.move is None and not cell.moves and cell.part is None and cell.pending is None


def wait_idle(limit=1500):
    for i in range(limit):
        tick(i); app.update()
        if i > 30 and idle():
            break


def one_cycle(defect="none", kg=None):
    cell.force_defect, cell.next_payload = defect, kg
    cell.start_cycle()
    wait_idle()


if args.probe:                                      # 개발용: 부품 무게별 J2 토크
    runner.run(120)
    cell.torque_limit, cell.torque_ng_rate = 999.0, 0.0
    for kg in (None, None, None, rc.HEAVY_KG[0], rc.HEAVY_KG[1], rc.OVERLOAD_KG):
        t0 = cell.t; cell.track_err = np.zeros(6)
        one_cycle("none", kg)
        print('PROBE track err deg', np.round(cell.track_err, 1), flush=True)
        f = cell.cycles[-1]["feat"]
        print(f"PROBE kg {cell.payload_kg:.2f}  J2 mean {f[0]:.2f} max {f[1]:.2f}  J3 mean {f[2]:.2f}  cycle {cell.t - t0:.1f}s  ratio {vision.ratio:.2f}  trip '{cell.trip}' msg '{cell.msg}'", flush=True)
    if args.capture:                                # 동작 중 자세 확인용 연속 촬영
        set_view("robot"); cell.force_defect = "none"; cell.start_cycle()
        runner.rec(Path(args.capture) / "probe_seq", 620, 20)
        wait_idle()
    one_cycle("connector"); print("PROBE ng ratio", round(vision.ratio, 2), cell.counts, flush=True)
    print("PROBE range", cell.goto(rc.OUT_OF_RANGE, "x"), cell.msg, [k for k in rc.TEACH_DEFAULT if cell.ik(*rc.TEACH_DEFAULT[k]) is None], flush=True)
    runner.still(Path(args.capture or "/tmp") / "probe.png") if args.capture else None
elif args.dataset:                                    # 실습용 로봇 로그: 30 사이클, 중간에 무거운 부품
    out = Path(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    cell.log_dir, cell.torque_ng_rate, cell.torque_limit = out, 0.0, 999.0
    plan = [None] * 12 + [rc.HEAVY_KG[0]] * 5 + [None] * 5 + [rc.HEAVY_KG[1]] * 3 + [None] * 5
    runner.run(120)
    cell.start_log()
    truth = []
    for kg in plan:
        one_cycle("none", kg)
        truth.append((cell.cycle, round(cell.payload_kg, 3)))
        print("cycle", cell.cycle, "kg", round(cell.payload_kg, 3), "feat", np.round(cell.cycles[-1]["feat"], 2), flush=True)
    cell.stop_log()
    cell.log_path.rename(out / "robot_log_raw.csv")
    (out / "robot_log_truth_raw.csv").write_text("cycle,payload_kg\n" + "".join(f"{c},{k}\n" for c, k in truth))
elif args.capture:
    out = Path(args.capture)
    out.mkdir(parents=True, exist_ok=True)
    cell.torque_ng_rate = 0.0
    runner.run(120)
    one_cycle("none"); one_cycle("connector")
    cell.force_defect = "none"; cell.start_cycle(); runner.run(330); runner.still(out / "s4_cell.png")
    wait_idle()
    panel.period(2); panel.go_teach("pick_above"); runner.run(200); runner.still(out / "s4_p2_teach.png")
    cell.goto("home", "대기로"); wait_idle(400)
    panel.period(3)
    for _ in range(4):
        one_cycle("none")
    cell.learn_baseline(); one_cycle("none", rc.HEAVY_KG[0]); one_cycle("none"); one_cycle("none", rc.HEAVY_KG[1])
    runner.run(30); runner.still(out / "s4_p3_judge.png")
    panel.set_level(2); cell.force_defect = "connector"; cell.start_cycle(); runner.run(260); runner.still(out / "s4_p3_level2.png")
    cell.decide("ng"); wait_idle(); panel.set_level(3)
    panel.period(4); cell.force_defect = "none"; cell.start_cycle(); runner.run(250); panel.toggle_person(); runner.run(260)
    runner.still(out / "s4_p4_interlock.png")
    set_view("top"); runner.run(60); runner.still(out / "s4_top.png")
elif args.record:
    out = Path(args.record)
    want = lambda name: args.clips == "all" or name in args.clips.split(",")
    cell.torque_ng_rate = 0.0
    runner.run(120)
    one_cycle("none")

    def rec_cycle(name, defect="none", kg=None, frames=760, every=5, each=None):
        cell.force_defect, cell.next_payload = defect, kg
        cell.start_cycle()
        runner.rec(out / name, frames, every, each)
        wait_idle()

    if want("cycle"):
        panel.period(1); set_view("robot")
        rec_cycle("cycle_ok", "none")
        rec_cycle("cycle_ng", "connector")
        set_view("cell"); cell.auto = True
        runner.rec(out / "auto", 900, 6); cell.auto = False; wait_idle()
    if want("teach"):
        panel.period(2)
        order = ["pick_above", "pick", "pick_above", "ok_above", "ok_place", "ok_above", "ng_above", "ng_place", "ng_above", "home"]
        runner.rec(out / "teach", 1080, 6, lambda i: panel.go_teach(order[i // 108]) if i % 108 == 0 else None)
        wait_idle(400)
        panel.toggle_log()
        rec_cycle("log", "none")
        panel.toggle_log()
    if want("judge"):
        panel.period(3); panel.sliders["speed"].model.set_value(90.0)
        cell.cycles.clear()
        for _ in range(5):
            one_cycle("none")
        cell.learn_baseline()
        rec_cycle("load_normal", "none", None, 560, 5)
        rec_cycle("load_heavy", "none", rc.HEAVY_KG[0], 560, 5)
        rec_cycle("load_heavier", "none", rc.HEAVY_KG[1], 560, 5)
        panel.sliders["speed"].model.set_value(60.0)
    if want("level"):
        panel.period(3)
        for name, fr in panel.sec.items():
            fr.collapsed = name not in ("run", "control")
        cell.counts.update(ok=0, recheck=0, wrong=0)
        panel.set_level(1); rec_cycle("level1", "connector")
        panel.set_level(2); rec_cycle("level2", "connector", None, 900, 6, lambda i: cell.decide("ng") if i == 330 else None)
        panel.set_level(3); rec_cycle("level3", "connector")
        panel.sliders["vis_th"].model.set_value(5.5)
        panel.set_level(2); rec_cycle("level2_override", "none", None, 900, 6, lambda i: cell.decide("ok") if i == 330 else None)
        panel.sliders["vis_th"].model.set_value(1.0); panel.set_level(3)
    if want("interlock"):
        panel.period(4); cell.checks.update({k: False for k in cell.checks})
        runner.rec(out / "il_range", 240, 4, lambda i: cell.goto(rc.OUT_OF_RANGE, "범위 밖 지령") if i == 60 else None)
        panel.sliders["speed"].model.set_value(180.0)
        rec_cycle("il_speed", "none", None, 520, 5)
        panel.sliders["speed"].model.set_value(60.0)
        cell.force_defect, cell.next_payload = "none", rc.OVERLOAD_KG
        cell.start_cycle()
        runner.rec(out / "il_torque", 760, 5, lambda i: cell.reset() if i == 520 else None)
        wait_idle()
        cell.force_defect = "none"; cell.start_cycle()
        def person(i):
            if i == 200:
                panel.toggle_person()
            if i == 560:
                panel.toggle_person()
            if i == 800:
                cell.reset()
        runner.rec(out / "il_estop", 1080, 6, person)
        wait_idle()
else:
    last, i = time.perf_counter(), 0
    while app.is_running():
        tick(i)
        i += 1
        app.update()

web.close()
cell.close()
app.close()
