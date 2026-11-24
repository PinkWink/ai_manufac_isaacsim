"""회차별 실행 파일이 함께 쓰는 틀: Isaac Sim 시작, 3차원 라벨, 화면 전환, 교재용 촬영.

start() 를 가장 먼저 불러야 함 (Isaac Sim 모듈은 앱이 뜬 뒤에만 import 할 수 있음).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
app = None


def quality():
    """화질 설정: DLSS 품질 모드, 주변광 차폐, 반사, 간접광. 자동 노출은 끔."""
    import carb
    s = carb.settings.get_settings()
    for key, value in (("/rtx/post/aa/op", 3), ("/rtx/post/dlss/execMode", 2),             # DLSS 품질 모드
                       ("/rtx/ambientOcclusion/enabled", True), ("/rtx/reflections/enabled", True),
                       ("/rtx/indirectDiffuse/enabled", True), ("/rtx/directLighting/sampledLighting/enabled", True),
                       ("/rtx/directLighting/sampledLighting/samplesPerPixel", 2),
                       ("/rtx/post/histogram/enabled", False), ("/rtx/post/tonemap/filmIso", 100.0)):
        s.set(key, value)


def start(width=1920, height=1080):
    global app
    from isaacsim import SimulationApp
    fonts = ROOT / "assets" / "fonts"
    app = SimulationApp({
        "headless": False, "width": width, "height": height, "window_width": width, "window_height": height,
        "extra_args": [   # 패널·라벨에 한글을 표시하기 위한 글꼴 설정
            f"--/app/font/customFontPath={fonts / 'NanumSquareRoundB.ttf'}",
            f"--/app/font/customRegionFiles=[{fonts / 'korean_region.txt'}]",
            "--/app/viewport/grid/enabled=false",
            "--/persistent/app/viewport/displayOptions=0",
        ],
    })
    quality()
    import omni.usd
    omni.usd.get_context().new_stage()
    for _ in range(10):
        app.update()
    return app, omni.usd.get_context().get_stage()


def abgr(rgb, a=1.0):
    r, g, b = (int(max(0.0, min(1.0, c)) * 255) for c in rgb)
    return (int(a * 255) << 24) | (b << 16) | (g << 8) | r


class Labels:
    """뷰포트 위에 겹쳐 그리는 3차원 라벨. factory.labels 를 그대로 보여 줌."""

    def __init__(self, factory):
        import omni.ui as ui
        from omni.ui import scene as sc
        from omni.kit.viewport.utility import get_active_viewport_window
        self.factory = factory
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
        self.vw.viewport_api.fill_frame = True

    def sync(self):
        if self.factory.labels_dirty:
            for key, lb in self.factory.labels.items():
                self.items[key].text = lb["text"] or " "
            self.factory.labels_dirty = False

    def set_detail(self, on):
        for key, lb in self.factory.labels.items():
            if lb["detail"]:
                self.items[key].visible = on


def set_camera(eye, target):
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=list(eye), target=list(target), camera_prim_path="/OmniverseKit_Persp")


def arrange_windows(panel_window, fraction=0.31):
    import omni.ui as ui
    for name in ("Stage", "Property", "Content", "Console", "Render Settings", "Layer", "Semantics Schema Editor",
                 "Simulation Settings", "Physics Authoring Toolbar"):
        ui.Workspace.show_window(name, False)
    vp = ui.Workspace.get_window("Viewport")
    if vp is not None:
        panel_window.dock_in(vp, ui.DockPosition.RIGHT, fraction)


class Runner:
    """고정 시간 간격으로 장면을 돌리며 화면을 찍는 도구 (교재 제작용)."""

    def __init__(self, tick):
        import omni.renderer_capture
        self.tick = tick                                   # tick(i): 한 프레임 갱신
        self.cap = omni.renderer_capture.acquire_renderer_capture_interface()

    def run(self, frames):
        for i in range(frames):
            self.tick(i)
            app.update()

    def still(self, path):
        self.cap.capture_next_frame_swapchain(str(path))
        for _ in range(4):
            app.update()

    def rec(self, folder, frames, every=3, each=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for i in range(frames):
            if each:
                each(i)
            self.tick(i)
            if i % every == 0:
                self.cap.capture_next_frame_swapchain(str(folder / f"{n:04d}.png"))
                n += 1
            app.update()
        for _ in range(6):
            app.update()
