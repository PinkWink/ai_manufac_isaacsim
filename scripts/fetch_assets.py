"""가상 공장의 3D 에셋(로봇 · 건물)을 NVIDIA 에셋 서버에서 내려받아 assets/sim 에 모음 (인터넷 필요, 처음 한 번).

    ./scripts/factory assets

Isaac Sim 의 Collect 기능으로 USD 가 참조하는 재질 · 텍스처까지 함께 받으므로, 받은 뒤에는 인터넷 없이 가상 공장이 뜸.
에셋은 NVIDIA 의 라이선스를 따르며 이 저장소에는 넣지 않음 (.gitignore).
"""
import asyncio
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

enable_extension("omni.kit.usd.collect")
app.update()
from omni.kit.usd.collect import Collector  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "sim"
ASSETS = {                                   # 이름: 에셋 서버 안의 경로
    "ur5e": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
}


async def collect(name, url):
    done = asyncio.Event()
    c = Collector(url, str(OUT / name), usd_only=False, flat_collection=False, skip_existing=True)
    await c.collect(lambda step, total: None, lambda: done.set())
    await done.wait()


def main():
    root = get_assets_root_path()
    if not root:
        print("에셋 서버에 연결하지 못했습니다 (인터넷 확인)."); return 1
    want = sys.argv[1:] or list(ASSETS)
    for name in want:
        print(f"받는 중: {name}  ←  {root}{ASSETS[name]}", flush=True)
        task = asyncio.ensure_future(collect(name, root + ASSETS[name]))
        while not task.done():
            app.update()
        task.result()
        n = sum(1 for _ in (OUT / name).rglob("*") if _.is_file())
        mb = sum(f.stat().st_size for f in (OUT / name).rglob("*") if f.is_file()) / 1e6
        print(f"  완료: assets/sim/{name}  ({n}개 파일, {mb:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
