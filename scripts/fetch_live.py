#!/usr/bin/env python3
"""강사 PC 의 가상 공장이 방금 기록한 데이터를 내려받음 (같은 내부망).

    ./scripts/factory fetch <강사 PC 주소>          예: ./scripts/factory fetch 192.168.0.10

받은 파일은 data/session<회차>/live/ 에 저장됨. 실습 노트북에서 USE_LIVE = True 로 바꾸면 이 파일을 씀.
"""
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

WS = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    addr = sys.argv[1].replace("http://", "").strip("/")
    base = f"http://{addr if ':' in addr else addr + ':8080'}"
    try:
        state = json.load(urllib.request.urlopen(base + "/api/state", timeout=5))
    except Exception as e:  # noqa: BLE001
        print(f"강사 PC 에 연결하지 못했습니다: {base}  ({e})")
        print("확인: 주소(강사 패널의 '수강생 연결' 구역), 같은 내부망인지, 가상 공장이 떠 있는지"); return 1
    files = state.get("files", [])
    out = WS / "data" / f"session{state['session']}" / "live"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{state['title']}  →  {out.relative_to(WS)}")
    if not files:
        print("아직 기록된 파일이 없습니다. 강사 화면에서 기록이 시작된 뒤 다시 실행하세요."); return 0
    for f in files:
        name = Path(f["name"]).name
        data = urllib.request.urlopen(f"{base}/live/{quote(name)}", timeout=20).read()
        (out / name).write_bytes(data)
        print(f"  받음: {name}  ({len(data) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
