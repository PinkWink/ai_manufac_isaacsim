#!/usr/bin/env python3
"""가상 공장 수업 환경 점검.

    ./scripts/factory doctor

✅ 통과  ⚠️ 경고(확인 필요)  ❌ 실패(수업 전에 해결)
노트북 실습만 하는 PC(수강생)는 "분석 환경"과 "실습 자료"가 ✅ 이면 됨. Isaac Sim · GPU 는 강사 PC 에만 필요.
종료 코드: 분석 환경이나 실습 자료에 ❌ 가 있으면 1.
"""
import os
import platform
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "✅", "⚠️ ", "❌"
fails = []


def add(status, name, detail="", required=True):
    if status == FAIL and required:
        fails.append(name)
    width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in name)      # 한글은 두 칸
    print(f"{status} {name}{' ' * max(26 - width, 1)}{detail}", flush=True)


def section(title):
    print(f"\n== {title}", flush=True)


def system():
    section("PC")
    osr = {}
    if Path("/etc/os-release").exists():
        for line in Path("/etc/os-release").read_text().splitlines():
            k, _, v = line.partition("=")
            osr[k] = v.strip('"')
    add(OK, "운영체제", osr.get("PRETTY_NAME", platform.platform()))
    try:
        gb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 1024 / 1024
        add(OK if gb >= 7.5 else WARN, "RAM (8GB+)", f"{gb:.1f} GB")
    except Exception as e:  # noqa: BLE001
        add(WARN, "RAM", f"확인 불가: {e}")
    free = shutil.disk_usage(WS).free / 1e9
    add(OK if free >= 3 else WARN, "디스크 여유 (3GB+)", f"{free:.0f} GB")


def analysis():
    section("분석 환경 (노트북 실습)")
    py = WS / ".venv" / "bin" / "python"
    if not py.exists():
        add(FAIL, ".venv", "없음 → ./scripts/factory setup")
        return
    add(OK, ".venv", str(py.parent.parent))
    code = ("import importlib,sys\n"
            "for m in ['numpy','pandas','sklearn','matplotlib','jupyterlab','nbformat']:\n"
            "    try: print(m, importlib.import_module(m).__version__)\n"
            "    except Exception as e: print(m, 'MISSING')\n")
    out = subprocess.run([str(py), "-c", code], capture_output=True, text=True).stdout.split("\n")
    for line in filter(None, out):
        name, ver = line.split(" ", 1)
        add(FAIL if ver == "MISSING" else OK, name, "없음 → ./scripts/factory setup" if ver == "MISSING" else ver)


def materials():
    section("실습 자료")
    for rel, what in (("data/session2/injection_shots_raw.csv", "사출 조건 원본"),
                      ("data/session2/inspection_results.csv", "검사 결과"),
                      ("data/session2/paint_booth.csv", "도장 부스"),
                      ("data/session2/injection_clean_provided.csv", "제공 정제본"),
                      ("session2/notebooks/2-1_data_cleaning.ipynb", "노트북 2-1"),
                      ("session2/notebooks/2-2_analysis_dashboard.ipynb", "노트북 2-2"),
                      ("session2/notebooks/2-3_process_lab.ipynb", "노트북 2-3"),
                      ("data/session3/paint_images.npz", "도장 검사 이미지"),
                      ("data/session3/pump_vibration.npz", "펌프 진동"),
                      ("session3/notebooks/3-1_visual_anomaly.ipynb", "노트북 3-1"),
                      ("session3/notebooks/3-2_predictive_maintenance.ipynb", "노트북 3-2"),
                      ("session3/notebooks/3-3_pump_lab.ipynb", "노트북 3-3"),
                      ("data/session4/robot_log_provided.csv", "로봇 로그"),
                      ("data/session4/assembly_inspection.csv", "조립 · 검사 이력"),
                      ("session4/notebooks/4-1_robot_log_and_branch.ipynb", "노트북 4-1"),
                      ("assets/fonts/NanumSquareRoundB.ttf", "한글 글꼴")):
        p = WS / rel
        add(OK if p.exists() else FAIL, what, rel if p.exists() else f"{rel} 없음 → ./scripts/factory data")


def simulator():
    section("가상 공장 (강사 PC 에만 필요)")
    isaac = Path(os.environ.get("ISAACSIM_PATH", Path.home() / "isaacsim"))
    if (isaac / "python.sh").exists():
        ver = (isaac / "VERSION").read_text().strip().split("+")[0] if (isaac / "VERSION").exists() else "버전 확인 불가"
        add(OK if ver.startswith("5.1") else WARN, "Isaac Sim", f"{ver}  ({isaac})" + ("" if ver.startswith("5.1") else "  ← 5.1 에서 확인했음"),
            required=False)
    else:
        add(WARN, "Isaac Sim", f"{isaac} 에 없음 (다른 곳이면 ISAACSIM_PATH 지정)", required=False)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
        mem = int(q.split(",")[1].split()[0]) if q else 0
        add(OK if mem >= 8000 else WARN, "GPU (8GB+)", q or "없음", required=False)
    except Exception:  # noqa: BLE001
        add(WARN, "GPU", "nvidia-smi 없음", required=False)
    for rel, what in (("assets/sim/ur5e/ur5e.usd", "UR5e 로봇 (4회차)"), ("assets/sim/warehouse/warehouse.usd", "공장 건물")):
        ok = (WS / rel).exists()
        add(OK if ok else WARN, what, rel if ok else "없음 → ./scripts/factory assets  (인터넷 필요)", required=False)
    disp = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    add(OK if disp else WARN, "디스플레이", disp or "없음 (원격 셸에서는 창이 뜨지 않음)", required=False)


if __name__ == "__main__":
    system(); analysis(); materials(); simulator()
    print()
    if fails:
        print(f"{FAIL} 해결할 항목: {', '.join(fails)}")
        sys.exit(1)
    print(f"{OK} 노트북 실습 준비 완료")
