#!/usr/bin/env python3
"""이 저장소의 wrapper(scripts/factory)가 띄운 프로세스를 찾고 끝냄.

wrapper 가 AI_MANUFAC_HOME=<저장소> 를 내보내므로, 환경에 그 값이 있는 프로세스가 이 저장소의 것임.

    python3 scripts/factory_procs.py          # 목록, 하나라도 있으면 종료 코드 1
    python3 scripts/factory_procs.py --sim    # 가상 공장(Isaac Sim)만
    python3 scripts/factory_procs.py --kill   # 모두 끝냄 (SIGTERM, 안 끝나면 SIGKILL)
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
MARKER = f"AI_MANUFAC_HOME={WS}".encode()
SIM_KEYS = ("run_session", "isaacsim", "/kit/")
IGNORE = ("factory_procs.py",)


def ours():
    skip, pid = set(), os.getpid()
    while pid > 1:                                   # 자기 자신과 부모(이 점검을 부른 wrapper)는 제외
        skip.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            if MARKER not in (d / "environ").read_bytes().split(b"\0"):
                continue
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if cmd and not any(k in cmd for k in IGNORE):
            out.append((int(d.name), cmd))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--kill", action="store_true")
    a = ap.parse_args()
    procs = [(p, c) for p, c in ours() if not a.sim or any(k in c for k in SIM_KEYS)]
    if not procs:
        if a.kill:
            print("남아 있는 프로그램 없음")
        return 0
    for pid, cmd in procs:
        print(f"  {pid:>7}  {cmd[:110]}")
    if not a.kill:
        return 1
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
        for pid, _ in procs:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        end = time.time() + wait
        while time.time() < end and any(Path(f"/proc/{p}").exists() for p, _ in procs):
            time.sleep(0.2)
        procs = [(p, c) for p, c in procs if Path(f"/proc/{p}").exists()]
        if not procs:
            break
    print("끝냈습니다" if not procs else f"끝내지 못한 프로세스 {len(procs)}개")
    return 0 if not procs else 1


if __name__ == "__main__":
    sys.exit(main())
