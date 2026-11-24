"""4회차 실습 데이터 생성.

실행:  ./scripts/factory data [--answers]        (또는 .venv/bin/python session4/make_dataset.py)

만드는 파일 (data/session4/)
- robot_log_provided.csv     : 가상 공장 로봇암의 관절 각도 · 토크 로그 30 사이클 (20 Hz). 중간에 무거운 부품이 섞여 있음
- robot_log_truth_answer.csv : 사이클별 실제 부품 무게 (강사용)
- assembly_inspection.csv    : 조립 체결 토크 이력 · 최종검사 외관 판정 2,000대 (교안 표 28 의 6번 데이터셋)

robot_log_raw.csv 가 있으면 그것으로 다시 만듦 (가상 공장에서 `run_session4.py --dataset data/session4` 로 생성).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "session4"
SEED = 20261217
ANSWERS = "--answers" in sys.argv        # 강사용 정답 파일도 만들기: ./scripts/factory data --answers
TORQUE_SPEC = (1.8, 2.2)        # 체결 토크 규격 (N·m)


def robot_log():
    raw = OUT / "robot_log_raw.csv"
    if raw.exists():
        df = pd.read_csv(raw)
        df = df[df["cycle"] > 0].copy()
        df["time_s"] = (df["time_s"] - df["time_s"].iloc[0]).round(3)
        df.to_csv(OUT / "robot_log_provided.csv", index=False)
        if ANSWERS and (OUT / "robot_log_truth_raw.csv").exists():
            pd.read_csv(OUT / "robot_log_truth_raw.csv").to_csv(OUT / "robot_log_truth_answer.csv", index=False)
    df = pd.read_csv(OUT / "robot_log_provided.csv")
    print(f"로봇 로그 {len(df)}행, {df['cycle'].nunique()} 사이클")


def assembly(rng, n=2000):
    t0 = pd.Timestamp("2026-12-01 08:00")
    kind = rng.choice(["ok", "low_torque", "missing_screw", "no_connector", "half_connector"], n, p=[0.905, 0.035, 0.01, 0.035, 0.015])
    tq = rng.normal(2.0, 0.045, (n, 4))
    ratio = rng.normal(4.8, 0.25, n)
    for i, k in enumerate(kind):
        j = rng.integers(4)
        if k == "low_torque":
            tq[i, j] = rng.uniform(1.25, 1.78)          # 체결 토크 부족
        elif k == "missing_screw":
            tq[i, j] = 0.0                               # 체결 누락
        elif k == "no_connector":
            ratio[i] = abs(rng.normal(0.05, 0.05))       # 커넥터 미삽입
        elif k == "half_connector":
            ratio[i] = rng.uniform(1.2, 3.2)             # 커넥터 반삽입: 영상으로는 애매함
    df = pd.DataFrame(np.round(tq, 3), columns=[f"torque{j}_nm" for j in range(1, 5)])
    df.insert(0, "unit_id", [f"U{i:05d}" for i in range(1, n + 1)])
    df.insert(1, "timestamp", t0 + pd.to_timedelta(np.arange(n) * 42, unit="s"))
    df["connector_ratio_pct"] = np.round(np.clip(ratio, 0, None), 2)
    df["recheck_result"] = np.where(kind == "ok", "양품", "불량")       # 사람이 재검해서 확정한 결과
    df["defect_kind"] = pd.Series(kind).map({"ok": "", "low_torque": "토크 부족", "missing_screw": "체결 누락",
                                             "no_connector": "커넥터 미삽입", "half_connector": "커넥터 반삽입"}).values
    df.to_csv(OUT / "assembly_inspection.csv", index=False)
    print(f"조립·검사 {n}대:", df["defect_kind"].replace("", "양품").value_counts().to_dict())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    assembly(np.random.default_rng(SEED))
    if (OUT / "robot_log_raw.csv").exists() or (OUT / "robot_log_provided.csv").exists():
        robot_log()
    else:
        print("로봇 로그 없음: 가상 공장에서 먼저 생성해야 함")
