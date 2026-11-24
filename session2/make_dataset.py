"""2회차 실습 데이터 생성 (교안 표 28 의 3번·4번 데이터셋).

실행:  .venv/bin/python session2/make_dataset.py

만드는 파일 (data/session2/)
- injection_shots_raw.csv   : 사출기 PLC 가 올린 샷 단위 성형 조건 (오류 5종 포함)
- inspection_results.csv    : 외관 검사기가 올린 판정 결과 (검사기 시계 기준)
- injection_clean_answer.csv: 정제·결합 정답본 (강사용, 교육생에게 배포하지 않음)
- injection_clean_provided.csv: 실습 2-1 을 끝내지 못한 교육생이 2-2 에서 쓰는 정제본
- error_log_answer.csv      : 어느 행에 어떤 오류를 넣었는지 (강사용)
- paint_booth.csv           : 도장 부스 환경·도막·불량 데이터
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import process_model as pm  # noqa: E402

OUT = ROOT / "data" / "session2"
SEED = 20261208
ANSWERS = "--answers" in sys.argv        # 강사용 정답 파일도 만들기: ./scripts/factory data --answers
MACHINES = ["IMM-01", "IMM-02", "IMM-03"]
DAYS = pd.bdate_range("2026-11-16", periods=12)       # 영업일 12일
SHOTS_PER_LOT = 139                                   # 3대 x 12일 x 139 = 5,004 샷
START_HOUR = {"IMM-01": 8.5, "IMM-02": 9.0, "IMM-03": 13.0}
CLOCK_OFFSET_S = 7 * 60 + 12                          # IMM-02 PLC 시계가 빨라진 양
CLOCK_OFFSET_FROM = DAYS[5]                           # 6일째 PLC 재기동 이후


def day_setpoint(machine, day_idx, rng):
    """날짜·설비별 설정값. 현장에서 실제로 생기는 변동 세 가지를 넣었음."""
    sp = dict(pm.INJ_NOMINAL)
    sp["mold_temp_c"] += rng.normal(0, 1.0)
    sp["hold_pressure_mpa"] += rng.normal(0, 1.0)
    sp["melt_temp_c"] += rng.normal(0, 1.5)
    note = ""
    if machine == "IMM-01" and day_idx in (3, 4):         # 금형 온조기(칠러) 이상 → 금형온도 저하
        sp["mold_temp_c"] -= 7.0 if day_idx == 3 else 9.0
        note = "금형 온조기 이상"
    if machine == "IMM-03" and day_idx in (7, 8, 9):      # 생산량을 맞추려 냉각 시간 단축
        sp["cooling_time_s"] -= 4.5
        note = "냉각 시간 단축"
    if machine == "IMM-02" and day_idx >= 9:              # 보압 설정을 낮춘 채 운전
        sp["hold_pressure_mpa"] -= 6.0
        note = "보압 설정 낮음"
    if machine == "IMM-01" and day_idx in (10, 11):       # 노즐 막힘으로 사출압이 덜 걸림
        sp["inj_pressure_mpa"] -= 11.0
        note = "사출압 저하"
    return sp, note


def make_true_table(rng):
    rows = []
    for d_idx, day in enumerate(DAYS):
        for m in MACHINES:
            sp, note = day_setpoint(m, d_idx, rng)
            n = SHOTS_PER_LOT
            v = {k: sp[k] + rng.normal(0, pm.INJ_NOISE[k], n) for k in pm.INJ_NOMINAL}
            # 가동 초반 금형이 덜 데워진 구간 (첫 15샷)
            v["mold_temp_c"] -= 5.0 * np.exp(-np.arange(n) / 6.0)
            cyc = pm.injection_cycle_time(v["hold_time_s"], v["cooling_time_s"]) + rng.normal(0, 0.3, n)
            t0 = day + pd.Timedelta(hours=START_HOUR[m])
            ts = t0 + pd.to_timedelta(np.cumsum(cyc), unit="s")
            defect = pm.sample_defect_array(pm.injection_defect_probs(**v), rng)
            df = pd.DataFrame({k: np.round(x, 2) for k, x in v.items()})
            df.insert(0, "timestamp", ts.round("s"))
            df.insert(1, "machine_id", m)
            df.insert(2, "lot_id", f"L{day:%m%d}-{m[-2:]}")
            df.insert(3, "shot_no", np.arange(1, n + 1))
            df["cycle_time_s"] = np.round(cyc, 2)
            df["ambient_temp_c"] = np.round(21 + 3 * np.sin(d_idx / 2.0) + rng.normal(0, 0.4, n), 1)
            df["defect_type"] = defect
            df["is_defect"] = (defect != "none").astype(int)
            df["note"] = note
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_inspection(true, rng):
    """외관 검사기 기록. 검사기 시계는 정확하고, 샷 번호는 모름(시각·설비로만 묶을 수 있음)."""
    ins = pd.DataFrame({
        "inspect_time": true["timestamp"] + pd.to_timedelta(
            np.round(true["machine_id"].map(pm.INSPECT_DELAY_S) + rng.normal(0, 2.0, len(true))), unit="s"),
        "machine_id": true["machine_id"],
        "result": np.where(true["is_defect"] == 1, "NG", "OK"),
        "defect_type": true["defect_type"],
    })
    ins = ins.sort_values(["inspect_time", "machine_id"]).reset_index(drop=True)
    ins.insert(0, "inspect_id", [f"V{i:05d}" for i in range(1, len(ins) + 1)])
    return ins


def inject_errors(true, rng):
    """정답 표에 오류 5종을 넣어 PLC 원본처럼 만듦."""
    raw = true.drop(columns=["defect_type", "is_defect", "note"]).copy()
    log = []
    n = len(raw)

    # ③ 단위 혼재: IMM-03 은 압력을 bar 로 올림 (1 MPa = 10 bar)
    m3 = raw["machine_id"] == "IMM-03"
    raw.loc[m3, ["inj_pressure_mpa", "hold_pressure_mpa"]] *= 10.0
    log.append(("단위 혼재", int(m3.sum()), "IMM-03 의 사출압·보압이 bar 단위"))

    # ④ 시각 오차: IMM-02 PLC 시계가 6일째부터 7분 12초 빠름
    m2 = (raw["machine_id"] == "IMM-02") & (raw["timestamp"] >= CLOCK_OFFSET_FROM)
    raw.loc[m2, "timestamp"] += pd.Timedelta(seconds=CLOCK_OFFSET_S)
    log.append(("시각 오차", int(m2.sum()), f"IMM-02, {CLOCK_OFFSET_FROM:%m-%d} 이후 +{CLOCK_OFFSET_S}초"))

    # ① 결측: 금형온도 센서 단선 구간 4회(각 30~45샷) + 낱개 결측
    miss = 0
    for _ in range(4):
        m = rng.choice(MACHINES)
        day = rng.choice(len(DAYS))
        idx = raw.index[(raw["machine_id"] == m) & (raw["lot_id"].str[1:5] == f"{DAYS[day]:%m%d}")]
        s = rng.integers(10, 80)
        seg = idx[s:s + rng.integers(30, 46)]
        raw.loc[seg, "mold_temp_c"] = np.nan
        miss += len(seg)
    for col, frac in (("melt_temp_c", 0.006), ("hold_pressure_mpa", 0.006), ("cooling_time_s", 0.004)):
        idx = rng.choice(n, int(n * frac), replace=False)
        raw.loc[idx, col] = np.nan
        miss += len(idx)
    log.append(("결측", miss, "금형온도 센서 단선 구간 4회 + 낱개 결측"))

    # ② 이상치: 통신 오류 값(0, 999.9)과 자릿수 밀림
    out = 0
    for col, bad in (("inj_pressure_mpa", 999.9), ("melt_temp_c", 0.0), ("mold_temp_c", 999.9)):
        idx = rng.choice(n, 12, replace=False)
        raw.loc[idx, col] = bad
        out += len(idx)
    idx = rng.choice(n, 10, replace=False)
    raw.loc[idx, "cycle_time_s"] *= 10
    out += len(idx)
    log.append(("이상치", out, "통신 오류 값 0·999.9, 사이클타임 자릿수 밀림"))

    # ⑤ 중복: 게이트웨이 재전송으로 같은 행이 두 번
    dup_idx = rng.choice(n, 96, replace=False)
    raw = pd.concat([raw, raw.loc[dup_idx]], ignore_index=True)
    log.append(("중복", len(dup_idx), "게이트웨이 재전송"))

    raw = raw.sort_values(["timestamp", "machine_id"], kind="stable").reset_index(drop=True)
    return raw, pd.DataFrame(log, columns=["오류 유형", "건수", "내용"])


def make_paint(rng, n=2400):
    days = DAYS
    per = n // len(days)
    rows = []
    for d_idx, day in enumerate(days):
        sp = dict(pm.PAINT_NOMINAL)
        # 날씨에 따라 부스 습도가 오르는 날 (공조 용량 부족), 도료 희석이 과했던 날
        sp["booth_humidity_pct"] += [0, 2, 9, 16, 5, 0, -3, 1, 12, 20, 7, 0][d_idx]
        sp["booth_temp_c"] += [0, 0, 1, 2.5, 1, 0, -2, -3, 1, 2, 0, 0][d_idx]
        sp["viscosity_s"] += [0, 0, 0, 0, 0, -4, -4.5, 0, 0, 0, 0, 0][d_idx]
        v = {k: sp[k] + rng.normal(0, pm.PAINT_NOISE[k], per) for k in pm.PAINT_NOMINAL}
        ts = day + pd.Timedelta(hours=9) + pd.to_timedelta(np.arange(per) * 144, unit="s")
        defect = pm.sample_defect_array(pm.paint_defect_probs(**v), rng)
        df = pd.DataFrame({k: np.round(x, 1) for k, x in v.items()})
        df.insert(0, "timestamp", ts)
        df.insert(1, "booth_id", "PB-01")
        df.insert(2, "lot_id", f"P{day:%m%d}")
        df["defect_type"] = defect
        df["is_defect"] = (defect != "none").astype(int)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    true = make_true_table(rng)
    ins = make_inspection(true, rng)
    raw, log = inject_errors(true, rng)
    paint = make_paint(rng)

    raw.to_csv(OUT / "injection_shots_raw.csv", index=False)
    ins.to_csv(OUT / "inspection_results.csv", index=False)
    if ANSWERS:
        true.to_csv(OUT / "injection_clean_answer.csv", index=False)
    provided = true.drop(columns=["note"]).copy()          # 실습 2-1 을 못 끝낸 교육생이 2-2 에서 쓰는 정제본
    provided.insert(provided.columns.get_loc("cycle_time_s") + 1, "cooling_ratio",
                    (provided["cooling_time_s"] / provided["cycle_time_s"]).round(3))
    provided.to_csv(OUT / "injection_clean_provided.csv", index=False)
    if ANSWERS:
        log.to_csv(OUT / "error_log_answer.csv", index=False)
    paint.to_csv(OUT / "paint_booth.csv", index=False)

    print(f"샷 {len(true)}행 → 원본 {len(raw)}행 (중복 포함), 검사 {len(ins)}행, 도장 {len(paint)}행")
    print(f"사출 불량률 {true['is_defect'].mean():.3f}")
    print(true["defect_type"].value_counts().to_string())
    print(log.to_string(index=False))
    print(f"도장 불량률 {paint['is_defect'].mean():.3f}")
    print(paint["defect_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
