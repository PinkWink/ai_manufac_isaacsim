"""3회차 실습 데이터 생성.

실행:  ./scripts/factory data [--answers]        (또는 .venv/bin/python session3/make_dataset.py)

만드는 파일 (data/session3/)
- paint_images.npz          : 도장 외관 검사 이미지 390장 (가상 공장 검사 카메라 촬영) + paint_labels.csv
- pixel_model_provided.npz  : 정상 이미지로 만든 기준 모델 (가상 공장의 실시간 판정이 사용)
- pump_vibration.npz        : 유압 펌프 4대의 진동 구간 + pump_windows.csv, failure_log.csv
- pump_truth_answer.csv     : 구간별 실제 열화 정도 (강사용)
- recommend_provided.json   : 2회차 분류 모델로 찾은 성형 조건 추천 범위

paint_images_raw.npz 는 `./scripts/factory dataset3` 으로 가상 공장에서 다시 촬영할 수 있음.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import process_model as pm  # noqa: E402
from common.anomaly import PixelModel, prepare  # noqa: E402

OUT = ROOT / "data" / "session3"
SEED = 20261210
ANSWERS = "--answers" in sys.argv        # 강사용 정답 파일도 만들기: ./scripts/factory data --answers
UNITS = {"PUMP-01": 21.5, "PUMP-02": 27.2, "PUMP-03": 13.4, "PUMP-04": None}     # 고장까지의 일수 (None: 고장 없음)
STEP_H, DAYS, LIFE_H, DOWN_H = 2, 30, 1500.0, 24


def paint():
    if (OUT / "paint_images_raw.npz").exists():          # 가상 공장에서 새로 촬영한 원본이 있으면 그것으로 다시 만듦
        raw = np.load(OUT / "paint_images_raw.npz")
        images, split, kind, cond = raw["images"], raw["split"], raw["kind"], raw["cond"]
        np.savez_compressed(OUT / "paint_images.npz", images=images)
        lab = pd.DataFrame({"index": np.arange(len(images)), "split": split, "kind": kind,
                            "is_defect": (kind != "none").astype(int), "condition": cond})
        lab.to_csv(OUT / "paint_labels.csv", index=False)
    images = np.load(OUT / "paint_images.npz")["images"]
    lab = pd.read_csv(OUT / "paint_labels.csv")
    split = lab["split"].values
    x = np.stack([prepare(im) for im in images[split == "train"]])
    PixelModel().fit(x).save(OUT / "pixel_model_provided.npz")
    print(f"도장 이미지 {len(images)}장:", lab.groupby(["split", "kind"]).size().to_dict())


def pump(rng):
    t0 = pd.Timestamp("2026-11-09 00:00")
    rows, wins, truth, fails = [], [], [], []
    for u, (unit, fail_day) in enumerate(UNITS.items()):
        fail_h = None if fail_day is None else fail_day * 24
        p = pm.PumpUnit(seed=SEED + u, life_h=None if fail_h is None else LIFE_H, age_h=0.0 if fail_h is None else LIFE_H - fail_h)
        run0 = 3200 + 400 * u
        for h in range(0, DAYS * 24, STEP_H):
            if fail_h is not None and fail_h <= h < fail_h + DOWN_H:        # 고장 후 교체까지 정지
                continue
            if fail_h is not None and h >= fail_h + DOWN_H and p.life_h == LIFE_H and p.age_h > 0:
                p = pm.PumpUnit(seed=SEED + 50 + u, life_h=LIFE_H, age_h=0.0)   # 새 베어링
                run0 = -h
            if fail_h is not None and h < fail_h:
                p.age_h = LIFE_H - fail_h + h
            elif fail_h is not None:
                p.age_h = h - fail_h - DOWN_H
            shock = rng.random() < 0.015
            wins.append(p.window("bearing", shock=shock))
            rows.append(dict(window_id=len(rows), unit_id=unit, timestamp=t0 + pd.Timedelta(hours=h),
                             run_hours=run0 + h, current_rms_a=round(p.current_rms(), 2)))
            left = p.hours_to_failure if (fail_h is not None and h < fail_h) else None
            truth.append(dict(window_id=len(rows) - 1, unit_id=unit, hours_to_failure=left,
                              severity=round(p.severity(), 3), shock=int(shock)))
        if fail_h is not None:
            fails.append(dict(unit_id=unit, failure_time=t0 + pd.Timedelta(hours=fail_h), part="펌프측 베어링",
                              action="베어링 교체", downtime_h=DOWN_H))
    w = np.stack(wins)
    np.savez_compressed(OUT / "pump_vibration.npz", vibration=w.astype(np.float16), fs=pm.PUMP_FS)
    pd.DataFrame(rows).to_csv(OUT / "pump_windows.csv", index=False)
    pd.DataFrame(fails).to_csv(OUT / "failure_log.csv", index=False)
    if ANSWERS:
        pd.DataFrame(truth).to_csv(OUT / "pump_truth_answer.csv", index=False)
    print(f"펌프 진동 {w.shape[0]}구간 x {w.shape[1]}표본, 고장 {len(fails)}건")


def reco():
    from sklearn.ensemble import RandomForestClassifier
    from common.recommend import CONDITIONS, recommend
    df = pd.read_csv(ROOT / "data" / "session2" / "injection_clean_provided.csv")
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(df[CONDITIONS], df["is_defect"])
    out = recommend(model, df, dict(pm.INJ_NOMINAL))
    (OUT / "recommend_provided.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("추천 범위:", {k: (v["low"], v["high"], v["recommend"]) for k, v in out.items()})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    paint()
    pump(np.random.default_rng(SEED))
    reco()
