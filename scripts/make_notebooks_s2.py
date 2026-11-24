"""2회차 실습 노트북(2-1, 2-2)을 만듦.  실행: .venv/bin/python scripts/make_notebooks_s2.py"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "session2" / "notebooks"


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)


SETUP = '''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "common").exists():      # 저장소 최상위 폴더 찾기
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from common.plot_style import use_korean_font
use_korean_font()

DATA = ROOT / "data" / "session2"
OUT = DATA / "work"                        # 내 산출물이 저장되는 곳
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
'''

# ============================================================================
# 실습 2-1 데이터 정제
# ============================================================================
nb21 = [
("md", '''
# 실습 2-1. 사출 데이터 정제

가상 공장의 사출기 3대(IMM-01·02·03)가 PLC 로 올린 **샷 단위 성형 조건** 5,000여 행을 정제함.
이 데이터에는 현장에서 흔히 생기는 오류 다섯 종이 섞여 있음.

| 오류 | 가상 공장에서 본 원인 |
|---|---|
| 결측 | 금형온도 센서 단선 |
| 이상치 | 통신 오류 값(0, 999.9), 자릿수 밀림 |
| 단위 혼재 | 한 설비만 압력을 bar 로 전송 |
| 시각 오차 | 한 설비의 PLC 시계가 어긋남 |
| 중복 | 게이트웨이 재전송 |

- **기본 트랙**: 셀을 위에서부터 차례로 실행(`Shift` + `Enter`)하고, 각 단계의 결과를 워크북 규칙표에 옮겨 적음.
- **심화 트랙**: `# 심화` 표시가 있는 값을 바꿔 보고 결과가 어떻게 달라지는지 확인함.
'''),
("code", SETUP),
("md", '''
## 1. 수집 검증: 무엇이 들어왔는지 확인

파일은 두 개임. 성형 조건은 사출기 PLC 가, 검사 결과는 외관 검사기가 따로 올림.
'''),
("code", '''
shots = pd.read_csv(DATA / "injection_shots_raw.csv", parse_dates=["timestamp"])
insp = pd.read_csv(DATA / "inspection_results.csv", parse_dates=["inspect_time"])
print("성형 조건:", shots.shape, "  검사 결과:", insp.shape)
shots.head()
'''),
("code", '''
insp.head()
'''),
("code", '''
COND = ["mold_temp_c", "melt_temp_c", "inj_pressure_mpa", "hold_pressure_mpa",
        "hold_time_s", "cooling_time_s", "cycle_time_s"]
KO = {"mold_temp_c": "금형온도(℃)", "melt_temp_c": "수지온도(℃)", "inj_pressure_mpa": "사출압(MPa)",
      "hold_pressure_mpa": "보압(MPa)", "hold_time_s": "보압 시간(s)", "cooling_time_s": "냉각 시간(s)",
      "cycle_time_s": "사이클타임(s)"}
shots[COND].describe().T.round(1)
'''),
("md", '''
> **읽어 보기**: 최솟값·최댓값이 상식과 맞는지 봄. 금형온도 최댓값 999.9, 수지온도 최솟값 0, 보압 최댓값 500 이상은 정상 값이 아님.

규칙을 정할 때마다 아래 `rules` 목록에 한 줄씩 기록함. 이것이 산출물인 **정제 규칙표**가 됨.
'''),
("code", '''
rules = []          # 정제 규칙표
def log_rule(error, how_found, rule, n):
    rules.append({"오류 유형": error, "찾은 방법": how_found, "처리 규칙": rule, "해당 행 수": int(n)})
df = shots.copy()
'''),
("md", '''
## 2. 중복: 같은 행이 두 번 들어옴

게이트웨이가 응답을 못 받으면 같은 데이터를 다시 보냄. 설비·시각·샷 번호가 모두 같은 행은 하나만 남김.
'''),
("code", '''
dup = df.duplicated(subset=["machine_id", "timestamp", "shot_no"], keep="first")
print("중복 행:", dup.sum())
df[df.duplicated(subset=["machine_id", "timestamp", "shot_no"], keep=False)].head(4)
'''),
("code", '''
df = df[~dup].reset_index(drop=True)
log_rule("중복", "설비·시각·샷 번호가 같은 행", "첫 행만 남김", dup.sum())
print("남은 행:", len(df))
'''),
("md", '''
## 3. 단위 혼재: 한 설비만 값이 10배

설비별로 압력 분포를 나란히 그려 봄.
'''),
("code", '''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, col in zip(axes, ["inj_pressure_mpa", "hold_pressure_mpa"]):
    ok = df[df[col] < 990]                      # 통신 오류 값은 잠시 제외하고 봄
    ax.boxplot([ok.loc[ok.machine_id == m, col].dropna() for m in ["IMM-01", "IMM-02", "IMM-03"]],
               tick_labels=["IMM-01", "IMM-02", "IMM-03"])
    ax.set_title(KO[col] + "  (원본)")
plt.tight_layout(); plt.show()
df.groupby("machine_id")[["inj_pressure_mpa", "hold_pressure_mpa"]].median()
'''),
("md", '''
> IMM-03 만 약 10배임. 이 설비는 압력을 **bar** 로 올리고 있음(1 MPa = 10 bar). MPa 로 맞춤.
'''),
("code", '''
is_bar = df["machine_id"] == "IMM-03"
df.loc[is_bar, ["inj_pressure_mpa", "hold_pressure_mpa"]] /= 10.0
log_rule("단위 혼재", "설비별 압력 분포 비교 (IMM-03 만 10배)", "IMM-03 의 사출압·보압을 10 으로 나눠 MPa 로 통일", is_bar.sum())
df.groupby("machine_id")[["inj_pressure_mpa", "hold_pressure_mpa"]].median().round(1)
'''),
("md", '''
## 4. 이상치: 있을 수 없는 값

항목마다 **현장에서 나올 수 있는 범위**를 정하고, 벗어난 값은 "값 없음"으로 바꿈.
범위는 설비 사양과 공정 담당자의 경험으로 정함.
'''),
("code", '''
# 심화: 범위를 좁히거나 넓혀 보고 걸리는 건수가 어떻게 달라지는지 확인
VALID = {
    "mold_temp_c": (20, 120),
    "melt_temp_c": (150, 300),
    "inj_pressure_mpa": (30, 200),
    "hold_pressure_mpa": (10, 150),
    "hold_time_s": (1, 30),
    "cooling_time_s": (5, 90),
    "cycle_time_s": (30, 200),
}
n_out = 0
for col, (lo, hi) in VALID.items():
    bad = (df[col] < lo) | (df[col] > hi)
    if bad.sum():
        print(f"{KO[col]:14s} 범위 {lo}~{hi} 밖: {bad.sum():3d}건   예) {sorted(df.loc[bad, col].unique())[:4]}")
    df.loc[bad, col] = np.nan
    n_out += bad.sum()
log_rule("이상치", "항목별 허용 범위 밖 (0, 999.9, 10배 값)", "값을 비우고 결측으로 처리", n_out)
'''),
("md", '''
## 5. 결측: 값이 비어 있음

어느 항목이, 어느 설비에서, 언제 비었는지부터 봄.
'''),
("code", '''
print(df[COND].isna().sum())
miss = df[df["mold_temp_c"].isna()]
miss.groupby(["machine_id", miss["timestamp"].dt.date]).size().rename("금형온도 결측 수")
'''),
("code", '''
# 금형온도 결측이 시간상 어떻게 몰려 있는지
fig, ax = plt.subplots(figsize=(10, 2.6))
for i, m in enumerate(["IMM-01", "IMM-02", "IMM-03"]):
    d = df[df.machine_id == m]
    ax.scatter(d["timestamp"], np.full(len(d), i), s=2, color="#bbbbbb")
    d = d[d["mold_temp_c"].isna()]
    ax.scatter(d["timestamp"], np.full(len(d), i), s=14, color="crimson")
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["IMM-01", "IMM-02", "IMM-03"])
ax.set_title("금형온도 결측 위치 (빨간 점)"); plt.tight_layout(); plt.show()
'''),
("md", '''
> 낱개로 빠진 값과 **수십 샷이 연달아 빠진 구간**(센서 단선)이 섞여 있음. 둘은 다르게 처리함.
>
> - 짧게 빠진 값: 같은 설비의 앞뒤 값으로 이어 채움 (조건은 샷 사이에 천천히 변하므로)
> - 길게 빠진 구간: 채우면 지어낸 값이 되므로 그 행을 쓰지 않음
'''),
("code", '''
MAX_FILL = 3        # 심화: 연속 몇 개까지 채울지 (0 이면 채우지 않음)
before = df[COND].isna().sum().sum()
df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
df[COND] = df.groupby("machine_id")[COND].transform(
    lambda s: s.interpolate(limit=MAX_FILL, limit_area="inside") if MAX_FILL > 0 else s)
# 긴 구간은 앞에서부터 MAX_FILL 개만 채워지므로, 구간 길이가 MAX_FILL 을 넘으면 채운 값도 되돌림
raw_na = shots.drop_duplicates(["machine_id", "timestamp", "shot_no"]).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)["mold_temp_c"].isna()
run_id = (raw_na != raw_na.shift()).cumsum()
long_gap = raw_na & (raw_na.groupby(run_id).transform("size") > MAX_FILL)
df.loc[long_gap.values, "mold_temp_c"] = np.nan
filled = before - df[COND].isna().sum().sum()
drop = df[COND].isna().any(axis=1)
print(f"채운 값 {filled}개,  쓰지 않는 행 {drop.sum()}개")
df["usable"] = ~drop           # 행은 지우지 않고 표시만 해 둠 (시각 보정에 필요)
log_rule("결측", "항목별 빈 값 수, 설비·시각별 분포", f"연속 {MAX_FILL}개 이하는 앞뒤 값으로 보간, 그보다 긴 구간은 행 제외", drop.sum() + filled)
'''),
("md", '''
## 6. 시각 오차: 설비 시계가 어긋남

성형 조건과 검사 결과에는 공통 번호가 없어서 **설비 번호와 시각**으로 묶어야 함.
사출 후 검사기까지 걸리는 시간은 설비 위치에 따라 다름(취출 10초 + 컨베이어 이동).

| 설비 | 검사기까지 거리 | 걸리는 시간 |
|---|---|---|
| IMM-01 | 11.5 m | 154초 |
| IMM-02 | 7.5 m | 104초 |
| IMM-03 | 3.5 m | 54초 |

시계가 어긋난 설비가 있으면 엉뚱한 부품끼리 묶임.
설비·날짜별로 "그날 첫 검사 시각 − 첫 샷 시각"을 구해 봄. 정상이라면 위 표의 값 근처여야 함.
'''),
("code", '''
first_shot = df.groupby(["machine_id", df["timestamp"].dt.date])["timestamp"].min()
first_insp = insp.groupby(["machine_id", insp["inspect_time"].dt.date])["inspect_time"].min()
gap = (first_insp - first_shot).dt.total_seconds().unstack(0)
gap.index.name = "날짜"
gap.round(0)
'''),
("md", '''
> IMM-02 는 11월 23일부터 값이 약 −328초임. 104 − (−328) = **432초(7분 12초)** 만큼 PLC 시계가 빨라진 것임.
> 그날 PLC 를 재기동하면서 시각 동기화가 풀린 경우에 해당함. 서버 시각 기준으로 되돌림.
'''),
("code", '''
NORMAL_DELAY = pd.Series({"IMM-01": 154, "IMM-02": 104, "IMM-03": 54})   # 사출 → 검사 소요 시간(초)
offset = (NORMAL_DELAY - gap).round(0)              # 설비·날짜별 시계 오차
bad = offset[offset.abs() > 30].stack().dropna()
print("보정이 필요한 설비·날짜\\n", bad, sep="")
n_fix = 0
for (day, machine), sec in bad.items():
    sel = (df["machine_id"] == machine) & (df["timestamp"].dt.date == day)
    df.loc[sel, "timestamp"] -= pd.Timedelta(seconds=float(sec))
    n_fix += sel.sum()
log_rule("시각 오차", "설비·날짜별 (첫 검사 시각 − 첫 샷 시각) 이 정상 소요 시간에서 벗어남",
         "IMM-02 의 11-23 이후 시각에서 432초를 뺌", n_fix)
'''),
("md", '''
## 7. 라벨 결합: 조건과 검사 결과를 묶음

샷 시각에 설비별 소요 시간을 더한 시각과 가장 가까운 검사 기록을 같은 설비 안에서 찾음. 20초 넘게 차이 나면 묶지 않음.
'''),
("code", '''
TOLERANCE = 20      # 심화: 허용 오차(초)를 바꿔 결합률이 어떻게 달라지는지 확인
left = df.assign(expect_time=df["timestamp"] + pd.to_timedelta(df["machine_id"].map(NORMAL_DELAY), unit="s")).sort_values("expect_time")
right = insp.sort_values("inspect_time")
merged = pd.merge_asof(left, right, left_on="expect_time", right_on="inspect_time", by="machine_id",
                       direction="nearest", tolerance=pd.Timedelta(seconds=TOLERANCE))
print(f"결합된 행: {merged['result'].notna().sum()} / {len(merged)}  ({merged['result'].notna().mean()*100:.1f}%)")
'''),
("md", '''
## 8. 특성 생성과 저장

- `is_defect`: 불량이면 1
- `cooling_ratio`: 사이클타임 중 냉각 시간의 비율 (생산량을 맞추려고 냉각을 줄였는지 보는 값)
'''),
("code", '''
clean = merged[merged["usable"] & merged["result"].notna()].copy()
clean["is_defect"] = (clean["result"] == "NG").astype(int)
clean["cooling_ratio"] = (clean["cooling_time_s"] / clean["cycle_time_s"]).round(3)
clean = clean.sort_values(["timestamp", "machine_id"])[
    ["timestamp", "machine_id", "lot_id", "shot_no", *COND, "cooling_ratio", "ambient_temp_c", "defect_type", "is_defect"]]
clean.to_csv(OUT / "injection_clean.csv", index=False)
rule_table = pd.DataFrame(rules)
rule_table.to_csv(OUT / "cleaning_rules.csv", index=False, encoding="utf-8-sig")
print(f"원본 {len(shots)}행 → 정제본 {len(clean)}행,  불량률 {clean['is_defect'].mean()*100:.1f}%")
rule_table
'''),
("md", '''
## 9. 확인: 정제 전과 후

같은 그림을 정제 전·후로 나란히 놓고 봄.
'''),
("code", '''
fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
for j, col in enumerate(["mold_temp_c", "inj_pressure_mpa", "hold_pressure_mpa"]):
    axes[0, j].hist(shots[col].dropna(), bins=60, color="#c0392b"); axes[0, j].set_title("정제 전  " + KO[col])
    axes[1, j].hist(clean[col], bins=60, color="#2471a3"); axes[1, j].set_title("정제 후  " + KO[col])
plt.tight_layout(); plt.show()
'''),
("md", '''
### 워크북에 옮겨 적기

1. 위의 **정제 규칙표** 다섯 줄 (오류 유형 · 찾은 방법 · 처리 규칙 · 해당 행 수)
2. 원본 행 수와 정제본 행 수
3. 우리 회사 데이터라면 다섯 오류 중 어떤 것이 가장 자주 생길지 한 문장

저장된 파일: `data/session2/work/injection_clean.csv`, `data/session2/work/cleaning_rules.csv`
'''),
]

# ============================================================================
# 실습 2-2 분석·시각화
# ============================================================================
nb22 = [
("md", '''
# 실습 2-2. 불량 분석과 대시보드

실습 2-1 에서 만든 정제본으로 **어떤 성형 조건이 불량을 만드는지** 찾고, 불량 분류 모델을 만든 뒤,
일자별 불량률 추이를 대시보드 패널로 그림.

- **기본 트랙**: 셀을 차례로 실행하고 마지막의 "해석 세 문장"을 워크북에 적음.
- **심화 트랙**: `# 심화` 표시가 있는 부분을 바꿔 불량 분류 F1 점수 0.75 이상을 확인하고 더 올려 봄.
'''),
("code", SETUP),
("code", '''
src = OUT / "injection_clean.csv"
if not src.exists():                      # 실습 2-1 을 끝내지 못했으면 제공 정제본을 사용
    src = DATA / "injection_clean_provided.csv"
df = pd.read_csv(src, parse_dates=["timestamp"])
df["date"] = df["timestamp"].dt.strftime("%m-%d")
FEATURES = ["mold_temp_c", "melt_temp_c", "inj_pressure_mpa", "hold_pressure_mpa",
            "hold_time_s", "cooling_time_s", "cycle_time_s", "ambient_temp_c"]
KO = {"mold_temp_c": "금형온도", "melt_temp_c": "수지온도", "inj_pressure_mpa": "사출압",
      "hold_pressure_mpa": "보압", "hold_time_s": "보압 시간", "cooling_time_s": "냉각 시간",
      "cycle_time_s": "사이클타임", "ambient_temp_c": "주변 온도", "cooling_ratio": "냉각 비율"}
DEFECT_KO = {"none": "양품", "flow_mark": "흐름자국", "sink": "수축", "short_shot": "미성형"}
print(src.name, df.shape, f" 불량률 {df['is_defect'].mean()*100:.1f}%")
df["defect_type"].map(DEFECT_KO).value_counts()
'''),
("md", '''
## 1. 언제, 어느 설비에서 불량이 많았나
'''),
("code", '''
pivot = df.pivot_table(index="date", columns="machine_id", values="is_defect", aggfunc="mean") * 100
ax = pivot.plot(marker="o", figsize=(10, 3.4))
ax.set_ylabel("불량률 (%)"); ax.set_title("일자·설비별 불량률"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
pivot.round(1)
'''),
("md", '''
> 불량이 몰린 날이 설비마다 다름. 그날 조건이 평소와 어떻게 달랐는지 봄.
'''),
("code", '''
daily = df.groupby(["machine_id", "date"])[["mold_temp_c", "hold_pressure_mpa", "cooling_time_s", "inj_pressure_mpa"]].mean()
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, col in zip(axes, daily.columns):
    daily[col].unstack(0).plot(ax=ax, marker=".", legend=(col == "mold_temp_c"))
    ax.set_title("일평균 " + KO[col]); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.show()
'''),
("md", '''
## 2. 조건과 불량의 관계

양품과 불량 유형별로 조건 분포를 겹쳐 봄.
'''),
("code", '''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (col, kind) in zip(axes, [("mold_temp_c", "flow_mark"), ("hold_pressure_mpa", "sink"), ("cooling_time_s", "sink")]):
    bins = np.linspace(df[col].min(), df[col].max(), 45)
    ax.hist(df.loc[df.defect_type == "none", col], bins=bins, alpha=0.6, label="양품", color="#2471a3")
    ax.hist(df.loc[df.defect_type == kind, col], bins=bins, alpha=0.8, label=DEFECT_KO[kind], color="#c0392b")
    ax.set_title(f"{KO[col]}  vs  {DEFECT_KO[kind]}"); ax.legend()
plt.tight_layout(); plt.show()
'''),
("code", '''
# 조건을 구간으로 나눠 구간별 불량률을 봄
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
for ax, col in zip(axes, ["mold_temp_c", "hold_pressure_mpa", "cooling_time_s"]):
    g = df.groupby(pd.cut(df[col], 12), observed=True)["is_defect"].agg(["mean", "size"])
    g = g[g["size"] >= 15]
    ax.bar([f"{iv.mid:.1f}" for iv in g.index], g["mean"] * 100, color="#af601a")
    ax.set_title(f"{KO[col]} 구간별 불량률 (%)"); ax.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.show()
'''),
("md", '''
> 금형온도가 낮을수록, 보압이 낮을수록, 냉각 시간이 짧을수록 불량률이 급히 오름.
> 가상 공장 화면에서 슬라이더로 조건을 바꿨을 때 본 현상과 같은지 비교함.
'''),
("md", '''
## 3. 불량 분류 모델

성형 조건만 보고 "이 샷은 불량일 것"을 맞히는 모델을 만듦. 전체의 75% 로 학습하고 나머지 25% 로 평가함.
'''),
("code", '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

USE = list(FEATURES)                      # 심화: 특성을 빼거나 "cooling_ratio" 를 더해 봄
X_train, X_test, y_train, y_test = train_test_split(df[USE], df["is_defect"], test_size=0.25,
                                                    random_state=0, stratify=df["is_defect"])
model = RandomForestClassifier(n_estimators=200,      # 심화: 나무 수
                               min_samples_leaf=3,    # 심화: 잎의 최소 샘플 수
                               random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
'''),
("code", '''
THRESHOLD = 0.5                           # 심화: 불량으로 판정하는 기준 확률
pred = (prob >= THRESHOLD).astype(int)
cm = confusion_matrix(y_test, pred)
result = pd.DataFrame(cm, index=["실제 양품", "실제 불량"], columns=["양품으로 예측", "불량으로 예측"])
print(result, "\\n")
print(f"정밀도 {precision_score(y_test, pred):.3f}   (불량이라 한 것 중 진짜 불량)")
print(f"재현율 {recall_score(y_test, pred):.3f}   (진짜 불량 중 잡아낸 것)")
print(f"F1 점수 {f1_score(y_test, pred):.3f}   (목표 0.75 이상)")
'''),
("code", '''
# 기준 확률을 바꾸면 과검(양품을 불량으로)과 미검(불량을 양품으로)이 어떻게 달라지나
rows = []
for th in np.arange(0.1, 0.95, 0.1):
    p = (prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    rows.append({"기준 확률": round(th, 1), "과검(건)": fp, "미검(건)": fn, "F1": round(f1_score(y_test, p), 3)})
pd.DataFrame(rows).set_index("기준 확률")
'''),
("md", '''
## 4. 변수 중요도: 모델은 무엇을 보고 판단했나
'''),
("code", '''
imp = pd.Series(model.feature_importances_, index=[KO[c] for c in USE]).sort_values()
ax = imp.plot.barh(figsize=(7, 3.4), color="#1e8449")
ax.set_title("변수 중요도"); plt.tight_layout(); plt.show()
imp.sort_values(ascending=False).round(3)
'''),
("md", '''
> 상위 항목이 금형온도·보압·냉각 시간(사이클타임)인지 확인함.
> 중요도는 "이 데이터에서 판정에 많이 쓰였다"는 뜻이지 원인을 증명하는 값이 아님. 그래서 2절의 그림, 공정 지식과 함께 봄.
'''),
("md", '''
## 5. 대시보드 패널

현장 모니터에 띄운다고 생각하고 네 칸짜리 패널로 정리함.
'''),
("code", '''
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
fig.suptitle("사출 라인 불량 현황", fontsize=16)

ax = axes[0, 0]
total = df.groupby("date")["is_defect"].mean() * 100
ax.plot(total.index, total.values, marker="o", color="#c0392b", lw=2)
ax.axhline(5, color="gray", ls="--", lw=1); ax.text(0, 5.6, "관리 기준 5%", color="gray")
ax.set_title("일자별 불량률 (%)"); ax.tick_params(axis="x", rotation=45); ax.grid(alpha=0.3)

ax = axes[0, 1]
kinds = df[df.is_defect == 1].pivot_table(index="date", columns="defect_type", values="is_defect", aggfunc="size", fill_value=0)
kinds = kinds.reindex(total.index, fill_value=0).rename(columns=DEFECT_KO)
kinds.columns.name = None
kinds.plot.bar(stacked=True, ax=ax, color=["#d68910", "#6c3483", "#2e86c1"][:kinds.shape[1]])
ax.set_title("일자별 불량 유형 (건)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=45)

ax = axes[1, 0]
pivot = df.pivot_table(index="date", columns="machine_id", values="is_defect", aggfunc="mean") * 100
im = ax.imshow(pivot.T.values, aspect="auto", cmap="Reds", vmin=0, vmax=60)
ax.set_xticks(range(len(pivot.index))); ax.set_xticklabels(pivot.index, rotation=45)
ax.set_yticks(range(3)); ax.set_yticklabels(pivot.columns)
for i in range(pivot.shape[1]):
    for j in range(pivot.shape[0]):
        v = pivot.values[j, i]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 35 else "black")
ax.set_title("설비·일자별 불량률 (%)")

ax = axes[1, 1]
imp.tail(5).plot.barh(ax=ax, color="#1e8449"); ax.set_title("불량에 영향이 큰 조건 (상위 5)")

plt.tight_layout(); fig.savefig(OUT / "dashboard.png", dpi=130); plt.show()
'''),
("md", '''
## 6. 도장 부스 데이터 (2교시 사례 확인용)

부스 습도·온도·도료 점도·도막 두께와 흐름·핀홀 불량의 관계를 같은 방법으로 봄.
'''),
("code", '''
paint = pd.read_csv(DATA / "paint_booth.csv", parse_dates=["timestamp"])
PK = {"booth_humidity_pct": "부스 습도(%)", "booth_temp_c": "부스 온도(℃)", "viscosity_s": "도료 점도(s)", "film_thickness_um": "도막 두께(㎛)"}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, col in zip(axes, PK):
    for kind, name, color in (("pinhole", "핀홀", "#6c3483"), ("sag", "흐름", "#d68910")):
        g = paint.assign(hit=(paint.defect_type == kind)).groupby(pd.cut(paint[col], 10), observed=True)["hit"].agg(["mean", "size"])
        g = g[g["size"] >= 15]
        ax.plot([iv.mid for iv in g.index], g["mean"] * 100, marker="o", label=name, color=color)
    ax.set_title(PK[col] + " 구간별 불량률 (%)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),
("md", '''
## 7. 산출물 저장과 해석 세 문장
'''),
("code", '''
summary = pd.DataFrame({
    "항목": ["평가 데이터 수", "기준 확률", "정밀도", "재현율", "F1 점수", "중요도 1위", "중요도 2위", "중요도 3위"],
    "값": [len(y_test), THRESHOLD, round(precision_score(y_test, pred), 3), round(recall_score(y_test, pred), 3),
           round(f1_score(y_test, pred), 3), *imp.sort_values(ascending=False).index[:3]],
})
summary.to_csv(OUT / "model_result.csv", index=False, encoding="utf-8-sig")
summary
'''),
("md", '''
### 워크북에 적기: 해석 세 문장

1. **언제·어디서**: 불량률이 가장 높았던 날과 설비는 ( ) 이고, 그날 ( ) 조건이 평소와 달랐음.
2. **무엇이**: 불량에 영향이 큰 조건은 ( ), ( ), ( ) 순이었음.
3. **그래서**: 우리 라인이라면 ( ) 값을 ( ) 범위로 관리하고, ( ) 를 대시보드에 띄우겠음.

저장된 파일: `data/session2/work/model_result.csv`, `data/session2/work/dashboard.png`
'''),
]

build("2-1_data_cleaning.ipynb", nb21)
build("2-2_analysis_dashboard.ipynb", nb22)
