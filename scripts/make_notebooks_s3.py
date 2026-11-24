"""3회차 실습 노트북(3-0, 3-1, 3-2)을 만듦.  실행: .venv/bin/python scripts/make_notebooks_s3.py"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "session3" / "notebooks"


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
import sys, time
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

DATA = ROOT / "data" / "session3"
OUT = DATA / "work"                        # 내 산출물이 저장되는 곳
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
'''

# ============================================================================
# 3-0 조건 추천 (시연)
# ============================================================================
nb30 = [
("md", '''
# 시연 3-0. 분류 모델로 성형 조건 추천하기

2회차에서 만든 불량 분류 모델을 **거꾸로** 씀: "이 조건이면 불량일까?" 대신 "불량이 안 나는 조건 범위는 어디일까?"

- 조건 하나만 바꿔 가며(나머지는 기준값) 모델이 예측하는 불량 확률을 구함
- 예측 불량 확률이 5% 아래인 구간을 **권장 조건 범위**로 삼음
- 결과를 저장하면 가상 공장의 `추천 조건 적용` 버튼이 이 파일을 읽음
'''),
("code", SETUP),
("code", '''
from sklearn.ensemble import RandomForestClassifier
from common import process_model as pm
from common.recommend import CONDITIONS, recommend, sweep

S2 = ROOT / "data" / "session2"
src = S2 / "work" / "injection_clean.csv"
if not src.exists():
    src = S2 / "injection_clean_provided.csv"
df = pd.read_csv(src)
KO = {"mold_temp_c": "금형온도 (℃)", "melt_temp_c": "수지온도 (℃)", "inj_pressure_mpa": "사출압 (MPa)",
      "hold_pressure_mpa": "보압 (MPa)", "hold_time_s": "보압 시간 (s)", "cooling_time_s": "냉각 시간 (s)"}
model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1)
model.fit(df[CONDITIONS], df["is_defect"])
print(src.name, df.shape)
'''),
("md", '''
## 1. 조건을 하나씩 바꿔 보기

회색 띠가 예측 불량 확률 5% 아래인 구간(권장 범위), 점선이 현재 기준값임.
'''),
("code", '''
LIMIT = 0.05                              # 허용하는 예측 불량 확률
base = dict(pm.INJ_NOMINAL)               # 나머지 조건을 고정해 둘 값
reco = recommend(model, df, base, LIMIT)
fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
for ax, key in zip(axes.ravel(), CONDITIONS):
    grid, p = sweep(model, df, base, key)
    ax.plot(grid, p * 100, color="#c0392b", lw=2)
    ax.axvspan(reco[key]["low"], reco[key]["high"], color="#2e86c1", alpha=0.15)
    ax.axvline(base[key], color="gray", ls="--", lw=1)
    ax.axhline(LIMIT * 100, color="gray", lw=0.8)
    ax.set_title(KO[key]); ax.set_ylabel("예측 불량 확률 (%)"); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUT / "recommend.png", dpi=120); plt.show()
'''),
("md", '''
## 2. 권장 조건 범위
'''),
("code", '''
table = pd.DataFrame(reco).T.rename(columns={"low": "하한", "high": "상한", "recommend": "추천값"})
table.insert(0, "기준값", [base[k] for k in table.index])
table.index = [KO[k] for k in table.index]
(OUT / "recommend.json").write_text(__import__("json").dumps(reco, ensure_ascii=False, indent=1))
table
'''),
("md", '''
## 3. 가상 공장에서 확인

1. 가상 공장 패널의 `현장 변동 (금형온도 · 보압 저하)` → 불량이 늘어나는 것을 확인
2. `추천 조건 적용` → 위 표의 추천값이 슬라이더에 들어감 → 불량이 줄어드는 것을 확인

> 모델은 **데이터가 있었던 범위 안에서만** 믿을 수 있음. 그래서 범위를 데이터의 1~99% 구간으로 제한했음.
> 추천은 "권고"임. 설비에 바로 쓰기 전에 공정 담당자가 확인하는 단계가 필요함 (4회차의 제어 3수준).
'''),
]

# ============================================================================
# 3-1 외관 이상탐지
# ============================================================================
nb31 = [
("md", '''
# 실습 3-1. 도장 외관 이상탐지

가상 공장의 **도장 외관 검사 셀**에서 찍은 패널 이미지로, **정상 이미지만 학습해서** 결함을 찾음.

- 불량 이미지를 모으기 어려운 현장에서 쓰는 방법임 (정상은 많고 불량은 드묾)
- 원리: 정상 이미지의 화소별 평균과 흔들림(표준편차)을 기억해 두고, 새 이미지가 거기서 얼마나 벗어났는지 봄
- 조건: 조명 · 배경 · 위치가 **고정**되어 있어야 함

- **기본 트랙**: 셀을 차례로 실행하고 임계값별 과검 · 미검 표를 워크북에 옮겨 적음
- **심화 트랙**: `# 심화` 표시가 있는 값을 바꿔 AUROC 0.90 이상을 확인하고 과검 · 미검의 균형점을 정함
'''),
("code", SETUP),
("md", '''
## 1. 데이터 확인

| 구분 | 장수 | 내용 |
|---|---|---|
| train | 240 | 양품만 (학습용) |
| test | 120 | 양품 60 + 결함 60 (핀홀 · 흐름 · 이물 · 색차 각 15) |
| shift | 30 | 양품이지만 촬영 조건이 바뀐 것 (조명 각도 · 밝기 · 배경판) |
'''),
("code", '''
images = np.load(DATA / "paint_images.npz")["images"]          # [390, 192, 192, 3]
labels = pd.read_csv(DATA / "paint_labels.csv")
KIND_KO = {"none": "양품", "pinhole": "핀홀", "sag": "흐름", "dust": "이물", "tint": "색차"}
print(images.shape)
labels.groupby(["split", "kind"]).size().unstack(fill_value=0)
'''),
("code", '''
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
for ax, kind in zip(axes, KIND_KO):
    i = labels[(labels.split == "test") & (labels.kind == kind)].index[0]
    ax.imshow(images[i]); ax.set_title(KIND_KO[kind]); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
("md", '''
## 2. 정상 학습

양품 240장의 화소별 평균과 표준편차를 구함. 이것이 "정상의 모습"임.
'''),
("code", '''
from common.anomaly import PixelModel, prepare

t0 = time.time()
X = np.stack([prepare(im) for im in images])                 # 96 x 96 으로 줄임
is_train = (labels.split == "train").values
model = PixelModel().fit(X[is_train])
print(f"학습 시간 {time.time() - t0:.1f}초  (목표: 노트북 CPU 5분 이내)")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
axes[0].imshow(model.mean); axes[0].set_title("정상의 평균 모습"); axes[0].axis("off")
im = axes[1].imshow(model.std.max(axis=-1), cmap="viridis"); axes[1].set_title("정상에서도 흔들리는 곳"); axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046); plt.tight_layout(); plt.show()
'''),
("md", '''
> 패널 가장자리는 시편 위치가 조금씩 달라서 정상에서도 흔들림. 그런 곳은 조금 달라도 이상으로 보지 않음.

## 3. 결함 영역 검출

새 이미지가 정상에서 벗어난 정도를 화소마다 구해 지도로 그림.
'''),
("code", '''
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for j, kind in enumerate(KIND_KO):
    i = labels[(labels.split == "test") & (labels.kind == kind)].index[1]
    amap = model.anomaly_map(X[i])
    axes[0, j].imshow(images[i]); axes[0, j].set_title(KIND_KO[kind]); axes[0, j].axis("off")
    axes[1, j].imshow(amap, cmap="inferno", vmin=0, vmax=10)
    axes[1, j].set_title(f"이상 점수 {model.score(X[i]):.1f}"); axes[1, j].axis("off")
plt.tight_layout(); fig.savefig(OUT / "anomaly_maps.png", dpi=120); plt.show()
'''),
("md", '''
## 4. 이상 점수와 AUROC

이미지 한 장의 이상 점수 = 지도에서 가장 높은 부분의 평균. 양품과 결함의 점수가 얼마나 갈리는지 봄.
'''),
("code", '''
from sklearn.metrics import roc_auc_score

labels["score"] = [model.score(x) for x in X]
test = labels[labels.split == "test"]
auroc = roc_auc_score(test["is_defect"], test["score"])
print(f"AUROC {auroc:.3f}   (목표 0.90 이상)")

fig, ax = plt.subplots(figsize=(10, 3.4))
bins = np.linspace(0, 13, 53)
for kind, color in zip(KIND_KO, ["#2471a3", "#7d3c98", "#d68910", "#566573", "#1e8449"]):
    ax.hist(test.loc[test.kind == kind, "score"], bins=bins, alpha=0.75, label=KIND_KO[kind], color=color)
ax.set_xlabel("이상 점수"); ax.set_ylabel("장수"); ax.legend(ncol=5); ax.set_title("시험 이미지의 이상 점수 분포")
plt.tight_layout(); fig.savefig(OUT / "score_hist.png", dpi=120); plt.show()
test.groupby("kind")["score"].agg(["min", "median", "max"]).round(1).rename(index=KIND_KO)
'''),
("md", '''
> 핀홀 · 흐름은 점수가 뚜렷하게 높음. **색차**는 양품과 거의 붙어 있음 → 임계값을 어디에 두느냐에 따라 잡히기도 놓치기도 함.

## 5. 임계값과 과검 · 미검
'''),
("code", '''
rows = []
for th in [1.0, 1.5, 2.0, 2.2, 2.3, 2.4, 2.5, 3.0, 4.0, 6.0, 8.0]:
    ng = test["score"] >= th
    rows.append({"임계값": th, "과검(양품→불량)": int((ng & (test.is_defect == 0)).sum()),
                 "미검(불량→양품)": int((~ng & (test.is_defect == 1)).sum())})
result = pd.DataFrame(rows).set_index("임계값")
result["과검률(%)"] = (result["과검(양품→불량)"] / (test.is_defect == 0).sum() * 100).round(1)
result["미검률(%)"] = (result["미검(불량→양품)"] / (test.is_defect == 1).sum() * 100).round(1)
result.to_csv(OUT / "anomaly_result.csv", encoding="utf-8-sig")
result
'''),
("code", '''
THRESHOLD = 2.5                         # 심화: 과검과 미검의 균형점을 직접 정함
ng = test["score"] >= THRESHOLD
print(f"임계값 {THRESHOLD}:  과검 {int((ng & (test.is_defect == 0)).sum())}장,  미검 {int((~ng & (test.is_defect == 1)).sum())}장")
(test.assign(검출=ng).groupby("kind")["검출"].mean() * 100).round(0).rename(index=KIND_KO).rename("불량으로 판정한 비율(%)").to_frame()
'''),
("md", '''
> 임계값을 낮추면 색차까지 잡지만 양품을 불량으로 보내는 과검이 늘고, 높이면 과검은 줄지만 미검이 늚.
> 가상 공장의 임계값 슬라이더로 같은 현상을 확인함.

## 6. 촬영 조건이 바뀌면

같은 **양품**을 조명 각도 · 밝기 · 배경판을 바꿔 찍은 30장임. 결함이 없는데도 점수가 어떻게 되는지 봄.
'''),
("code", '''
shift = labels[labels.split == "shift"].copy()
COND_KO = {'{"angle": 25.0}': "조명 각도 45→25도", '{"level": 1.5}': "조명 밝기 1.5배", '{"backdrop": false}': "배경판 없음"}
shift["조건"] = shift["condition"].map(COND_KO)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
i0 = labels[(labels.split == "test") & (labels.kind == "none")].index[0]
axes[0].imshow(images[i0]); axes[0].set_title(f"기준 조건  (점수 {labels.score[i0]:.1f})"); axes[0].axis("off")
for ax, (name, g) in zip(axes[1:], shift.groupby("조건", sort=False)):
    ax.imshow(images[g.index[0]]); ax.set_title(f"{name}  (점수 {g.score.iloc[0]:.1f})"); ax.axis("off")
plt.tight_layout(); fig.savefig(OUT / "shift.png", dpi=120); plt.show()
out = shift.groupby("조건", sort=False)["score"].agg(["median", "min", "max"]).round(1)
out["불량 판정(장)"] = shift.groupby("조건", sort=False)["score"].apply(lambda s: int((s >= THRESHOLD).sum()))
out
'''),
("md", '''
> 결함이 하나도 없는데 대부분 불량으로 판정됨. 이 방법은 **촬영 조건을 고정**해야만 쓸 수 있음.
> 현장에서 조명을 교체하거나 지그 위치를 옮겼다면 정상 이미지를 다시 모아 **재학습**해야 함.

## 7. 저장
'''),
("code", '''
model.save(OUT / "pixel_model.npz")       # 가상 공장의 "모델 다시 읽기" 버튼이 이 파일을 읽음
print("저장:", [p.name for p in sorted(OUT.glob("*")) if p.name.startswith(("anomaly", "pixel", "score", "shift"))])
'''),
("md", '''
### 워크북에 옮겨 적기

1. AUROC 값과 학습 시간
2. 임계값별 과검 · 미검 표 (5절) 와 내가 정한 임계값
3. 색차를 잡으려면 임계값 외에 무엇을 바꿔야 할지 한 문장

### 심화

- `PixelModel` 의 표준편차 하한(0.012)을 바꾸면: `model.std = X[is_train].std(axis=0) + 0.005` 후 4절부터 다시 실행
- 정상 이미지를 60장만 써서 학습하면 AUROC 와 과검이 어떻게 달라지는지: `PixelModel().fit(X[is_train][:60])`
'''),
]

# ============================================================================
# 3-2 예지보전
# ============================================================================
nb32 = [
("md", '''
# 실습 3-2. 유압 펌프 예지보전

사출기 유압 펌프 4대의 **진동**과 **전류**를 30일 동안 2시간마다 모은 데이터임. 그중 3대가 베어링 고장으로 멈췄음.
고장 전에 나타나는 **전조**를 찾아 경고 · 정지 2단 알람 기준을 만듦.

진행 순서는 상태 감시 표준(ISO 13374)의 여섯 단계를 따름:

| 단계 | 이 노트북에서 |
|---|---|
| 1. 데이터 수집 | 진동 · 전류 불러오기 (1절) |
| 2. 데이터 처리 | 특징 추출: 실효값 · 첨도 · 파고율 (2절) |
| 3. 상태 검출 | 정상 기준과 이상 점수 (3절) |
| 4. 건전성 평가 | 전조를 얼마나 잘 가려내는가 (4절) |
| 5. 예지 평가 | 고장 몇 시간 전에 알 수 있는가 (5절) |
| 6. 조치 권고 | 경고 · 정지 알람 기준표 (6절) |

- **기본 트랙**: 셀을 차례로 실행하고 알람 기준표를 워크북에 옮겨 적음
- **심화 트랙**: `# 심화` 표시가 있는 값을 바꿔 전조 재현율 0.80 이상을 지키면서 오경보를 줄임
'''),
("code", SETUP),
("md", '''
## 1. 데이터 수집
'''),
("code", '''
from common import process_model as pm

win = pd.read_csv(DATA / "pump_windows.csv", parse_dates=["timestamp"])
vib = np.load(DATA / "pump_vibration.npz")["vibration"].astype(np.float32)      # [구간 수, 1024] 가속도(g), 2 kHz
fail = pd.read_csv(DATA / "failure_log.csv", parse_dates=["failure_time"])
FS = 2000
print("구간:", vib.shape, "  설비:", list(win.unit_id.unique()))
fail
'''),
("code", '''
# 같은 설비의 초기와 고장 직전 파형
u = win[win.unit_id == "PUMP-03"]
t_fail = fail.set_index("unit_id").loc["PUMP-03", "failure_time"]
early, late = u.index[5], u[u.timestamp < t_fail].index[-3]
t = np.arange(vib.shape[1]) / FS * 1000
fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), sharey=True)
for ax, i, name in ((axes[0], early, "초기 (정상)"), (axes[1], late, "고장 6시간 전")):
    ax.plot(t[:600], vib[i][:600], lw=0.8, color="#1f618d"); ax.set_title(f"PUMP-03  {name}"); ax.set_xlabel("시간 (ms)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("가속도 (g)"); plt.tight_layout(); fig.savefig(OUT / "pump_wave.png", dpi=120); plt.show()
'''),
("md", '''
> 고장이 가까워지면 뾰족한 충격이 주기적으로 나타남. 닳은 베어링 위를 구름체가 지날 때마다 생기는 충격임.

## 2. 데이터 처리: 특징 추출

파형 1,024개 점을 숫자 몇 개로 줄임.

| 특징 | 뜻 | 잘 잡는 것 |
|---|---|---|
| 실효값 (RMS) | 진동의 평균적인 크기 | 전체적으로 커진 진동 (후기) |
| 첨도 | 뾰족한 충격이 섞인 정도 | 초기 베어링 결함 |
| 파고율 | 최댓값 ÷ 실효값 | 충격 |
'''),
("code", '''
feat = pd.DataFrame(pm.vib_features(vib))
df = pd.concat([win, feat], axis=1)
fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
for ax, col, name in zip(axes, ["rms", "kurtosis", "current_rms_a"], ["진동 실효값 (g)", "진동 첨도", "모터 전류 (A)"]):
    for unit, g in df.groupby("unit_id"):
        ax.plot(g.timestamp, g[col].rolling(3, min_periods=1).median(), lw=1.2, label=unit)
    for _, r in fail.iterrows():
        ax.axvline(r.failure_time, color="crimson", ls="--", lw=1)
    ax.set_ylabel(name); ax.grid(alpha=0.3)
axes[0].legend(ncol=4, loc="upper left"); axes[0].set_title("특징의 추이 (빨간 점선: 고장)")
plt.tight_layout(); fig.savefig(OUT / "pump_trend.png", dpi=120); plt.show()
'''),
("md", '''
> 첨도가 가장 먼저 오르고, 실효값이 그 뒤를 따르고, 전류는 고장 직전에야 조금 오름. **무엇을 재느냐에 따라 알 수 있는 시점이 다름.**

## 3. 상태 검출: 정상 기준과 이상 점수

설비마다 처음 5일을 정상으로 보고 기준(평균 · 표준편차)을 잡음. 이상 점수 = 기준에서 벗어난 정도.
'''),
("code", '''
BASE_DAYS = 5                               # 심화: 기준을 잡는 기간
KEYS = ["rms", "kurtosis", "crest"]         # 심화: 점수에 쓰는 특징
df["score"] = 0.0
for unit, g in df.groupby("unit_id"):
    base = g[g.timestamp < g.timestamp.min() + pd.Timedelta(days=BASE_DAYS)]
    hs = pm.HealthScore(KEYS).fit({k: base[k].values for k in KEYS})
    df.loc[g.index, "score"] = hs.score({k: g[k].values for k in KEYS})

fig, ax = plt.subplots(figsize=(13, 3.6))
for unit, g in df.groupby("unit_id"):
    ax.plot(g.timestamp, g.score, lw=1, label=unit)
for _, r in fail.iterrows():
    ax.axvline(r.failure_time, color="crimson", ls="--", lw=1)
ax.set_ylim(0, 40); ax.set_ylabel("이상 점수"); ax.legend(ncol=4); ax.grid(alpha=0.3)
ax.set_title("이상 점수의 추이"); plt.tight_layout(); fig.savefig(OUT / "pump_score.png", dpi=120); plt.show()
'''),
("md", '''
> 고장 없이 잘 돌던 구간에도 점수가 튀는 점이 있음. 지게차 충돌 같은 일시적인 충격임. 이것이 **오경보**의 재료가 됨.
> 같은 열화라도 설비마다 점수의 크기가 다름. 기준을 잡은 5일 동안 충격이 섞였던 설비는 기준이 느슨해져 점수가 낮게 나옴.

## 4. 건전성 평가: 전조를 가려내는가

고장 기록으로 구간마다 "고장까지 남은 시간"을 계산하고, 전조(고장 전 120시간 이내)와 정상(고장 300시간 이전, 고장 없는 설비)을 나눔.
'''),
("code", '''
from sklearn.metrics import roc_auc_score

ft = fail.set_index("unit_id")["failure_time"]
df["to_fail_h"] = [(ft[u] - t).total_seconds() / 3600 if (u in ft.index and t < ft[u]) else np.nan
                   for u, t in zip(df.unit_id, df.timestamp)]
PRECURSOR_H = 120
is_pre = df.to_fail_h <= PRECURSOR_H
is_normal = df.to_fail_h.isna() | (df.to_fail_h > 300)
ev = df[is_pre | is_normal]
auroc = roc_auc_score(is_pre[ev.index], ev.score)
print(f"전조 구간 {is_pre.sum()}개,  정상 구간 {is_normal.sum()}개")
print(f"이상탐지 AUROC {auroc:.3f}   (목표 0.90 이상)")
'''),
("md", '''
## 5. 예지 평가: 알람 기준과 선행 시간

- **경고**: 점검을 계획함 (부품 준비, 정비 일정)
- **정지**: 설비를 세움
- 오경보 억제: 기준을 **연속으로 몇 번** 넘어야 알람을 낼지 정함
'''),
("code", '''
WARN, STOP = 5.0, 15.0          # 심화: 경고 · 정지 기준 점수
N_CONSEC = 3                    # 심화: 연속 몇 번 넘으면 알람 (1 이면 한 번만 넘어도 알람)

def alarms(score, th, n):
    """기준을 n 번 연속으로 넘은 지점이 True"""
    over = (score >= th).astype(int)
    return over.rolling(n).sum() >= n

rows, false_warn = [], 0
for unit, g in df.groupby("unit_id"):
    warn, stop = alarms(g.score, WARN, N_CONSEC), alarms(g.score, STOP, N_CONSEC)
    false_warn += int((warn & is_normal[g.index]).sum())
    if unit in ft.index:
        before = g.timestamp < ft[unit]
        w, s = g[warn & before & (g.to_fail_h < 250)], g[stop & before & (g.to_fail_h < 250)]
        rows.append({"설비": unit, "경고 (고장 몇 시간 전)": round(w.to_fail_h.max(), 0) if len(w) else None,
                     "정지 (고장 몇 시간 전)": round(s.to_fail_h.max(), 0) if len(s) else None})
lead = pd.DataFrame(rows).set_index("설비")
recall = (alarms(df.score, WARN, 1) & is_pre).sum() / is_pre.sum()
print(f"전조 재현율 {recall:.2f}   (전조 구간 중 경고 기준을 넘은 비율, 목표 0.80 이상)")
print(f"정상 구간에서의 오경보 {false_warn}회   (연속 {N_CONSEC}회 기준)")
lead
'''),
("code", '''
# 연속 횟수에 따른 오경보
cmp = []
for n in (1, 2, 3, 5):
    fw = sum(int((alarms(g.score, WARN, n) & is_normal[g.index]).sum()) for _, g in df.groupby("unit_id"))
    first = [g[alarms(g.score, WARN, n) & (g.timestamp < ft[u]) & (g.to_fail_h < 250)].to_fail_h.max()
             for u, g in df.groupby("unit_id") if u in ft.index]
    cmp.append({"연속 횟수": n, "정상 구간 오경보(회)": fw, "평균 경고 선행 시간(h)": round(float(np.nanmean(first)), 0)})
pd.DataFrame(cmp).set_index("연속 횟수")
'''),
("md", '''
> 한 번만 넘어도 알람을 내면 충격 때마다 오경보가 남. 연속 횟수를 늘리면 오경보는 사라지지만 알람이 그만큼 늦어짐.

## 6. 조치 권고: 알람 기준표
'''),
("code", '''
rules = pd.DataFrame([
    {"단계": "정상", "기준": f"이상 점수 {WARN} 미만", "조치": "정상 운전, 2시간마다 측정"},
    {"단계": "경고", "기준": f"이상 점수 {WARN} 이상이 연속 {N_CONSEC}회", "조치": "측정 주기 단축, 베어링 준비, 다음 비가동 시간에 점검",
     "평균 선행 시간(h)": round(float(lead.iloc[:, 0].mean()), 0)},
    {"단계": "정지", "기준": f"이상 점수 {STOP} 이상이 연속 {N_CONSEC}회", "조치": "설비 정지, 베어링 교체",
     "평균 선행 시간(h)": round(float(lead.iloc[:, 1].mean()), 0)},
]).set_index("단계")
rules.to_csv(OUT / "alarm_rules.csv", encoding="utf-8-sig")
rules
'''),
("md", '''
### 워크북에 옮겨 적기

1. AUROC 와 전조 재현율
2. 알람 기준표 (단계 · 기준 · 조치 · 평균 선행 시간)
3. 오경보가 났을 때 현장에서 어떻게 처리할지 한 문장

### 심화

- `KEYS` 에서 첨도를 빼고 실효값만 쓰면 경고 선행 시간이 얼마나 줄어드는지
- `WARN` 을 3 으로 낮추면 선행 시간과 오경보가 어떻게 달라지는지
- 전류(`current_rms_a`)만으로 점수를 만들면 몇 시간 전에 알 수 있는지

저장된 파일: `data/session3/work/alarm_rules.csv`
'''),
]

build("3-0_condition_recommend.ipynb", nb30)
build("3-1_visual_anomaly.ipynb", nb31)
build("3-2_predictive_maintenance.ipynb", nb32)
