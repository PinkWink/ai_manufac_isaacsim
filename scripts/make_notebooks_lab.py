"""공정 실험 노트북(2-3, 3-3)을 만듦: 수강생이 공정 모델을 직접 돌려 보는 실습.  실행: .venv/bin/python scripts/make_notebooks_lab.py

가상 공장(강사 PC)과 같은 공정 모델(common/process_model.py)을 노트북에서 그대로 씀. GPU 가 필요 없음.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(session, name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    out = ROOT / f"session{session}" / "notebooks" / name
    nbf.write(nb, out)
    print("wrote", out)


def setup(session):
    return f'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "common").exists():      # 저장소 최상위 폴더 찾기
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import process_model as pm
from common.plot_style import use_korean_font
use_korean_font()

DATA = ROOT / "data" / "session{session}"
OUT = DATA / "work"                        # 내 산출물이 저장되는 곳
OUT.mkdir(exist_ok=True)
LIVE = DATA / "live"                       # 강사 PC 에서 받은 데이터 (./scripts/factory fetch <주소>)
pd.set_option("display.width", 160)
'''


# ============================================================================
# 2-3 공정 실험: 성형 조건을 바꾸면 불량이 어떻게 달라지나
# ============================================================================
nb23 = [
("md", '''
# 실습 2-3. 공정 실험: 조건을 바꿔 사출기를 돌려 보기

강사 화면의 가상 공장은 **공정 모델**이 계산한 결과를 3D 로 보여 줍니다. 이 노트북은 그 공정 모델을 그대로 불러와서, 내 PC 에서 사출기를 직접 돌려 봅니다.

- 조건(금형온도 · 보압 · 냉각 시간 …)을 **내가 정하고**, 샷을 만들고, 불량률을 확인합니다.
- `# 내 값` 표시가 있는 줄을 고친 뒤 그 셀부터 아래로 다시 실행합니다.
- 실행: 셀을 선택하고 `Shift` + `Enter`
'''),
("code", setup(2) + '''
KO = dict(mold_temp_c="금형온도 (℃)", melt_temp_c="수지온도 (℃)", inj_pressure_mpa="사출압 (MPa)",
          hold_pressure_mpa="보압 (MPa)", hold_time_s="보압 시간 (s)", cooling_time_s="냉각 시간 (s)")
print("기준 조건")
for k, v in pm.INJ_NOMINAL.items():
    print(f"  {KO[k]:<14} {v}")
'''),
("md", '''
## 1. 기준 조건으로 300샷

`run(조건, 샷 수)` 는 사출기 한 대를 그 조건으로 돌려 샷 기록을 표로 돌려줍니다. 샷마다 값이 조금씩 흔들리는 것(설비 변동)도 들어 있습니다.
'''),
("code", '''
def run(setpoint=None, n=300, seed=1):
    """설정값을 바꾼 사출기로 n 샷을 만들어 표로 돌려줌."""
    m = pm.InjectionMachine(seed=seed)
    m.setpoint.update(setpoint or {})
    df = pd.DataFrame([m.next_shot().as_dict() for _ in range(n)])
    df["is_defect"] = (df["defect"] != "none").astype(int)
    return df

base = run()
print(f"샷 {len(base)}개, 불량률 {base['is_defect'].mean():.1%}, 평균 사이클타임 {base['cycle_time_s'].mean():.1f}초")
print(base["defect"].map(pm.DEFECT_KO).value_counts().to_string())
base.head()
'''),
("md", '''
## 2. 조건 하나를 바꿔 보기

2회차 2교시에 강사 화면에서 본 "금형 온조기 이상"을 직접 재현합니다. 금형온도 설정을 60 → 54 ℃ 로 낮춥니다.
'''),
("code", '''
MY = dict(mold_temp_c=54.0)          # 내 값: 바꿀 조건과 값. 예) dict(cooling_time_s=24.0), dict(hold_pressure_mpa=50.0)

mine = run(MY)
cmp = pd.DataFrame({"기준 조건": base["defect"].map(pm.DEFECT_KO).value_counts(normalize=True),
                    "내 조건": mine["defect"].map(pm.DEFECT_KO).value_counts(normalize=True)}).fillna(0).drop("양품", errors="ignore")
ax = (cmp * 100).plot.bar(figsize=(7, 3.6), rot=0, color=["#8da0cb", "#fc8d62"])
ax.set_xlabel(""); ax.set_ylabel("발생 비율 (%)"); ax.set_title(f"조건 변경: {', '.join(f'{KO[k]} {v}' for k, v in MY.items())}")
plt.tight_layout(); plt.show()
print(f"불량률  기준 {base['is_defect'].mean():.1%}  →  내 조건 {mine['is_defect'].mean():.1%}")
'''),
("md", '''
## 3. 한 조건을 훑어 보기

조건 하나를 범위 전체로 바꿔 가며 결함별 발생 확률을 그립니다. 어느 값부터 불량이 늘기 시작하는지(**관리 한계**)를 찾습니다.
'''),
("code", '''
def defect_rate(setpoint, n=3000, seed=0):
    """그 설정값에서의 결함별 발생 확률 (샷 변동 포함)."""
    rng = np.random.default_rng(seed)
    sp = dict(pm.INJ_NOMINAL); sp.update(setpoint)
    v = {k: sp[k] + rng.normal(0, pm.INJ_NOISE[k], n) for k in pm.INJ_NOMINAL}
    p = pm.injection_defect_probs(**v)
    out = {pm.DEFECT_KO[k]: float(p[k].mean()) for k in pm.INJ_DEFECTS}
    out["전체"] = float(1 - ((1 - p["flow_mark"]) * (1 - p["sink"]) * (1 - p["short_shot"])).mean())
    return out

KEY = "mold_temp_c"                  # 내 값: 훑을 조건
lo, hi = pm.INJ_RANGE[KEY]
grid = np.linspace(lo, hi, 41)
sw = pd.DataFrame([defect_rate({KEY: g}) for g in grid], index=grid) * 100
ax = sw.plot(figsize=(8, 3.8), lw=2)
ax.axvline(pm.INJ_NOMINAL[KEY], color="gray", ls=":"); ax.axhline(3, color="red", ls="--", lw=1)
ax.set_xlabel(KO[KEY]); ax.set_ylabel("발생 확률 (%)"); ax.set_title("점선: 기준 값, 빨간 선: 불량률 3%")
plt.tight_layout(); plt.show()
ok = sw.index[sw["전체"] <= 3.0]
print(f"불량률 3% 이하인 범위: {ok.min():.1f} ~ {ok.max():.1f}  (기준 {pm.INJ_NOMINAL[KEY]})")
'''),
("md", '''
## 4. 두 조건을 함께 보기

조건 두 개를 격자로 바꿔 불량률 지도를 그립니다. 밝은 영역이 안전한 조건입니다.
'''),
("code", '''
KX, KY = "cooling_time_s", "hold_pressure_mpa"       # 내 값: 가로 · 세로 조건
gx, gy = np.linspace(*pm.INJ_RANGE[KX], 25), np.linspace(*pm.INJ_RANGE[KY], 25)
Z = np.array([[defect_rate({KX: x, KY: y}, n=600)["전체"] for x in gx] for y in gy]) * 100
fig, ax = plt.subplots(figsize=(7, 4.6))
im = ax.pcolormesh(gx, gy, Z, cmap="RdYlGn_r", vmin=0, vmax=30, shading="auto")
ax.contour(gx, gy, Z, levels=[3], colors="k", linewidths=1.5)
ax.plot(pm.INJ_NOMINAL[KX], pm.INJ_NOMINAL[KY], "k*", ms=14)
ax.set_xlabel(KO[KX]); ax.set_ylabel(KO[KY]); ax.set_title("불량률 (%) · 검은 선: 3% 경계 · 별: 기준 조건")
fig.colorbar(im); plt.tight_layout(); plt.show()
'''),
("md", '''
## 5. 미션: 사이클타임을 줄여 생산량 늘리기

생산 관리자가 "사이클타임을 줄여 달라"고 합니다. 냉각 시간 · 보압 시간을 줄이면 사이클타임은 짧아지지만 수축 불량이 늘어납니다.

- 목표: **불량률 3% 이하**를 지키면서 **시간당 양품 수**를 가장 크게
- `MY_BEST` 를 바꿔 가며 아래 셀을 다시 실행합니다.
'''),
("code", '''
MY_BEST = dict(cooling_time_s=28.0, hold_time_s=8.0)      # 내 값

def evaluate(sp, n=2000):
    df = run(sp, n=n, seed=7)
    rate, ct = df["is_defect"].mean(), df["cycle_time_s"].mean()
    return dict(불량률=rate, 사이클타임_초=ct, 시간당_양품=3600 / ct * (1 - rate))

rows = {"기준 조건": evaluate({}), "내 조건": evaluate(MY_BEST)}
res = pd.DataFrame(rows).T
res["판정"] = np.where(res["불량률"] <= 0.03, "합격", "불량률 초과")
display(res.style.format({"불량률": "{:.1%}", "사이클타임_초": "{:.1f}", "시간당_양품": "{:.1f}"}))
pd.DataFrame([dict(MY_BEST, **rows["내 조건"])]).to_csv(OUT / "process_lab_result.csv", index=False)
print("저장:", (OUT / "process_lab_result.csv").relative_to(ROOT))
'''),
("md", '''
## 6. 강사 PC 의 가상 공장 데이터 받아 보기

강사 화면의 가상 공장이 방금 만든 기록을 내 PC 로 받아 같은 방식으로 봅니다. 터미널에서 먼저 실행합니다 (주소는 강사 화면의 "수강생 연결" 구역에 있음).

```bash
cd ~/ai_manufac
./scripts/factory fetch <강사 PC 주소>
```
'''),
("code", '''
shots = sorted(LIVE.glob("shots_*.csv"))
if not shots:
    print("받은 파일이 없습니다. 위의 fetch 명령을 먼저 실행하세요. (없어도 다음 실습에는 지장 없음)")
else:
    s = pd.read_csv(shots[-1])
    ins = pd.read_csv(LIVE / shots[-1].name.replace("shots_", "inspection_"))
    print(f"{shots[-1].name}: 샷 {len(s)}행, 검사 {len(ins)}행, 검사 불량률 {(ins['result'] != 'OK').mean():.1%}")
    cond = list(pm.INJ_NOMINAL)
    live_mean = s[cond].apply(pd.to_numeric, errors="coerce").mean()
    print("이 조건을 공정 모델에 넣었을 때의 기대 불량률:", f"{defect_rate(live_mean.to_dict())['전체']:.1%}")
    display(pd.DataFrame({"가상 공장 평균": live_mean.round(1), "기준 조건": pd.Series(pm.INJ_NOMINAL)}).rename(index=KO))
'''),
("md", '''
## 정리

- 조건을 바꾸면 **어떤 결함이** 늘어나는지가 조건마다 다릅니다 (금형온도 ↓ → 흐름자국, 냉각 시간 ↓ → 수축).
- 불량은 어느 값부터 갑자기 늘어납니다. 그 경계가 **관리 한계**이고, 3회차의 조건 추천이 찾는 것도 이 경계입니다.
- 사이클타임과 불량률은 서로 맞바꾸는 관계입니다. 어디까지 줄일지는 데이터로 정합니다.
'''),
]

# ============================================================================
# 3-3 예지보전 실험: 펌프를 직접 닳게 해 보고 알람 기준 정하기
# ============================================================================
nb33 = [
("md", '''
# 실습 3-3. 예지보전 실험: 펌프를 직접 돌려 알람 기준 정하기

강사 화면의 유압 유닛은 **펌프 모델**이 만든 진동으로 움직입니다. 이 노트북은 그 모델을 불러와, 내 PC 에서 펌프 한 대를 수명 끝까지 돌려 봅니다.

- 알람 기준(경고 · 정지 · 연속 횟수)을 **내가 정하고**, 얼마나 일찍 알았는지와 오경보가 몇 번인지 확인합니다.
- `# 내 값` 표시가 있는 줄을 고친 뒤 그 셀부터 아래로 다시 실행합니다.
'''),
("code", setup(3) + '''
t = np.arange(pm.PUMP_N) / pm.PUMP_FS
print(f"한 구간: {pm.PUMP_N}개 표본 / {pm.PUMP_FS} Hz = {pm.PUMP_N / pm.PUMP_FS:.2f}초,  축 회전 {pm.SHAFT_HZ} Hz,  베어링 결함 주파수 {pm.BPFO_HZ} Hz")
'''),
("md", '''
## 1. 정상 펌프와 닳은 펌프의 진동

수명 1,500시간인 펌프를 운전 시간만 바꿔서 한 구간씩 측정합니다.
'''),
("code", '''
AGES = [300, 1350, 1450]             # 내 값: 운전 시간 (h)

fig, axes = plt.subplots(len(AGES), 1, figsize=(9, 2.0 * len(AGES)), sharex=True, sharey=True)
rows = []
for ax, age in zip(axes, AGES):
    w = pm.PumpUnit(seed=5, life_h=1500, age_h=age).window("bearing")
    f = pm.vib_features(w)
    rows.append(dict(운전시간_h=age, 실효값=f["rms"], 첨도=f["kurtosis"], 파고율=f["crest"], 최댓값=f["peak"]))
    ax.plot(t, w, lw=0.6); ax.set_ylabel("가속도 (g)"); ax.set_title(f"운전 {age}시간 (고장까지 {1500 - age}시간)", fontsize=10)
axes[-1].set_xlabel("시간 (초)"); plt.tight_layout(); plt.show()
pd.DataFrame(rows).astype(float).round(2)
'''),
("md", '''
## 2. 펌프 한 대를 수명 끝까지

2시간마다 한 구간씩 측정하며 고장까지 돌립니다. 처음 300시간을 **정상 기준**으로 삼아 이상 점수를 계산합니다. 중간에 지게차 충돌 같은 **일시 충격**도 가끔 들어옵니다.
'''),
("code", '''
def run_life(sensor="bearing", life_h=1500, step_h=2, shock_rate=0.01, seed=21):
    """펌프 한 대의 수명 전체: 구간별 특징과 이상 점수."""
    rng = np.random.default_rng(seed)
    pump = pm.PumpUnit(seed=seed, life_h=life_h)
    rows = []
    while not pump.failed:
        shock = bool(rng.random() < shock_rate)
        f = pm.vib_features(pump.window(sensor, shock=shock))
        rows.append(dict(hours=pump.age_h, shock=shock, **{k: float(f[k]) for k in pm.PUMP_FEATURES}))
        pump.age_h += step_h
    df = pd.DataFrame(rows)
    normal = df[(df["hours"] < 300) & ~df["shock"]]
    hs = pm.HealthScore().fit({k: normal[k].values for k in pm.PUMP_FEATURES})
    df["score"] = hs.score({k: df[k].values for k in pm.PUMP_FEATURES})
    return df

life = run_life()
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(life["hours"], life["score"].clip(upper=57), lw=0.8)
ax.scatter(life.loc[life["shock"], "hours"], life.loc[life["shock"], "score"].clip(upper=57), color="red", s=22, zorder=3, label="일시 충격")
ax.set_ylim(-3, 60); ax.set_xlabel("운전 시간 (h)"); ax.set_ylabel("이상 점수"); ax.legend(); plt.tight_layout(); plt.show()
print(f"구간 {len(life)}개, 일시 충격 {int(life['shock'].sum())}회, 고장 시각 {life['hours'].iloc[-1] + 2:.0f}시간")
'''),
("md", '''
## 3. 내 알람 기준

- `WARN` : 경고 기준 (점검 계획을 잡는 시점)
- `STOP` : 정지 기준 (설비를 세우는 시점)
- `NEED_N` : 몇 번 연속으로 넘어야 알람을 낼지 (1 이면 한 번만 넘어도 알람)

**경고가 고장보다 얼마나 앞섰는지**(리드타임)와, 고장이 멀었는데 울린 **오경보**를 셉니다.
'''),
("code", '''
WARN, STOP, NEED_N = 5.0, 15.0, 3            # 내 값

def alarms(df, th, need_n, life_h=1500, far_h=300):
    """기준을 need_n 번 연속 넘은 시각들. 고장까지 far_h 넘게 남았으면 오경보로 셈."""
    over = (df["score"] >= th).astype(int)
    run_len = over.groupby((over == 0).cumsum()).cumsum()
    hit = df.loc[run_len == need_n, "hours"]
    false = hit[life_h - hit > far_h]
    true = hit[life_h - hit <= far_h]
    return dict(오경보=len(false), 첫_알람_h=float(true.min()) if len(true) else np.nan,
                리드타임_h=float(life_h - true.min()) if len(true) else 0.0)

def report(df, warn, stop, need_n):
    w, s = alarms(df, warn, need_n), alarms(df, stop, need_n)
    return pd.DataFrame({"경고": w, "정지": s}).T

rep = report(life, WARN, STOP, NEED_N)
display(rep.round(0))
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(life["hours"], life["score"], lw=0.8)
for th, c, name in ((WARN, "orange", "경고"), (STOP, "red", "정지")):
    ax.axhline(th, color=c, ls="--", lw=1, label=f"{name} 기준 {th}")
ax.set_xlim(900, 1510); ax.set_ylim(-3, 60); ax.set_xlabel("운전 시간 (h)"); ax.set_ylabel("이상 점수"); ax.legend(); plt.tight_layout(); plt.show()
'''),
("md", '''
## 4. 기준을 바꾸면

경고 기준과 연속 횟수를 바꿔 가며 리드타임과 오경보가 어떻게 달라지는지 한꺼번에 봅니다. 펌프마다 결과가 다르므로 펌프 다섯 대(씨앗 다섯 개)의 평균을 냅니다.
'''),
("code", '''
lives = [run_life(seed=s) for s in (21, 22, 23, 24, 25)]
rows = []
for th in (3, 5, 8, 12):
    for n in (1, 3):
        r = [alarms(df, th, n) for df in lives]
        rows.append(dict(경고_기준=th, 연속_횟수=n, 평균_리드타임_h=np.mean([x["리드타임_h"] for x in r]),
                         오경보_합계=sum(x["오경보"] for x in r)))
grid = pd.DataFrame(rows)
display(grid.round(0))
'''),
("md", '''
## 5. 센서 위치

같은 펌프를 **베이스 프레임**에 붙인 센서로 재면 충격이 약하게 전달되고 잡음이 커집니다. 같은 기준으로 리드타임이 얼마나 줄어드는지 봅니다.
'''),
("code", '''
base = run_life(sensor="base")
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(life["hours"], life["score"], lw=0.8, label="베어링 하우징")
ax.plot(base["hours"], base["score"], lw=0.8, label="베이스 프레임")
ax.axhline(WARN, color="orange", ls="--", lw=1)
ax.set_xlim(900, 1510); ax.set_ylim(-3, 60); ax.set_xlabel("운전 시간 (h)"); ax.set_ylabel("이상 점수"); ax.legend(); plt.tight_layout(); plt.show()
print("경고 리드타임   베어링 하우징", f"{alarms(life, WARN, NEED_N)['리드타임_h']:.0f}시간   베이스 프레임 {alarms(base, WARN, NEED_N)['리드타임_h']:.0f}시간")
'''),
("md", '''
## 6. 미션: 우리 조의 알람 기준

- 목표: 펌프 다섯 대에서 **오경보 0회**, 경고 리드타임 **평균 100시간 이상**
- 정비팀이 부품을 준비하는 데 100시간이 걸린다고 가정합니다.
'''),
("code", '''
MY_WARN, MY_STOP, MY_NEED_N = 5.0, 15.0, 1       # 내 값 (처음 값은 오경보가 나옴)

r = [report(df, MY_WARN, MY_STOP, MY_NEED_N) for df in lives]
lead = np.mean([x.loc["경고", "리드타임_h"] for x in r]); false = sum(x.loc["경고", "오경보"] for x in r)
stop_lead = np.mean([x.loc["정지", "리드타임_h"] for x in r])
print(f"경고: 평균 리드타임 {lead:.0f}시간, 오경보 {false:.0f}회   정지: 평균 리드타임 {stop_lead:.0f}시간")
print("판정:", "합격" if (false == 0 and lead >= 100) else "기준을 다시 정하세요")
pd.DataFrame([dict(warn=MY_WARN, stop=MY_STOP, need_n=MY_NEED_N, warn_lead_h=lead, false_alarms=false, stop_lead_h=stop_lead)]
             ).to_csv(OUT / "pump_lab_result.csv", index=False)
print("저장:", (OUT / "pump_lab_result.csv").relative_to(ROOT))
'''),
("md", '''
## 7. 강사 PC 의 유압 유닛 기록 받아 보기

터미널에서 먼저 실행합니다 (주소는 강사 화면의 "수강생 연결" 구역).

```bash
cd ~/ai_manufac
./scripts/factory fetch <강사 PC 주소>
```
'''),
("code", '''
files = sorted(LIVE.glob("pump_*.csv"))
if not files:
    print("받은 파일이 없습니다. 위의 fetch 명령을 먼저 실행하세요. (없어도 실습에는 지장 없음)")
else:
    pl = pd.read_csv(files[-1])
    print(f"{files[-1].name}: {len(pl)}행, 운전 {pl['run_hours'].min():.0f} ~ {pl['run_hours'].max():.0f}시간")
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(pl["run_hours"], pl["score"], lw=0.9)
    for th, c in ((MY_WARN, "orange"), (MY_STOP, "red")):
        ax.axhline(th, color=c, ls="--", lw=1)
    ax.set_xlabel("운전 시간 (h)"); ax.set_ylabel("이상 점수 (가상 공장 기준)"); plt.tight_layout(); plt.show()
'''),
("md", '''
## 정리

- 기준을 낮추면 일찍 알지만 오경보가 늘고, 높이면 오경보는 줄지만 늦게 압니다.
- **연속 횟수** 조건은 일시 충격에 의한 오경보를 줄이는 가장 값싼 방법입니다.
- 센서를 어디에 붙였는지가 리드타임을 좌우합니다. 알고리즘보다 먼저 정할 일입니다.
'''),
]

if __name__ == "__main__":
    build(2, "2-3_process_lab.ipynb", nb23)
    build(3, "3-3_pump_lab.ipynb", nb33)
