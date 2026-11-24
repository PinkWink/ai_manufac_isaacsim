"""4회차 실습 노트북(4-1)을 만듦.  실행: .venv/bin/python scripts/make_notebooks_s4.py"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "session4" / "notebooks"


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)


nb41 = [
("md", '''
# 실습 4-1. 장비 로그로 이상 찾기 · 판정으로 분기하기

가상 공장의 **협동로봇 UR5e**(진공 흡착 툴)가 부품을 집어 옮기는 동안 기록한 **관절 각도 · 관절 토크 로그**를 다룸.
(관절 토크는 물리 엔진이 계산한 값이고, 실물 장비라면 모터 전류에 해당함)

| 단계 | 폐루프 위치 | 이 노트북에서 |
|---|---|---|
| 1단계 인출 | 인출 | 로그 불러오기, 한 사이클 들여다보기 (1절) |
| 2단계 판단 | 판단 | 사이클별 특징과 이상 점수, 이상 구간 표시 (2 ~ 3절) |
| 3단계 제어 | 제어 지령 | 체결 이력 · 비전 판정으로 양품 · 재검 분기 (4절) |

- **기본 트랙**: 셀을 차례로 실행하고 이상 구간과 분기 결과를 워크북에 옮겨 적음
- **심화 트랙**: `# 심화` 표시가 있는 값을 바꿔 기준을 조정함
'''),
("code", '''
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

DATA = ROOT / "data" / "session4"
OUT = DATA / "work"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
'''),
("md", '''
## 1. 인출: 장비 로그

협동로봇 UR5e(진공 흡착 툴)가 30 사이클 동안 20 Hz 로 기록한 로그임. 관절 6개의 각도 · 토크와 진공 압력(`vacuum_kpa`)이 들어 있음. 강사 PC 에서 방금 기록한 로그를 받았으면(`./scripts/factory fetch <강사 PC 주소>` → `data/session4/live/`) 그것을 쓸 수 있음.
'''),
("code", '''
live = sorted((DATA / "live").glob("robot_log_*.csv"))
USE_LIVE = False                          # 심화: True 로 바꾸면 방금 기록한 로그를 사용
src = live[-1] if (USE_LIVE and live) else DATA / "robot_log_provided.csv"
log = pd.read_csv(src)
log = log[log["cycle"] > 0]
print(src.name, log.shape, " 사이클:", log["cycle"].nunique())
log.head()
'''),
("code", '''
# 한 사이클의 관절 각도와 토크
one = log[log["cycle"] == log["cycle"].min() + 1]
t = one["time_s"] - one["time_s"].iloc[0]
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for j, name in enumerate(["J1 베이스", "J2 어깨", "J3 팔꿈치", "J4 손목1", "J5 손목2", "J6 손목3"], start=1):
    axes[0].plot(t, one[f"q{j}_deg"], label=name)
    axes[1].plot(t, one[f"tau{j}_nm"], label=name)
grip = one["gripping"].values.astype(bool)
for ax in axes:
    ax.fill_between(t, *ax.get_ylim(), where=grip, color="#f5cba7", alpha=0.35)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("관절 각도 (도)"); axes[0].legend(ncol=6, fontsize=9); axes[0].set_title("한 사이클  (색칠한 구간: 진공으로 부품을 잡고 있는 동안)")
axes[1].set_ylabel("관절 토크 (N·m)"); axes[1].set_xlabel("시간 (초)")
plt.tight_layout(); fig.savefig(OUT / "robot_cycle.png", dpi=120); plt.show()
'''),
("md", '''
> 팔을 뻗을수록, 그리고 부품을 잡고 있을 때 어깨(J2) 토크가 커짐. 중력을 버티는 힘이기 때문임.

## 2. 판단: 사이클별 특징

부품을 잡고 옮기는 구간의 토크를 사이클마다 숫자 세 개로 줄임.
'''),
("code", '''
carry = log[log["gripping"] == 1]
feat = carry.groupby("cycle").agg(j2_mean=("tau2_nm", lambda s: s.abs().mean()),
                                  j2_max=("tau2_nm", lambda s: s.abs().max()),
                                  j3_mean=("tau3_nm", lambda s: s.abs().mean()))
feat.round(2).head(8)
'''),
("md", '''
## 3. 판단: 정상 기준과 이상 점수

처음 10 사이클을 정상으로 보고 기준을 잡음. 이상 점수 = 기준에서 벗어난 정도(표준편차의 몇 배인지).
'''),
("code", '''
N_BASE = 10                    # 심화: 기준을 잡는 사이클 수
THRESHOLD = 4.0                # 심화: 이상으로 보는 점수
base = feat.iloc[:N_BASE]
z = (feat - base.mean()) / (base.std() + 0.02)
feat["score"] = z.abs().max(axis=1)
feat["이상"] = feat["score"] >= THRESHOLD

fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
axes[0].plot(feat.index, feat["j2_mean"], marker="o", color="#1f618d"); axes[0].set_ylabel("J2 평균 토크 (N·m)"); axes[0].grid(alpha=0.3)
axes[1].bar(feat.index, feat["score"], color=["#c0392b" if b else "#7fb3d5" for b in feat["이상"]])
axes[1].axhline(THRESHOLD, color="gray", ls="--"); axes[1].set_ylabel("이상 점수"); axes[1].set_xlabel("사이클"); axes[1].grid(alpha=0.3)
axes[0].set_title("사이클별 어깨 토크와 이상 점수  (빨강: 이상 구간)")
plt.tight_layout(); fig.savefig(OUT / "robot_anomaly.png", dpi=120); plt.show()
feat.to_csv(OUT / "robot_anomaly.csv", encoding="utf-8-sig")
print("이상으로 표시된 사이클:", list(feat.index[feat["이상"]]))
'''),
("md", '''
> 토크가 높아진 구간은 평소보다 **무거운 부품**을 옮긴 사이클임. 다른 부품이 섞였거나, 이물이 붙었거나, 두 개를 한꺼번에 집은 경우에 해당함.
> 토크가 조금씩 계속 오른다면 부품이 아니라 **설비**(감속기 · 베어링 마찰)를 의심함 → 3회차의 예지보전과 같은 방법.

## 4. 제어: 판정으로 분기하기

조립 라인의 **체결 토크 이력**(나사 4개)과 최종검사 **비전 판정**(커넥터 색 화소 비율)을 묶어 양품 라인과 재검 라인으로 나눔.
`recheck_result` 는 사람이 재검해서 확정한 결과임.
'''),
("code", '''
asm = pd.read_csv(DATA / "assembly_inspection.csv", parse_dates=["timestamp"])
TQ = [f"torque{j}_nm" for j in range(1, 5)]
print(asm.shape)
asm.head()
'''),
("code", '''
TORQUE_MIN, TORQUE_MAX = 1.8, 2.2     # 체결 토크 규격 (N·m)
VISION_MIN = 1.0                      # 심화: 커넥터가 있다고 보는 화소 비율 (%)

asm["체결 판정"] = asm[TQ].ge(TORQUE_MIN).all(axis=1) & asm[TQ].le(TORQUE_MAX).all(axis=1)
asm["비전 판정"] = asm["connector_ratio_pct"] >= VISION_MIN
asm["분기"] = np.where(asm["체결 판정"] & asm["비전 판정"], "양품 라인", "재검 라인")
table = pd.crosstab(asm["분기"], asm["recheck_result"], margins=True)
table.to_csv(OUT / "branch_result.csv", encoding="utf-8-sig")
print(f"양품 라인으로 나간 불량: {int(((asm['분기'] == '양품 라인') & (asm['recheck_result'] == '불량')).sum())}대")
print(f"재검 라인으로 간 양품:   {int(((asm['분기'] == '재검 라인') & (asm['recheck_result'] == '양품')).sum())}대")
table
'''),
("code", '''
# 양품 라인으로 빠져나간 불량은 무엇인가
leak = asm[(asm["분기"] == "양품 라인") & (asm["recheck_result"] == "불량")]
fig, ax = plt.subplots(figsize=(11, 3.2))
bins = np.linspace(0, 6, 61)
ax.hist(asm.loc[asm.recheck_result == "양품", "connector_ratio_pct"], bins=bins, color="#2471a3", alpha=0.7, label="양품")
ax.hist(asm.loc[asm.defect_kind == "커넥터 미삽입", "connector_ratio_pct"], bins=bins, color="#c0392b", alpha=0.8, label="커넥터 미삽입")
ax.hist(asm.loc[asm.defect_kind == "커넥터 반삽입", "connector_ratio_pct"], bins=bins, color="#d68910", alpha=0.9, label="커넥터 반삽입")
ax.axvline(VISION_MIN, color="gray", ls="--"); ax.set_xlabel("커넥터 색 화소 비율 (%)"); ax.set_ylabel("대수"); ax.legend()
ax.set_title("비전 판정 기준과 커넥터 상태"); plt.tight_layout(); fig.savefig(OUT / "vision_ratio.png", dpi=120); plt.show()
leak["defect_kind"].value_counts().to_frame("양품 라인으로 나간 대수")
'''),
("md", '''
> 커넥터가 **반만 꽂힌** 제품은 화소 비율이 기준을 넘어 양품으로 나감. 기준을 올리면 잡을 수 있지만 양품이 재검으로 가는 수도 늘어남.
> 판정이 애매한 구간(예: 1 ~ 4%)은 자동으로 보내지 않고 **사람이 확인하는 재판정 경로**로 돌리는 것이 제어 수준 2(권고 · 사람 조치)임.

### 워크북에 옮겨 적기

1. 이상으로 표시된 사이클 번호와 이상 판정 기준
2. 분기 결과표 (양품 라인 · 재검 라인 × 재검 결과)
3. 양품 라인으로 나간 불량의 종류와, 이를 막기 위한 재판정 구간

저장된 파일: `data/session4/work/robot_anomaly.csv`, `branch_result.csv`
'''),
]

build("4-1_robot_log_and_branch.ipynb", nb41)
