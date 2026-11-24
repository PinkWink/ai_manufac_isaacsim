# AI 자율제조 교육 - Isaac Sim 가상 공장

AI 자율제조 전문인력 양성 교육(가전 제조공정 과정)의 2·3·4회차를 **Isaac Sim 가상 공장**으로 진행하기 위한 코드와 실습 자료임.

- 강사 PC 에서 가상 공장(사출·도장·검사 라인, UR5e 로봇 셀)을 3D 로 띄워 보여 줌.
- 수강생은 GPU 없는 PC 에서 **노트북 실습**을 하고, **브라우저로 강사 PC 의 가상 공장을 조작**하고, 방금 만들어진 데이터를 받아 분석함.
- 가상 공장과 실습 데이터, 실습 노트북은 같은 공정 모델(`common/process_model.py`)을 씀.

- 설치: [INSTALL.md](INSTALL.md)
- 교재(슬라이드·그림·영상): https://pinkwink.atlassian.net/wiki/spaces/EDU/pages/3798663177

## 회차

| 회차 | 주제 | 가상 공장 (강사 PC) | 수강생 실습 |
|---|---|---|---|
| 2회 | 제조데이터 수집·정제·분석 | 사출기 3대·컨베이어·외관 검사기·도장 부스, 데이터 오류 5종 재현 | 2-1 정제, 2-2 분석·대시보드, 2-3 공정 실험 |
| 3회 | AI 품질검사·이상탐지·예지보전·공정 최적화 | 도장 외관 검사 셀(조명·배경·실시간 판정), 유압 유닛(2단 알람), 조건 추천 | 3-0 조건 추천(시연), 3-1 외관 이상탐지, 3-2 예지보전, 3-3 예지보전 실험 |
| 4회 | 설비 연계·장비 제어·운영 | UR5e 협동로봇(진공 흡착)·비전 판정 분기·제어 3수준·인터록 4종 | 4-1 로봇 로그와 분기, 브라우저로 로봇 셀 조작 |

## 수강생 PC (노트북 실습만, GPU 불필요)

Ubuntu 22.04 · 24.04 에서 확인함.

```bash
./scripts/factory setup          # 분석 환경(.venv) 설치, 처음 한 번
./scripts/factory doctor         # 점검: 분석 환경 · 실습 자료가 모두 ✅ 이면 완료
./scripts/factory notebook 2-1   # 실습 노트북 열기: 2-1 · 2-2 · 2-3 · 3-0 · 3-1 · 3-2 · 3-3 · 4-1
./scripts/factory fetch <강사 PC 주소>   # 강사 PC 의 가상 공장이 방금 기록한 데이터 받기
```

- 인터넷이 안 되는 교육장에는 설치 묶음을 만들어 가져감: 인터넷이 되는 PC 에서 `./scripts/factory pack` → `dist/ai_manufac_offline_<날짜>.tar.gz` 를 복사해 풀고 `./scripts/factory setup`.
- 가상 공장 조작: 강사 화면의 "수강생 연결" 구역에 나오는 주소를 브라우저에 입력함.

## 강사 PC (가상 공장)

Isaac Sim 5.1, Ubuntu 24.04, RTX 5060 Ti 16GB 에서 확인함. Isaac Sim 은 `~/isaacsim` 에 설치함(다른 곳이면 `ISAACSIM_PATH` 지정).

```bash
./scripts/factory setup          # 분석 환경
./scripts/factory isaacsim       # Isaac Sim 5.1 내려받아 설치 (8.8GB), 이미 있으면 건너뜀
./scripts/factory assets         # 3D 에셋(UR5e 로봇 · 공장 건물) 받기, 처음 한 번 · 인터넷 필요 · 약 300MB
./scripts/factory doctor         # 점검
./scripts/factory session2       # 2회차 가상 공장 · session3 · session4 도 같음
./scripts/factory stop           # 남아 있는 프로그램 끝내기
./scripts/factory                # 전체 사용법
```

- 에셋은 NVIDIA 의 에셋 서버에서 받으며 이 저장소에 들어 있지 않음. 받은 뒤에는 인터넷 없이 실행됨.
- 2회차는 에셋이 없어도 건물 없이 설비만 뜸. 3회차(검사 영상의 조명이 건물 안 기준)와 4회차(UR5e)는 에셋이 있어야 함.
- 가상 공장을 띄우면 같은 내부망의 수강생 PC 에서 `http://<강사 PC 주소>:8080` 으로 접속할 수 있음. 로그인이 없으므로 내부망에서만 씀. `--no-web` 으로 끌 수 있음.

## 폴더

| 폴더 | 내용 |
|---|---|
| `common/` | 공정 모델(`process_model.py`), 가상 공장(`factory.py`, `factory3.py`), 로봇 셀(`robot_cell.py`), 수강생 연결(`web_link.py`), 이상탐지(`anomaly.py`), 도형 도구(`scene_kit.py`) |
| `session2/` `session3/` `session4/` | 회차별 실행 파일, 데이터 생성기, 실습 노트북 |
| `data/session*/` | 회차별 실습 데이터 |
| `scripts/` | 실행 도구(`factory`), 환경 점검, 노트북 생성, Isaac Sim · 에셋 받기 |
| `assets/fonts/` | 한글 표시용 글꼴 (나눔스퀘어라운드, SIL Open Font License) |

## 이 저장소에 없는 것

이 저장소에는 수강생과 강사가 자기 PC 를 세팅하는 데 필요한 것만 있음.

- 교재(슬라이드·그림·영상)와 강사용 진행안, 정답 파일: 컨플루언스에 있음. 정답 파일은 `[강사용]` 페이지에 첨부되어 있음.
- Isaac Sim, 3D 에셋, 파이썬 패키지: 인터넷에서 받음(`./scripts/factory isaacsim` · `assets` · `setup`).

## 모델과 데이터에 대해

- 사출·도장의 결함 발생, 펌프 베어링 열화는 교육용으로 만든 단순한 모델임. 실제 설비의 수치가 아님.
- 3회차 검사 이미지는 가상 공장의 검사 카메라로 찍은 것이고, 결함은 보이도록 실제보다 크게 표현했음.
- 4회차 로봇의 관절 토크는 Isaac Sim 의 물리 엔진(PhysX)이 계산한 값임.
