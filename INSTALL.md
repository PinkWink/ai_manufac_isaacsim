# 설치 가이드

이 저장소에는 PC 를 세팅하는 데 필요한 코드·실습 데이터·노트북만 들어 있음. 인터넷에서 받을 수 있는 것(파이썬 패키지, Isaac Sim, 3D 에셋)은 아래 명령이 받아서 설치함.

| PC | 하는 일 | 설치하는 것 | GPU |
|---|---|---|---|
| 수강생 PC | 노트북 실습, 브라우저로 가상 공장 조작, 데이터 받기 | 분석 환경 | 필요 없음 |
| 강사 PC | 가상 공장 3D 시연 + 위의 것 | 분석 환경, Isaac Sim 5.1, 3D 에셋 | NVIDIA RTX 8GB 이상 |

확인한 환경: Ubuntu 22.04 · 24.04 (수강생 PC), Ubuntu 24.04 + Isaac Sim 5.1.0 + RTX 5060 Ti 16GB + NVIDIA 드라이버 580 (강사 PC). 윈도우는 다루지 않음(실행 도구가 bash 스크립트임).

교재(슬라이드·그림·영상): https://pinkwink.atlassian.net/wiki/spaces/EDU/pages/3798663177

## 수강생 PC

### 인터넷이 될 때

```bash
cd ~
git clone <이 저장소 주소> ai_manufac
cd ~/ai_manufac
./scripts/factory setup      # 분석 환경(.venv)을 이 폴더 안에 설치. 관리자 암호 필요 없음
./scripts/factory doctor     # "분석 환경" 과 "실습 자료" 가 모두 ✅ 이면 완료
```

`git` 이 없으면 저장소 페이지의 "Download ZIP" 으로 받아 풀어도 됨. `python3 -m venv` 가 안 된다는 안내가 나오면 `sudo apt install python3-venv`.

### 인터넷이 안 될 때 (폐쇄망 교육장)

강사가 만들어 온 설치 묶음(`ai_manufac_offline_<날짜>.tar.gz`)을 홈 폴더에 복사한 뒤 실행함. 패키지가 묶음 안에 있어서 인터넷과 관리자 암호가 필요 없음.

```bash
cd ~
tar -xzf ai_manufac_offline_*.tar.gz
cd ~/ai_manufac
./scripts/factory setup
./scripts/factory doctor
```

### 쓰는 법

```bash
./scripts/factory notebook 2-1            # 실습 노트북: 2-1 · 2-2 · 2-3 · 3-0 · 3-1 · 3-2 · 3-3 · 4-1
./scripts/factory fetch <강사 PC 주소>    # 강사 PC 의 가상 공장이 방금 기록한 데이터 받기
./scripts/factory stop                    # 남아 있는 프로그램 끝내기
```

가상 공장 조작: 강사 화면의 "수강생 연결" 구역에 나오는 주소(`http://<강사 PC 주소>:8080`)를 브라우저에 입력함. 강사가 조작을 허용한 동안에만 조작됨.

## 강사 PC

### 1. 그래픽 드라이버

```bash
nvidia-smi                   # 표가 나오면 됨. 명령이 없으면 아래를 실행하고 다시 시작
sudo ubuntu-drivers install
```

### 2. 저장소와 분석 환경

```bash
cd ~
git clone <이 저장소 주소> ai_manufac
cd ~/ai_manufac
./scripts/factory setup
```

### 3. Isaac Sim 5.1

```bash
./scripts/factory isaacsim   # 8.8GB 를 내려받아 ~/isaacsim 에 설치 (풀면 약 21GB)
```

- 이미 설치되어 있으면 건너뜀. 다른 폴더에 설치했다면 실행할 때마다 `ISAACSIM_PATH=<설치 폴더>` 를 앞에 붙임.
- 압축 파일을 미리 받아 두었다면 `ISAACSIM_ZIP=<파일> ./scripts/factory isaacsim`.
- 직접 받는 주소: https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip

### 4. 3D 에셋 (UR5e 로봇, 공장 건물)

```bash
./scripts/factory assets     # NVIDIA 에셋 서버에서 약 300MB 를 받아 assets/sim/ 에 저장
```

에셋은 NVIDIA 의 라이선스를 따르므로 이 저장소에 넣지 않았음. 한 번 받으면 그 뒤로는 인터넷 없이 실행됨. 3회차(검사 영상의 조명)와 4회차(UR5e)는 에셋이 있어야 함.

### 5. 점검과 실행

```bash
./scripts/factory doctor     # "가상 공장" 항목까지 ✅
./scripts/factory session2   # 2회차 가상 공장. session3 · session4 도 같음
```

처음 실행할 때는 셰이더를 준비하느라 창이 뜨기까지 몇 분 걸림. 수업 전에 세 회차를 한 번씩 미리 실행해 둠.

### 6. 수강생 연결

가상 공장을 띄우면 강사 PC 가 내부망에 웹 서버(포트 8080)를 엶. 주소는 진행 패널 맨 아래 "수강생 연결" 구역에 나옴.

- 방화벽이 켜져 있으면: `sudo ufw allow 8080/tcp`
- 로그인이 없으므로 교육장 내부망에서만 씀. `조별 조작 허용` 을 켠 동안에만 수강생의 조작을 받음.
- 옵션: `--port 9000`(포트 변경), `--web-allow`(처음부터 조작 허용), `--no-web`(쓰지 않음)

### 7. 폐쇄망 교육장으로 가져갈 것

인터넷이 되는 곳에서 준비함.

```bash
./scripts/factory pack       # 수강생용 설치 묶음 → dist/ai_manufac_offline_<날짜>.tar.gz (약 300MB)
```

- 수강생 PC 의 파이썬 버전을 미리 확인함(`python3 --version`). 묶음에는 3.10 과 3.12 용 패키지가 들어감. 다른 버전이면 `PY_VERSIONS="3.11" ./scripts/factory pack`.
- 강사 PC 가 폐쇄망이면 Isaac Sim 압축 파일과 `assets/sim/` 폴더를 미리 받아 옮김.

## 강사만 보는 자료

정답 파일과 교시별 진행안은 이 저장소에 없음. 컨플루언스의 `[강사용]` 페이지에 있고, 정답 파일은 회차별 `[강사용]` 페이지에 첨부되어 있음. 받은 정답 파일은 `data/session<회차>/` 에 넣음.

## 문제 해결

| 증상 | 조치 |
|---|---|
| `이 PC 의 파이썬(3.x)에 맞는 묶음이 없습니다` | 그 버전으로 묶음을 다시 만듦: `PY_VERSIONS="3.x" ./scripts/factory pack` |
| `Isaac Sim 을 찾지 못했습니다` | `./scripts/factory isaacsim`, 또는 설치 폴더를 `ISAACSIM_PATH` 로 지정 |
| `UR5e 에셋이 없습니다` | `./scripts/factory assets` (인터넷 필요), 또는 `assets/sim/` 폴더를 복사 |
| `가상 공장이 이미 실행 중입니다` | `./scripts/factory stop` 후 다시 실행 |
| 수강생 브라우저가 접속되지 않음 | 같은 내부망인지(`ping <강사 PC 주소>`), 방화벽, 패널에 나온 주소 확인 |
| 노트북 브라우저가 열리지 않음 | `./scripts/factory notebook 2-1 --no-browser` 후 터미널에 나온 주소를 직접 입력 |
| `doctor` 에 ❌ | 항목 옆의 안내대로 `setup` 또는 `data` 실행 |
