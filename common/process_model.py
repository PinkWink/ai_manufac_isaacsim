"""가상 가전공장 공정 모델 (사출·도장).

- 시뮬 화면(Isaac Sim)과 실습 데이터셋이 같은 인과 모델을 쓰도록 한 곳에 모았음.
- numpy 만 사용함 (Isaac Sim 의 python.sh 와 분석용 가상환경 양쪽에서 import 함).
"""
from dataclasses import dataclass, asdict

import numpy as np

# ----------------------------------------------------------------------------
# 사출 (냉장고 도어 트림, 샷 단위)
# ----------------------------------------------------------------------------
INJ_NOMINAL = dict(
    mold_temp_c=60.0,         # 금형온도
    melt_temp_c=230.0,        # 용융수지온도
    inj_pressure_mpa=90.0,    # 사출압
    hold_pressure_mpa=55.0,   # 보압
    hold_time_s=8.0,          # 보압 시간
    cooling_time_s=28.0,      # 냉각 시간
)
# 샷마다 흔들리는 정도(표준편차)
INJ_NOISE = dict(
    mold_temp_c=1.2, melt_temp_c=2.0, inj_pressure_mpa=2.0,
    hold_pressure_mpa=1.6, hold_time_s=0.15, cooling_time_s=0.5,
)
# 슬라이더 범위 (시뮬 패널에서 사용)
INJ_RANGE = dict(
    mold_temp_c=(40.0, 80.0), melt_temp_c=(205.0, 255.0),
    inj_pressure_mpa=(65.0, 115.0), hold_pressure_mpa=(30.0, 75.0),
    hold_time_s=(4.0, 12.0), cooling_time_s=(16.0, 40.0),
)
INJ_DEFECTS = ("flow_mark", "sink", "short_shot")   # 흐름자국, 수축, 미성형
DEFECT_KO = dict(none="양품", flow_mark="흐름자국", sink="수축", short_shot="미성형",
                 sag="흐름", pinhole="핀홀")
FIXED_CYCLE_S = 37.0      # 형개폐·사출·취출 등 조건과 무관한 시간

# 설비 배치 (시뮬 화면과 데이터셋이 함께 씀)
MACHINE_X = {"IMM-01": -10.0, "IMM-02": -6.0, "IMM-03": -2.0}   # 사출기 위치 (m)
INSPECT_X = 1.5                                                  # 외관 검사기 위치 (m)
BELT_SPEED = 0.08                                                # 컨베이어 속도 (m/s)
TAKEOUT_S = 10.0                                                 # 취출 시간 (s)
# 샷 완료 → 외관 검사까지 걸리는 시간 (s): 취출 + 컨베이어 이동. 설비마다 다름
INSPECT_DELAY_S = {m: round(TAKEOUT_S + (INSPECT_X - x) / BELT_SPEED) for m, x in MACHINE_X.items()}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(x, 0.0)


def injection_cycle_time(hold_time_s, cooling_time_s):
    return FIXED_CYCLE_S + hold_time_s + cooling_time_s


def injection_defect_probs(mold_temp_c, melt_temp_c, inj_pressure_mpa,
                           hold_pressure_mpa, hold_time_s, cooling_time_s):
    """성형 조건 → 결함별 발생 확률. 배열 입력 가능.

    - 흐름자국: 금형·수지 온도가 낮아 수지가 금형 면에서 일찍 굳을 때
    - 수축    : 보압이 낮거나, 냉각이 짧거나(사이클타임 단축), 금형온도가 높을 때
    - 미성형  : 사출압과 수지 온도가 함께 낮을 때
    """
    r = _relu
    flow = _sigmoid(-6.0 + 2.00 * r(56.5 - mold_temp_c)
                    + 0.25 * r(226.0 - melt_temp_c)
                    + 0.10 * r(86.0 - inj_pressure_mpa))
    sink = _sigmoid(-6.0 + 2.00 * r(52.0 - hold_pressure_mpa)
                    + 2.80 * r(26.0 - cooling_time_s)
                    + 0.50 * r(mold_temp_c - 63.5)
                    + 1.20 * r(7.2 - hold_time_s))
    short = _sigmoid(-6.5 + 1.00 * r(84.0 - inj_pressure_mpa)
                     + 0.40 * r(222.0 - melt_temp_c))
    return dict(flow_mark=flow, sink=sink, short_shot=short)


def sample_defect(probs, rng):
    """결함 하나를 뽑음 (여러 개가 동시에 걸리면 확률이 큰 것). 스칼라용."""
    hit = [(p, k) for k, p in probs.items() if rng.random() < p]
    return max(hit)[1] if hit else "none"


def sample_defect_array(probs, rng):
    keys = list(probs)
    p = np.stack([np.asarray(probs[k], dtype=float) for k in keys], axis=1)
    hit = rng.random(p.shape) < p
    best = np.where(hit, p, -1.0).argmax(axis=1)
    out = np.array(keys, dtype=object)[best]
    out[~hit.any(axis=1)] = "none"
    return out


@dataclass
class Shot:
    shot_no: int
    mold_temp_c: float
    melt_temp_c: float
    inj_pressure_mpa: float
    hold_pressure_mpa: float
    hold_time_s: float
    cooling_time_s: float
    cycle_time_s: float
    defect: str

    def as_dict(self):
        return asdict(self)


class InjectionMachine:
    """시뮬용: 설정값(setpoint)을 받아 샷을 하나씩 만들어 냄."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.setpoint = dict(INJ_NOMINAL)
        self.shot_no = 0

    def next_shot(self):
        self.shot_no += 1
        v = {k: float(self.setpoint[k] + self.rng.normal(0.0, INJ_NOISE[k])) for k in INJ_NOMINAL}
        probs = injection_defect_probs(**v)
        return Shot(self.shot_no, **{k: round(x, 2) for k, x in v.items()},
                    cycle_time_s=round(float(injection_cycle_time(v["hold_time_s"], v["cooling_time_s"])
                                             + self.rng.normal(0, 0.3)), 2),
                    defect=sample_defect({k: float(p) for k, p in probs.items()}, self.rng))

    def expected_defect_rate(self):
        """현재 설정값에서의 기대 불량률 (샷 노이즈 포함, 몬테카를로)."""
        n = 4000
        v = {k: self.setpoint[k] + self.rng.normal(0.0, INJ_NOISE[k], n) for k in INJ_NOMINAL}
        p = injection_defect_probs(**v)
        ok = (1 - p["flow_mark"]) * (1 - p["sink"]) * (1 - p["short_shot"])
        return float(1.0 - ok.mean())


# ----------------------------------------------------------------------------
# 도장 (외판 분체·액체 도장 부스, 패널 단위)
# ----------------------------------------------------------------------------
PAINT_NOMINAL = dict(
    booth_temp_c=24.0,        # 부스 온도
    booth_humidity_pct=55.0,  # 부스 습도
    viscosity_s=22.0,         # 도료 점도 (포드컵 초)
    film_thickness_um=38.0,   # 도막 두께
)
PAINT_NOISE = dict(booth_temp_c=0.6, booth_humidity_pct=2.5, viscosity_s=0.8, film_thickness_um=2.5)
PAINT_RANGE = dict(booth_temp_c=(16.0, 34.0), booth_humidity_pct=(30.0, 90.0),
                   viscosity_s=(16.0, 28.0), film_thickness_um=(22.0, 60.0))
PAINT_DEFECTS = ("sag", "pinhole")   # 흐름, 핀홀


def paint_defect_probs(booth_temp_c, booth_humidity_pct, viscosity_s, film_thickness_um):
    """부스 환경·도막 → 결함 확률.

    - 흐름(sag) : 도막이 두껍고, 점도가 낮고, 부스 온도가 낮아 건조가 늦을 때
    - 핀홀      : 습도가 높고 부스 온도가 높아 용제가 급히 날아갈 때, 도막이 두꺼울 때
    """
    r = _relu
    sag = _sigmoid(-6.0 + 0.70 * r(film_thickness_um - 42.0)
                   + 1.80 * r(20.5 - viscosity_s)
                   + 0.60 * r(22.0 - booth_temp_c))
    pin = _sigmoid(-6.0 + 0.60 * r(booth_humidity_pct - 62.0)
                   + 1.00 * r(booth_temp_c - 26.0)
                   + 0.20 * r(film_thickness_um - 44.0))
    return dict(sag=sag, pinhole=pin)


class PaintBooth:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.setpoint = dict(PAINT_NOMINAL)
        self.panel_no = 0

    def next_panel(self):
        self.panel_no += 1
        v = {k: float(self.setpoint[k] + self.rng.normal(0.0, PAINT_NOISE[k])) for k in PAINT_NOMINAL}
        probs = paint_defect_probs(**v)
        d = sample_defect({k: float(p) for k, p in probs.items()}, self.rng)
        return dict(panel_no=self.panel_no, **{k: round(x, 2) for k, x in v.items()}, defect=d)

    def expected_defect_rate(self):
        n = 4000
        v = {k: self.setpoint[k] + self.rng.normal(0.0, PAINT_NOISE[k], n) for k in PAINT_NOMINAL}
        p = paint_defect_probs(**v)
        return float(1.0 - ((1 - p["sag"]) * (1 - p["pinhole"])).mean())


# ----------------------------------------------------------------------------
# 유압 유닛 (사출기 유압 펌프·모터) — 3회차 예지보전
# ----------------------------------------------------------------------------
PUMP_FS = 2000          # 진동 수집 주파수 (Hz)
PUMP_N = 1024           # 한 번에 모으는 표본 수 (약 0.5초)
SHAFT_HZ = 24.7         # 축 회전 주파수 (1,480 rpm)
BPFO_HZ = 88.3          # 베어링 외륜 결함 주파수
RESONANCE_HZ = 620.0    # 충격이 울리는 구조 공진
PUMP_FEATURES = ("rms", "kurtosis", "crest", "peak")


class PumpUnit:
    """베어링이 닳아 가는 유압 펌프. life_h=None 이면 닳지 않음."""

    def __init__(self, seed=0, life_h=1500.0, age_h=0.0):
        self.rng = np.random.default_rng(seed)
        self.life_h, self.age_h = life_h, age_h

    @property
    def hours_to_failure(self):
        return None if self.life_h is None else self.life_h - self.age_h

    @property
    def failed(self):
        return self.life_h is not None and self.age_h >= self.life_h

    def severity(self):
        """0(정상) ~ 1(고장 직전). 수명의 마지막 20% 구간에서 빠르게 커짐."""
        if self.life_h is None:
            return 0.0
        x = self.age_h / self.life_h
        return float(np.clip((x - 0.8) / 0.2, 0.0, 1.3) ** 2)

    def window(self, sensor="bearing", shock=False, load=None):
        """진동 가속도 한 구간 (g). sensor: bearing(베어링 하우징) / base(베이스 프레임)."""
        rng, sev = self.rng, self.severity()
        t = np.arange(PUMP_N) / PUMP_FS
        load = rng.uniform(0.75, 1.25) if load is None else load          # 생산 부하에 따른 변동
        x = load * (0.30 * np.sin(2 * np.pi * SHAFT_HZ * t + rng.uniform(0, 6.28))
                    + 0.12 * np.sin(2 * np.pi * 2 * SHAFT_HZ * t + rng.uniform(0, 6.28)))
        gain, noise = (1.0, 0.08) if sensor == "bearing" else (0.25, 0.13)   # 베이스에서는 충격이 약하고 잡음이 큼
        amp = gain * (0.04 + 2.4 * sev)
        k = np.arange(-0.05, PUMP_N / PUMP_FS, 1.0 / BPFO_HZ) + rng.uniform(0, 1.0 / BPFO_HZ)
        for t0 in k:                                                      # 결함을 지날 때마다 생기는 충격
            dt_ = t - t0
            m = dt_ >= 0
            x[m] += amp * rng.uniform(0.7, 1.3) * np.exp(-dt_[m] / 0.004) * np.sin(2 * np.pi * RESONANCE_HZ * dt_[m])
        if shock:                                                         # 지게차 충돌 같은 일시 외란
            dt_ = t - rng.uniform(0.1, 0.4)
            m = dt_ >= 0
            x[m] += gain * 3.0 * np.exp(-dt_[m] / 0.012) * np.sin(2 * np.pi * 180.0 * dt_[m])
        return (x + rng.normal(0, noise, PUMP_N)).astype(np.float32)

    def current_rms(self):
        """모터 전류 실효값 (A). 베어링이 닳으면 마찰이 늘어 조금 오름."""
        return float(12.0 + 1.8 * self.severity() + self.rng.normal(0, 0.15))


def vib_features(x):
    """진동 구간 → 특징 (실효값, 첨도, 파고율, 최댓값). x: [..., N]"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    rms = np.sqrt((x ** 2).mean(axis=-1))
    peak = np.abs(x).max(axis=-1)
    kurt = (x ** 4).mean(axis=-1) / np.maximum((x ** 2).mean(axis=-1) ** 2, 1e-12)
    return dict(rms=rms, kurtosis=kurt, crest=peak / np.maximum(rms, 1e-12), peak=peak)


class HealthScore:
    """정상 구간의 특징 분포를 기준으로 한 이상 점수 (특징별 z 점수 중 가장 큰 값)."""

    def __init__(self, keys=("rms", "kurtosis", "crest")):
        self.keys = keys

    def fit(self, feats):
        self.mean = {k: float(np.mean(feats[k])) for k in self.keys}
        self.std = {k: float(np.std(feats[k]) + 1e-9) for k in self.keys}
        return self

    def score(self, feats):
        z = [(np.asarray(feats[k]) - self.mean[k]) / self.std[k] for k in self.keys]
        return np.max(np.stack(z, axis=0), axis=0)
