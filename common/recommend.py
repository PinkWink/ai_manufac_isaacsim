"""분류 모델로 성형 조건의 안전 범위를 찾음 (공정 최적화 시연용). sklearn 모델을 받으므로 분석 환경에서만 씀."""
import numpy as np
import pandas as pd

CONDITIONS = ["mold_temp_c", "melt_temp_c", "inj_pressure_mpa", "hold_pressure_mpa", "hold_time_s", "cooling_time_s"]


def sweep(model, data, base, key, n=41):
    """key 조건만 바꿔 가며(나머지는 base 값) 예측 불량 확률을 구함. 범위는 데이터가 있는 구간(1~99%)."""
    lo, hi = np.percentile(data[key], [1, 99])
    grid = np.linspace(lo, hi, n)
    x = pd.DataFrame([base] * n)[CONDITIONS]
    x[key] = grid
    return grid, model.predict_proba(x)[:, 1]


def recommend(model, data, base, limit=0.05):
    """조건마다 예측 불량 확률이 limit 아래인 범위와 그 가운데 값(추천값)을 돌려줌."""
    out = {}
    for key in CONDITIONS:
        grid, p = sweep(model, data, base, key)
        ok = p < limit
        if not ok.any():
            out[key] = dict(low=float(base[key]), high=float(base[key]), recommend=float(base[key]))
            continue
        best, start = (0, 0), None                       # 가장 긴 연속 안전 구간
        for i, v in enumerate(list(ok) + [False]):
            if v and start is None:
                start = i
            if not v and start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        lo, hi = grid[best[0]], grid[best[1] - 1]
        out[key] = dict(low=round(float(lo), 1), high=round(float(hi), 1), recommend=round(float((lo + hi) / 2), 1))
    return out
