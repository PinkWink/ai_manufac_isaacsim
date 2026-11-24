"""외관 이상탐지: 정상 이미지의 화소별 평균·표준편차를 기준으로 결함을 찾음.

- numpy 만 사용함 (실습 노트북과 가상 공장의 실시간 판정이 같은 코드를 씀).
- 촬영 조건(조명·배경·위치)이 고정되어 있어야 성립하는 방법임.
"""
import numpy as np

SIZE = 96          # 분석용 크기 (원본을 줄여서 사용)


def prepare(img, size=SIZE):
    """uint8 [H, W, 3] → float [size, size, 3] (0~1). 구역 평균으로 줄임."""
    a = np.asarray(img)[..., :3].astype(np.float32) / 255.0
    f = a.shape[0] // size
    a = a[:size * f, :size * f]
    return a.reshape(size, f, size, f, 3).mean(axis=(1, 3))


def box_blur(a, r=1):
    """[H, W] 를 (2r+1) 크기 상자로 평균."""
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


class PixelModel:
    def fit(self, images):
        """images: 정상 이미지 [n, size, size, 3]"""
        x = np.asarray(images, dtype=np.float32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 0.012          # 표준편차가 너무 작은 화소가 과민해지지 않게 하한을 둠
        return self

    def anomaly_map(self, image):
        """정상에서 벗어난 정도 [size, size]. 색 채널 중 가장 큰 값을 쓰고 주변과 평균함."""
        z = np.abs(np.asarray(image, dtype=np.float32) - self.mean) / self.std
        return box_blur(z.max(axis=-1), 1)

    def score(self, image):
        """이미지 한 장의 이상 점수: 지도에서 가장 높은 0.3% 화소의 평균."""
        m = np.sort(self.anomaly_map(image).ravel())
        return float(m[-max(len(m) * 3 // 1000, 4):].mean())

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls()
        m.mean, m.std = d["mean"], d["std"]
        return m
