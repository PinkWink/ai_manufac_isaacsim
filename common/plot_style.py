"""노트북 그래프에 한글이 보이도록 글꼴을 등록함."""
from pathlib import Path

import matplotlib
from matplotlib import font_manager

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NanumSquareRoundB.ttf"


def use_korean_font():
    font_manager.fontManager.addfont(str(FONT))
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 110
