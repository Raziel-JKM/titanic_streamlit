"""app.py의 차트를 PNG로 뽑아 레이아웃(라벨 충돌·잘림)을 눈으로 확인하는 스크립트."""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

import app

mode = "dark" if "--dark" in sys.argv else "light"
c = app.TOKENS[mode]
df = app.load_data()

charts = {
    "sex_class": app.chart_sex_class,
    "age": app.chart_age,
    "family": app.chart_family,
    "title": app.chart_title,
    "cabin": app.chart_cabin,
    "embarked": app.chart_embarked,
    "fare": app.chart_fare,
}

for name, fn in charts.items():
    fig, _ = fn(df, c)
    if fig is None:
        continue
    out = app.BASE / f"_preview_{name}_{mode}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out.name)
