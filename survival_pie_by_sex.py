"""titanic.csv 성별 생존 여부 파이차트 (소형 배수 2개).

사용법:
    python survival_pie_by_sex.py           # 라이트 모드
    python survival_pie_by_sex.py --dark    # 다크 모드
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CSV = Path(__file__).parent / "titanic.csv"

# 색상 토큰 (검증된 기본 팔레트: 카테고리 슬롯 1=blue, 2=orange)
TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "series_1": "#3987e5",
        "series_2": "#d95926",
    },
}

# 색은 항상 같은 대상을 따라간다 — 패널이 바뀌어도 생존=슬롯1, 사망=슬롯2
SERIES = [
    {"key": 1, "label": "생존", "slot": "series_1"},
    {"key": 0, "label": "사망", "slot": "series_2"},
]

SEX_LABEL = {"male": "남성", "female": "여성"}


def load_panels():
    df = pd.read_csv(CSV)
    table = pd.crosstab(df["Sex"], df["Survived"])
    panels = []
    for sex in ("male", "female"):  # 패널 순서 고정
        values = [int(table.loc[sex, s["key"]]) for s in SERIES]
        total = sum(values)
        panels.append({
            "title": SEX_LABEL[sex],
            "values": values,
            "total": total,
            "rate": values[0] / total * 100,  # 생존율
        })
    return panels, int(table.to_numpy().sum())


def draw_panel(ax, panel, c):
    wedges, _ = ax.pie(
        panel["values"],
        colors=[c[s["slot"]] for s in SERIES],
        startangle=90,
        counterclock=False,
        radius=1.0,
        # 조각 사이는 테두리가 아니라 2px 서피스 간격으로 분리한다
        wedgeprops=dict(edgecolor=c["surface"], linewidth=2, antialiased=True),
    )

    # 조각마다 값을 직접 라벨링 (정체성은 범례가, 값은 라벨이 전달)
    for wedge, value in zip(wedges, panel["values"]):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ha = "left" if x > 0 else "right"
        pct = value / panel["total"] * 100
        ax.text(x * 1.14, y * 1.14 + 0.07, f"{pct:.1f}%",
                ha=ha, va="center", fontsize=12, fontweight="bold",
                color=c["text_primary"])
        ax.text(x * 1.14, y * 1.14 - 0.08, f"{value:,}명",
                ha=ha, va="center", fontsize=10, color=c["text_secondary"])

    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.35, 1.6)

    ax.text(0, 1.45, panel["title"], ha="center", va="center",
            fontsize=13, fontweight="bold", color=c["text_primary"])
    ax.text(0, 1.26, f"{panel['total']:,}명 · 생존율 {panel['rate']:.1f}%",
            ha="center", va="center", fontsize=10, color=c["text_secondary"])


def build_chart(panels, total, mode="light"):
    c = TOKENS[mode]

    plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2), dpi=160)
    fig.patch.set_facecolor(c["surface"])

    for ax, panel in zip(axes, panels):
        ax.set_facecolor(c["surface"])
        draw_panel(ax, panel, c)

    fig.text(0.5, 0.95, "타이타닉 승객 성별 생존 여부",
             ha="center", fontsize=15, fontweight="bold", color=c["text_primary"])
    gap = panels[1]["rate"] - panels[0]["rate"]
    fig.text(0.5, 0.898,
             f"전체 {total:,}명 · 여성 생존율이 남성보다 {gap:.1f}%p 높다",
             ha="center", fontsize=11, color=c["text_secondary"])

    # 시리즈가 2개이므로 범례는 항상 둔다 (두 패널이 공유)
    fig.legend(
        handles=[Patch(facecolor=c[s["slot"]], edgecolor="none", label=s["label"])
                 for s in SERIES],
        loc="lower center", bbox_to_anchor=(0.5, 0.075),
        ncol=2, frameon=False, handlelength=0.9, handleheight=0.9,
        columnspacing=1.8, fontsize=11, labelcolor=c["text_secondary"],
    )

    fig.text(0.5, 0.03, "출처: titanic.csv (Kaggle 학습용 891명 표본)",
             ha="center", fontsize=9, color=c["muted"])

    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.0)
    return fig


def print_table(panels, total):
    """표 형태의 동일 정보 — 값이 색에만 의존하지 않도록."""
    print(f"{'구분':<6}{'생존':>9}{'사망':>9}{'합계':>9}{'생존율':>9}")
    print("-" * 42)
    for p in panels:
        print(f"{p['title']:<6}{p['values'][0]:>8,}명{p['values'][1]:>8,}명"
              f"{p['total']:>8,}명{p['rate']:>8.1f}%")
    surv = sum(p["values"][0] for p in panels)
    dead = sum(p["values"][1] for p in panels)
    print("-" * 42)
    print(f"{'합계':<6}{surv:>8,}명{dead:>8,}명{total:>8,}명"
          f"{surv / total * 100:>8.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    args = parser.parse_args()
    mode = "dark" if args.dark else "light"

    panels, total = load_panels()
    print_table(panels, total)

    fig = build_chart(panels, total, mode)
    out = CSV.parent / f"survival_pie_by_sex_{mode}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\n저장: {out}")
    plt.show()


if __name__ == "__main__":
    main()
