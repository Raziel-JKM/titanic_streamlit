"""titanic.csv 성별 × 객실등급 생존 여부 파이차트 (소형 배수 2×3).

사용법:
    python survival_pie_by_sex_pclass.py           # 라이트 모드
    python survival_pie_by_sex_pclass.py --dark    # 다크 모드
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

ROWS = [("male", "남성"), ("female", "여성")]          # 행 = 성별
COLS = [(1, "1등석"), (2, "2등석"), (3, "3등석")]       # 열 = 객실등급


def load_grid():
    df = pd.read_csv(CSV)
    table = pd.crosstab([df["Sex"], df["Pclass"]], df["Survived"])
    grid = []
    for sex, _ in ROWS:
        row = []
        for pclass, _ in COLS:
            values = [int(table.loc[(sex, pclass), s["key"]]) for s in SERIES]
            total = sum(values)
            row.append({
                "values": values,
                "total": total,
                "rate": values[0] / total * 100,  # 생존율
            })
        grid.append(row)
    return grid, int(table.to_numpy().sum())


# 96.8% 같은 극단 비율에서는 얇은 조각의 라벨이 12시 부근에 서므로,
# 파이를 아래로 내려 헤더 텍스트와 겹치지 않게 한다
CENTER = (0.0, -0.15)


def draw_panel(ax, panel, c, col_title=None):
    wedges, _ = ax.pie(
        panel["values"],
        colors=[c[s["slot"]] for s in SERIES],
        startangle=90,
        counterclock=False,
        radius=1.0,
        center=CENTER,
        # 조각 사이는 테두리가 아니라 2px 서피스 간격으로 분리한다
        wedgeprops=dict(edgecolor=c["surface"], linewidth=2, antialiased=True),
    )

    # 패널이 6개라 조각 라벨은 인원수만 — 생존율은 패널 부제가 전달한다
    for wedge, value in zip(wedges, panel["values"]):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ax.text(CENTER[0] + x * 1.16, CENTER[1] + y * 1.16, f"{value:,}명",
                ha="left" if x > 0 else "right", va="center",
                fontsize=10, color=c["text_secondary"])

    ax.set_aspect("equal")
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.5, 2.12)

    if col_title:
        ax.text(0, 1.98, col_title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=c["text_primary"])
    ax.text(0, 1.58, f"생존율 {panel['rate']:.1f}%", ha="center", va="center",
            fontsize=12, fontweight="bold", color=c["text_primary"])
    ax.text(0, 1.41, f"{panel['total']:,}명 중 {panel['values'][0]:,}명",
            ha="center", va="center", fontsize=10, color=c["text_secondary"])


def build_chart(grid, total, mode="light"):
    c = TOKENS[mode]

    plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), dpi=160)
    fig.patch.set_facecolor(c["surface"])

    for r, row in enumerate(grid):
        for col, panel in enumerate(row):
            ax = axes[r][col]
            ax.set_facecolor(c["surface"])
            draw_panel(ax, panel, c, col_title=COLS[col][1] if r == 0 else None)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.13,
                        wspace=0.0, hspace=0.05)

    # 행 라벨(성별)은 왼쪽 여백에 한 번만 — 각 패널 제목을 반복하지 않는다
    for r, (_, label) in enumerate(ROWS):
        pos = axes[r][0].get_position()
        fig.text(0.035, pos.y0 + pos.height * 0.48, label,
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 color=c["text_primary"])

    fig.text(0.5, 0.955, "타이타닉 승객 성별 · 객실등급별 생존 여부",
             ha="center", fontsize=16, fontweight="bold", color=c["text_primary"])
    fig.text(0.5, 0.912,
             f"전체 {total:,}명 · 생존율은 1등석 여성 96.8%에서 3등석 남성 13.5%까지 벌어진다",
             ha="center", fontsize=11, color=c["text_secondary"])

    # 시리즈가 2개이므로 범례는 항상 둔다 (6개 패널이 공유)
    fig.legend(
        handles=[Patch(facecolor=c[s["slot"]], edgecolor="none", label=s["label"])
                 for s in SERIES],
        loc="lower center", bbox_to_anchor=(0.5, 0.055),
        ncol=2, frameon=False, handlelength=0.9, handleheight=0.9,
        columnspacing=1.8, fontsize=11, labelcolor=c["text_secondary"],
    )

    fig.text(0.5, 0.022, "출처: titanic.csv (Kaggle 학습용 891명 표본)",
             ha="center", fontsize=9, color=c["muted"])
    return fig


def print_table(grid, total):
    """표 형태의 동일 정보 — 값이 색에만 의존하지 않도록."""
    header = f"{'구분':<12}{'생존':>9}{'사망':>9}{'합계':>9}{'생존율':>9}"
    print(header)
    print("-" * 48)
    for r, (_, sex_label) in enumerate(ROWS):
        for col, (_, class_label) in enumerate(COLS):
            p = grid[r][col]
            name = f"{sex_label} {class_label}"
            print(f"{name:<12}{p['values'][0]:>8,}명{p['values'][1]:>8,}명"
                  f"{p['total']:>8,}명{p['rate']:>8.1f}%")
    surv = sum(p["values"][0] for row in grid for p in row)
    dead = sum(p["values"][1] for row in grid for p in row)
    print("-" * 48)
    print(f"{'합계':<12}{surv:>8,}명{dead:>8,}명{total:>8,}명"
          f"{surv / total * 100:>8.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    args = parser.parse_args()
    mode = "dark" if args.dark else "light"

    grid, total = load_grid()
    print_table(grid, total)

    fig = build_chart(grid, total, mode)
    out = CSV.parent / f"survival_pie_by_sex_pclass_{mode}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\n저장: {out}")
    plt.show()


if __name__ == "__main__":
    main()
