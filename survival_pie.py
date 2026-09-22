"""titanic.csv 생존 여부 파이차트.

사용법:
    python survival_pie.py           # 라이트 모드
    python survival_pie.py --dark    # 다크 모드
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


def load_counts():
    df = pd.read_csv(CSV)
    counts = df["Survived"].value_counts()
    total = int(counts.sum())
    # 순서: 생존(슬롯 1) → 사망(슬롯 2)
    return [
        {"label": "생존", "value": int(counts.get(1, 0)), "slot": "series_1"},
        {"label": "사망", "value": int(counts.get(0, 0)), "slot": "series_2"},
    ], total


def build_chart(segments, total, mode="light"):
    c = TOKENS[mode]

    plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    values = [s["value"] for s in segments]
    colors = [c[s["slot"]] for s in segments]

    # 조각 사이는 테두리가 아니라 2px 서피스 간격으로 분리한다
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor=c["surface"], linewidth=2, antialiased=True),
    )

    # 조각이 2개뿐이므로 둘 다 직접 라벨링 (텍스트는 잉크 색, 시리즈 색 아님)
    for wedge, seg in zip(wedges, segments):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ha = "left" if x > 0 else "right"
        ax.text(
            x * 1.12, y * 1.12 + 0.06, seg["label"],
            ha=ha, va="center", fontsize=13, fontweight="bold",
            color=c["text_primary"],
        )
        pct = seg["value"] / total * 100
        ax.text(
            x * 1.12, y * 1.12 - 0.07, f"{seg['value']:,}명 · {pct:.1f}%",
            ha=ha, va="center", fontsize=11, color=c["text_secondary"],
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.35, 1.35)

    fig.text(0.5, 0.95, "타이타닉 승객 생존 여부",
             ha="center", fontsize=15, fontweight="bold", color=c["text_primary"])
    fig.text(0.5, 0.895, f"전체 {total:,}명 중 생존 {segments[0]['value']:,}명",
             ha="center", fontsize=11, color=c["text_secondary"])

    # 시리즈가 2개이므로 범례는 항상 둔다 (색만으로 정체성을 전달하지 않기 위해)
    ax.legend(
        handles=[Patch(facecolor=c[s["slot"]], edgecolor="none", label=s["label"])
                 for s in segments],
        loc="lower center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, frameon=False, handlelength=0.9, handleheight=0.9,
        columnspacing=1.8, fontsize=11, labelcolor=c["text_secondary"],
    )

    fig.text(0.5, 0.03, "출처: titanic.csv (Kaggle 학습용 891명 표본)",
             ha="center", fontsize=9, color=c["muted"])

    fig.subplots_adjust(top=0.84, bottom=0.16)
    return fig


def print_table(segments, total):
    """표 형태의 동일 정보 — 값이 색에만 의존하지 않도록."""
    print(f"{'구분':<6}{'인원':>8}{'비율':>9}")
    print("-" * 23)
    for s in segments:
        print(f"{s['label']:<6}{s['value']:>7,}명{s['value'] / total * 100:>8.1f}%")
    print("-" * 23)
    print(f"{'합계':<6}{total:>7,}명{100.0:>8.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    args = parser.parse_args()
    mode = "dark" if args.dark else "light"

    segments, total = load_counts()
    print_table(segments, total)

    fig = build_chart(segments, total, mode)
    out = CSV.parent / f"survival_pie_{mode}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\n저장: {out}")
    plt.show()


if __name__ == "__main__":
    main()
