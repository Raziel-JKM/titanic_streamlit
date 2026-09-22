"""titanic.csv 가설 검증용 시각화 (titanic 분석 가설.md 의 우선순위 1~5위).

    age      H1   — 연령구간 × 성별 생존율 (꺾은선)
    agegrid  H2   — 등급 × 성별 × 연령구간 생존율 (히트맵)
    title    H12  — 호칭 그룹별 생존율
    ticket   H7   — 티켓 일행 크기별 생존율 / 일행 내 생사 일치도
    cabin    H10  — Cabin 기록 유무별 생존율 (등급 통제)
    family   H4, H5 — 가족 규모별 생존율
    fare     H8   — 등급 내 1인당 요금 4분위별 생존율
    embarked H11  — 항구별 등급 구성 / 항구 × 등급 생존율
    deck     H9   — 갑판별 생존율 / 갑판별 등급 구성
    mother   H13  — Mrs 중 자녀 동반 여부별 생존율

사용법:
    python hypothesis_viz.py                 # 전체, 라이트 모드
    python hypothesis_viz.py --dark          # 전체, 다크 모드
    python hypothesis_viz.py --only age      # 하나만
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CSV = Path(__file__).parent / "titanic.csv"
OUT_DIR = CSV.parent

# 색상 토큰 (검증된 기본 팔레트)
TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series_1": "#2a78d6",   # blue
        "series_2": "#eb6834",   # orange
        # 객실등급은 순서형 → 한 색상의 순차 램프 (밝을수록 낮은 등급)
        "ord_1": "#1c5cab", "ord_2": "#3987e5", "ord_3": "#86b6ef",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series_1": "#3987e5",
        "series_2": "#d95926",
        # 다크 서피스에서는 방향을 뒤집는다 (가장 어두운 단계도 2:1 확보)
        "ord_1": "#b7d3f6", "ord_2": "#5598e7", "ord_3": "#184f95",
    },
}

# 표본이 이보다 작은 칸은 비율을 믿지 말라고 표시한다 (가설 문서 '지킬 것' 2번)
SMALL_N = 20

TITLE_MAP = {
    "Mr": "Mr", "Master": "Master",
    "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Mrs": "Mrs", "Mme": "Mrs",
    "Dr": "직책", "Rev": "직책", "Col": "직책", "Major": "직책", "Capt": "직책",
    "Lady": "귀족", "Sir": "귀족", "Don": "귀족",
    "the Countess": "귀족", "Jonkheer": "귀족",
}

AGE_BINS = [0, 12, 18, 40, 60, 100]
AGE_LABELS = ["0–12", "13–18", "19–40", "41–60", "61+"]


# ---------------------------------------------------------------- 데이터 준비

def prepare():
    df = pd.read_csv(CSV)
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
    df["TitleGroup"] = df["Title"].map(TITLE_MAP).fillna("기타")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("size")
    df["HasCabin"] = df["Cabin"].notna()
    df["AgeBand"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS)
    return df


def rate_table(df, by):
    """생존율 + 표본 크기. 비율만 있는 표는 만들지 않는다."""
    g = df.groupby(by, observed=True)["Survived"]
    out = pd.DataFrame({"n": g.size(), "survived": g.sum()})
    out["rate"] = out["survived"] / out["n"] * 100
    return out


# ------------------------------------------------------------------ 공통 스타일

def use_fonts():
    plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def style_axes(ax, c, grid_axis="y"):
    """격자·축은 서피스에서 한 단계만 떨어진 헤어라인으로 물러나 있게."""
    ax.set_facecolor(c["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=c["muted"], length=0, labelsize=10)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=c["grid"], linewidth=1, linestyle="-")


def pct_axis(ax, c, axis="y"):
    ticks = [0, 25, 50, 75, 100]
    if axis == "y":
        ax.set_ylim(0, 105)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t}%" for t in ticks])
    else:
        ax.set_xlim(0, 105)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}%" for t in ticks])


def titles(fig, c, title, subtitle, source=True):
    fig.text(0.5, 0.955, title, ha="center", fontsize=15,
             fontweight="bold", color=c["text_primary"])
    fig.text(0.5, 0.908, subtitle, ha="center", fontsize=11,
             color=c["text_secondary"])
    if source:
        fig.text(0.5, 0.025, "출처: titanic.csv (Kaggle 학습용 891명 표본)",
                 ha="center", fontsize=9, color=c["muted"])


def baseline_rule(ax, c, value, label=None, orient="h"):
    """전체 생존율 기준선 — 점선이 아니라 실선 헤어라인.

    label=None 이면 선만 긋는다 (막대와 겹칠 자리밖에 없을 때).
    """
    if orient == "h":
        ax.axhline(value, color=c["baseline"], linewidth=1, zorder=1)
        if label:
            ax.text(ax.get_xlim()[1], value + 2, label, ha="right",
                    va="bottom", fontsize=9, color=c["muted"])
    else:
        ax.axvline(value, color=c["baseline"], linewidth=1, zorder=1)
        ax.text(value, ax.get_ylim()[1], f" {label}", ha="left", va="top",
                fontsize=9, color=c["muted"])


# --------------------------------------------------- fig1 · H1, H2 연령 × 성별

def fig_age(df, c):
    """연령구간별 생존율, 성별 2개 시리즈 (등급은 통제하지 않음).

    연령구간은 순서가 있는 축이라 꺾은선이 맞다 — 기울기가 곧 '나이 효과'다.
    등급까지 쪼개면 칸이 n=1~3까지 작아져 선이 튄다. 3원 분해는 히트맵(agegrid)에서.
    """
    sub = df.dropna(subset=["Age"])
    tab = rate_table(sub, ["Sex", "AgeBand"])

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c)
    pct_axis(ax, c)

    series = [("female", "여성", "series_1"), ("male", "남성", "series_2")]
    xs = np.arange(len(AGE_LABELS))
    ax.set_xticks(xs)
    ax.set_xticklabels(AGE_LABELS, fontsize=11)
    ax.set_xlim(-0.35, len(AGE_LABELS) - 0.65)

    for row, (sex, label, slot) in enumerate(series):
        rates = [tab.loc[(sex, b), "rate"] for b in AGE_LABELS]
        ns = [int(tab.loc[(sex, b), "n"]) for b in AGE_LABELS]
        ax.plot(xs, rates, color=c[slot], linewidth=2, zorder=3)
        # 표본이 작은 점은 속을 비워 둔다 — 색 외의 두 번째 신호
        for x, rate, n in zip(xs, rates, ns):
            ax.plot(x, rate, marker="o", markersize=9,
                    markerfacecolor=c["surface"] if n < SMALL_N else c[slot],
                    markeredgecolor=c[slot], markeredgewidth=2,
                    zorder=4, clip_on=False)
        # 끝점만 직접 라벨링
        ax.text(xs[-1] + 0.12, rates[-1], label, va="center", ha="left",
                fontsize=11, fontweight="bold", color=c["text_primary"])
        # 표본 크기 행 (성별로 한 줄씩)
        for x, n in zip(xs, ns):
            ax.annotate(f"{n}", (x, 0), xytext=(0, -40 - row * 17),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=c["muted"])
        ax.annotate(f"{label} 표본", (0, 0), xytext=(-24, -40 - row * 17),
                    textcoords="offset points", ha="right",
                    fontsize=9, color=c["muted"])

    # 기준선은 891명 전체(38.4%)가 아니라 Age가 기록된 714명 기준이다
    baseline_rule(ax, c, sub["Survived"].mean() * 100,
                  f"Age 기록자 전체 {sub['Survived'].mean() * 100:.1f}%")

    # 시리즈가 2개이므로 끝점 직접 라벨과 별개로 범례도 둔다
    fig.legend(
        handles=[Line2D([], [], color=c[s], linewidth=2, marker="o",
                        markersize=8, label=lab) for _, lab, s in series],
        loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False,
        columnspacing=2.0, fontsize=11, labelcolor=c["text_secondary"])

    titles(fig, c, "연령구간별 생존율 — 성별 (H1)",
           f"Age 결측 177명 제외 · {len(sub)}명 기준 · "
           f"속 빈 점은 n<{SMALL_N}이라 비율을 믿기 어려운 칸")

    fig.subplots_adjust(left=0.13, right=0.92, top=0.82, bottom=0.28)
    return fig, tab


def fig_age_grid(df, c):
    """등급 × 성별 × 연령구간 — 3원 분해는 격자이므로 히트맵.

    칸마다 n을 같이 찍어, n=1짜리 100%를 선 그래프처럼 과장하지 않는다.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sub = df.dropna(subset=["Age"])
    tab = rate_table(sub, ["Sex", "Pclass", "AgeBand"])

    # 순차 인코딩은 한 가지 색상, 밝음→어둠 (다크 모드에서는 반대로)
    steps = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
    if c is TOKENS["dark"]:
        steps = steps[::-1]
    cmap = LinearSegmentedColormap.from_list("blue_seq", steps)
    norm = Normalize(0, 100)

    rows = [(sex, pclass) for sex in ("female", "male") for pclass in (1, 2, 3)]
    row_labels = {"female": "여성", "male": "남성"}

    fig, ax = plt.subplots(figsize=(11.0, 6.4), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=c["muted"], length=0, labelsize=11)

    for r, (sex, pclass) in enumerate(rows):
        for col, band in enumerate(AGE_LABELS):
            key = (sex, pclass, band)
            if key in tab.index:
                rate, n = tab.loc[key, "rate"], int(tab.loc[key, "n"])
                fill = cmap(norm(rate))
                label, nlabel = f"{rate:.0f}%", f"n={n}"
            else:
                rate, n, fill = None, 0, c["surface"]
                label, nlabel = "–", ""
            # 표본이 작은 칸은 작게 그린다. 색을 흐리게 하면 범례 스케일과
            # 어긋나므로(흐린 100%가 진한 50%처럼 보인다) 크기로만 낮춘다.
            pad = 0.14 if 0 < n < SMALL_N else 0.0
            # 칸 사이 2px 서피스 간격
            ax.add_patch(plt.Rectangle((col + pad, r + pad), 1 - 2 * pad,
                                       1 - 2 * pad, facecolor=fill,
                                       edgecolor=c["surface"], linewidth=2))
            if rate is None:
                ax.text(col + 0.5, r + 0.5, label, ha="center", va="center",
                        fontsize=11, color=c["muted"])
                continue
            ink = _ink_on(fill, c)
            ax.text(col + 0.5, r + 0.38, label, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=ink)
            ax.text(col + 0.5, r + 0.68, nlabel, ha="center", va="center",
                    fontsize=9, color=ink, alpha=0.75)

    ax.set_xlim(0, len(AGE_LABELS))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(AGE_LABELS)) + 0.5)
    ax.set_xticklabels(AGE_LABELS)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"{row_labels[s]} {p}등석" for s, p in rows],
                       color=c["text_primary"])

    # 순차 색은 스케일 범례가 반드시 따라붙는다
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        orientation="horizontal", fraction=0.05, pad=0.09,
                        aspect=40)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(colors=c["muted"], length=0, labelsize=9)
    cbar.set_label("생존율", color=c["text_secondary"], fontsize=10)

    titles(fig, c, "등급 × 성별 × 연령구간 생존율 (H2)",
           f"Age 결측 177명 제외 · {len(sub)}명 기준 · "
           f"작게 그려진 칸은 n<{SMALL_N}이라 비율을 읽지 말 것")

    fig.subplots_adjust(left=0.13, right=0.97, top=0.79, bottom=0.14)
    return fig, tab


def _ink_on(rgba, c):
    """칸 색이 어두우면 흰 글씨, 밝으면 검은 글씨."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.55 else "#0b0b0b"


def class_colors(c):
    return [c["ord_1"], c["ord_2"], c["ord_3"]]


def stacked_composition(ax, c, counts, row_labels, min_label_pct=12):
    """100% 누적 가로 막대 — 등급 구성처럼 '부분-전체'를 볼 때.

    조각이 좁으면 라벨을 넣지 않는다(막대 밖으로 삐져나오거나 잘리는 것보다 낫다).
    """
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    ys = np.arange(len(pct))
    lefts = np.zeros(len(pct))
    for col, color in zip(pct.columns, class_colors(c)):
        vals = pct[col].to_numpy()
        # 조각 사이 2px 서피스 간격
        ax.barh(ys, vals, left=lefts, height=0.5, color=color,
                edgecolor=c["surface"], linewidth=2, zorder=3)
        for y, (v, x0) in enumerate(zip(vals, lefts)):
            if v >= min_label_pct:
                ax.text(x0 + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color=_ink_on(to_rgb(color), c), zorder=4)
        lefts += vals
    ax.set_yticks(ys)
    ax.set_yticklabels(row_labels, fontsize=10, color=c["text_primary"])
    ax.invert_yaxis()
    return pct


# ------------------------------------------------------------ fig6 · H8 요금

def fig_fare(df, c):
    """등급 안에서 1인당 요금이 더 비싸면 더 살아남았는가.

    Fare 는 티켓 단위 합계라 그대로 쓰면 가족 규모와 교락된다 →
    FarePerPerson 으로 정규화한 뒤, 등급 '안에서' 4분위로 나눠 본다.
    """
    sub = df[df["Fare"] > 0].copy()
    sub["FarePerPerson"] = sub["Fare"] / sub["TicketGroupSize"]
    sub["FareQ"] = sub.groupby("Pclass")["FarePerPerson"].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1\n(저)", "Q2", "Q3", "Q4\n(고)"],
                          duplicates="drop"))
    tab = rate_table(sub, ["Pclass", "FareQ"])
    medians = sub.groupby("Pclass")["FarePerPerson"].median()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), dpi=160, sharey=True)
    fig.patch.set_facecolor(c["surface"])

    for ax, pclass in zip(axes, [1, 2, 3]):
        style_axes(ax, c)
        pct_axis(ax, c)
        rows = tab.loc[pclass]
        xs = np.arange(len(rows))
        ax.bar(xs, rows["rate"], width=0.55, color=c["series_1"], zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(rows.index, fontsize=10)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        for x, (rate, n) in enumerate(zip(rows["rate"], rows["n"])):
            ax.text(x, rate + 3, f"{rate:.0f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=c["text_primary"])
            ax.annotate(f"{int(n)}명", (x, 0), xytext=(0, -42),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=c["muted"])
        ax.set_title(f"{pclass}등석  ·  1인당 중앙값 £{medians[pclass]:.2f}",
                     fontsize=12, fontweight="bold", color=c["text_primary"],
                     pad=12)
        # 막대가 가로선 라벨 자리를 다 차지하므로 선만 긋고 설명은 부제로 뺀다
        baseline_rule(ax, c, df[df["Pclass"] == pclass]["Survived"].mean() * 100)

    titles(fig, c, "등급 안에서 1인당 요금이 생존율을 더 설명하는가 (H8)",
           "FarePerPerson = Fare ÷ 같은 티켓 인원 · Fare=0인 15명 제외 · "
           "가로선은 각 등급의 평균 생존율 — 막대가 평평하면 요금은 등급의 그림자일 뿐")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.17, wspace=0.12)
    return fig, tab


# ---------------------------------------------------------- fig7 · H11 승선항

def fig_embarked(df, c):
    """항구별 생존율 차이가 등급 구성으로 설명되는지 — 구성과 층별 생존율을 나란히."""
    sub = df.dropna(subset=["Embarked"])
    ports = [("C", "Cherbourg (C)"), ("Q", "Queenstown (Q)"),
             ("S", "Southampton (S)")]
    comp = pd.crosstab(sub["Embarked"], sub["Pclass"]).loc[[p for p, _ in ports]]
    tab = rate_table(sub, ["Embarked", "Pclass"])
    overall = rate_table(sub, "Embarked")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.8), dpi=160)
    fig.patch.set_facecolor(c["surface"])

    # --- 왼쪽: 항구별 등급 구성 ---
    style_axes(ax1, c, grid_axis="x")
    pct_axis(ax1, c, axis="x")
    stacked_composition(ax1, c, comp, [lab for _, lab in ports])
    for y, (p, _) in enumerate(ports):
        ax1.annotate(f"{int(overall.loc[p, 'n'])}명", (105, y), xytext=(6, 0),
                     textcoords="offset points", va="center", ha="left",
                     fontsize=9, color=c["muted"], annotation_clip=False)
    ax1.set_title("항구별 등급 구성", fontsize=12, fontweight="bold",
                  color=c["text_primary"], pad=12)

    # --- 오른쪽: 항구 × 등급 생존율 ---
    style_axes(ax2, c)
    pct_axis(ax2, c)
    width = 0.24
    xs = np.arange(len(ports))
    for i, (pclass, color) in enumerate(zip([1, 2, 3], class_colors(c))):
        rates, ns = [], []
        for p, _ in ports:
            row = tab.loc[(p, pclass)]
            rates.append(row["rate"])
            ns.append(int(row["n"]))
        offset = (i - 1) * (width + 0.02)
        ax2.bar(xs + offset, rates, width=width, color=color, zorder=3)
        for x, rate, n in zip(xs + offset, rates, ns):
            ax2.text(x, rate + 9, f"{rate:.0f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color=c["text_primary"])
            ax2.text(x, rate + 3, f"n={n}" + (" ⚠" if n < SMALL_N else ""),
                     ha="center", va="bottom", fontsize=8, color=c["muted"])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [f"{lab}\n전체 {overall.loc[p, 'rate']:.1f}%" for p, lab in ports],
        fontsize=10, color=c["text_primary"])
    ax2.set_xlim(-0.6, len(ports) - 0.4)
    ax2.set_title("항구 × 등급 생존율", fontsize=12, fontweight="bold",
                  color=c["text_primary"], pad=12)

    fig.legend(handles=[Patch(facecolor=col, label=f"{p}등석")
                        for p, col in zip([1, 2, 3], class_colors(c))],
               loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3,
               frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=1.8, fontsize=11, labelcolor=c["text_secondary"])

    titles(fig, c, "승선 항구 효과는 등급 구성으로 설명되는가 (H11)",
           f"Embarked 결측 2명 제외 · ⚠는 n<{SMALL_N} · "
           "오른쪽에서 등급을 맞춰 비교했을 때 항구 간 차이가 남는지가 핵심")

    fig.subplots_adjust(left=0.13, right=0.94, top=0.80, bottom=0.18, wspace=0.42)
    return fig, (tab, comp)


# ------------------------------------------------------------ fig8 · H9 갑판

def fig_deck(df, c):
    """갑판별 생존율 — 등급과 거의 같은 정보라는 점을 오른쪽 패널이 드러낸다."""
    sub = df.dropna(subset=["Cabin"]).copy()
    sub["Deck"] = sub["Cabin"].str[0]
    order = [d for d in "ABCDEFGT" if d in set(sub["Deck"])]
    tab = rate_table(sub, "Deck").reindex(order)
    comp = pd.crosstab(sub["Deck"], sub["Pclass"]).reindex(order).fillna(0)
    for pclass in (1, 2, 3):
        if pclass not in comp.columns:
            comp[pclass] = 0
    comp = comp[[1, 2, 3]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 6.0), dpi=160)
    fig.patch.set_facecolor(c["surface"])

    # --- 왼쪽: 갑판별 생존율 ---
    style_axes(ax1, c, grid_axis="x")
    pct_axis(ax1, c, axis="x")
    ys = np.arange(len(tab))
    ax1.barh(ys, tab["rate"], height=0.5, color=c["series_1"], zorder=3)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f"{d}갑판" for d in tab.index], fontsize=11,
                        color=c["text_primary"])
    ax1.invert_yaxis()
    for y, (rate, n) in enumerate(zip(tab["rate"], tab["n"])):
        note = f"{rate:.0f}%  " + (f"n={int(n)}" if n >= SMALL_N
                                   else f"n={int(n)} ⚠")
        ax1.text(rate + 2, y, note, va="center", ha="left", fontsize=10,
                 color=c["text_primary"] if n >= SMALL_N else c["muted"])
    ax1.set_title("갑판별 생존율", fontsize=12, fontweight="bold",
                  color=c["text_primary"], pad=12)

    # --- 오른쪽: 갑판별 등급 구성 (교란 확인) ---
    style_axes(ax2, c, grid_axis="x")
    pct_axis(ax2, c, axis="x")
    stacked_composition(ax2, c, comp, [f"{d}갑판" for d in comp.index])
    ax2.set_title("갑판별 등급 구성", fontsize=12, fontweight="bold",
                  color=c["text_primary"], pad=12)
    ax2.legend(handles=[Patch(facecolor=col, label=f"{p}등석")
                        for p, col in zip([1, 2, 3], class_colors(c))],
               loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
               frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=1.8, fontsize=10, labelcolor=c["text_secondary"])

    titles(fig, c, "갑판(Cabin 앞글자)별 생존율 (H9)",
           f"Cabin 기록이 있는 {len(sub)}명만 · ⚠는 n<{SMALL_N} · "
           "오른쪽을 보면 A~C갑판은 전원 1등석이라 갑판과 등급을 분리할 수 없다")

    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.16, wspace=0.34)
    return fig, (tab, comp)


# --------------------------------------------------------- fig9 · H13 어머니

def fig_mother(df, c):
    """Mrs 중 자녀 동반 여부 — '있음 vs 없음' 대비라 강조(accent + 회색) 배색."""
    sub = df[df["TitleGroup"] == "Mrs"].copy()
    sub["kids"] = np.where(sub["Parch"] > 0, True, False)
    tab = rate_table(sub, ["Pclass", "kids"])
    overall = rate_table(sub, "kids")

    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c)
    pct_axis(ax, c)

    groups = [("1등석", 1), ("2등석", 2), ("3등석", 3), ("전체", None)]
    series = [(True, "자녀 동반", c["series_1"]),
              (False, "자녀 없음", c["muted"])]
    width = 0.3
    xs = np.arange(len(groups))
    picked = {}

    for i, (kids, label, color) in enumerate(series):
        rates, ns = [], []
        for _, pclass in groups:
            row = overall.loc[kids] if pclass is None else tab.loc[(pclass, kids)]
            rates.append(row["rate"])
            ns.append(int(row["n"]))
        picked[kids] = rates
        offset = (i - 0.5) * (width + 0.02)
        ax.bar(xs + offset, rates, width=width, color=color, zorder=3)
        for x, rate, n in zip(xs + offset, rates, ns):
            ax.text(x, rate + 3, f"{rate:.0f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=c["text_primary"])
            ax.annotate(f"{n}명", (x, 0), xytext=(0, -26),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=c["muted"])

    # 셀이 작아 유의성보다 효과 크기를 읽어야 하므로 차이를 %p로 명시
    for x, (a, b) in enumerate(zip(picked[True], picked[False])):
        ax.annotate(f"{a - b:+.1f}%p", (x, 0), xytext=(0, -46),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=c["text_secondary"])
    ax.annotate("차이", (0, 0), xytext=(-40, -46), textcoords="offset points",
                ha="right", fontsize=10, color=c["muted"])

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11,
                       color=c["text_primary"])
    ax.set_xlim(-0.6, len(groups) - 0.4)

    ax.legend(handles=[Patch(facecolor=col, label=lab) for _, lab, col in series],
              loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              frameon=False, handlelength=0.9, handleheight=0.9,
              columnspacing=1.8, fontsize=11, labelcolor=c["text_secondary"])

    titles(fig, c, "Mrs 중 자녀 동반 여부별 생존율 (H13)",
           f"Mrs·Mme {len(sub)}명 · Parch>0 을 '자녀 동반'으로 간주 · "
           "칸이 작으므로 유의성이 아니라 차이의 크기와 방향만 읽을 것")

    fig.subplots_adjust(left=0.09, right=0.97, top=0.81, bottom=0.25)
    return fig, tab


# --------------------------------------------------------- fig2 · H12 호칭 그룹

def fig_title(df, c):
    """호칭 그룹별 생존율 — 단일 시리즈이므로 한 가지 색, 범례 없음.

    이름이 길고 항목이 많아 가로 막대.
    """
    tab = rate_table(df, "TitleGroup").sort_values("rate")
    overall = df["Survived"].mean() * 100

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c, grid_axis="x")
    pct_axis(ax, c, axis="x")

    ys = np.arange(len(tab))
    ax.barh(ys, tab["rate"], height=0.55, color=c["series_1"], zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(tab.index, fontsize=11, color=c["text_primary"])
    ax.set_ylim(-0.7, len(tab) - 0.3)

    # 막대가 몇 개 안 되므로 전부 직접 라벨링 — 값 읽기를 툴팁에 의존시키지 않는다
    for y, (rate, n) in enumerate(zip(tab["rate"], tab["n"])):
        ax.text(rate + 1.5, y + 0.11, f"{rate:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold", color=c["text_primary"])
        ax.text(rate + 1.5, y - 0.16, f"{int(n)}명", va="center", ha="left",
                fontsize=9, color=c["muted"])

    baseline_rule(ax, c, overall, f"전체 {overall:.1f}%", orient="v")

    titles(fig, c, "호칭 그룹별 생존율 (H12)",
           "Name에서 추출 · Miss=Miss/Mlle/Ms, Mrs=Mrs/Mme, "
           "직책=Dr·Rev·Col·Major·Capt, 귀족=Lady·Sir·Don·Countess·Jonkheer")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.11)
    return fig, tab


# --------------------------------------------------------- fig3 · H7 티켓 일행

def fig_ticket(df, c):
    """왼쪽: 일행 크기별 생존율. 오른쪽: 일행 내 생사가 갈리는지 여부.

    오른쪽이 H7의 핵심 — '일행의 생사가 함께 움직인다'면 '혼재'가 드물어야 한다.
    """
    band = df["TicketGroupSize"].clip(upper=5)
    labels = {1: "1명\n(혼자)", 2: "2명", 3: "3명", 4: "4명", 5: "5명+"}
    left = rate_table(df.assign(band=band), "band")

    # 2명 이상 일행의 결과 구성
    g = df.groupby("Ticket")["Survived"].agg(["size", "sum"])
    g = g[g["size"] >= 2].copy()
    g["band"] = g["size"].clip(upper=5)
    g["outcome"] = np.where(g["sum"] == 0, "all_died",
                    np.where(g["sum"] == g["size"], "all_surv", "mixed"))
    comp = (g.groupby(["band", "outcome"]).size()
              .unstack(fill_value=0)
              .reindex(columns=["all_surv", "mixed", "all_died"], fill_value=0))
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])

    # --- 왼쪽: 생존율 ---
    style_axes(ax1, c)
    pct_axis(ax1, c)
    xs = np.arange(len(left))
    ax1.bar(xs, left["rate"], width=0.55, color=c["series_1"], zorder=3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([labels[b] for b in left.index], fontsize=10)
    for x, (rate, n) in enumerate(zip(left["rate"], left["n"])):
        ax1.text(x, rate + 3, f"{rate:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=c["text_primary"])
        ax1.annotate(f"{int(n)}명", (x, 0), xytext=(0, -42),
                     textcoords="offset points", ha="center",
                     fontsize=9, color=c["muted"])
    ax1.set_title("일행 크기별 생존율", fontsize=12, fontweight="bold",
                  color=c["text_primary"], pad=12)

    # --- 오른쪽: 일행 내 결과 구성 (부분-전체 → 누적 막대) ---
    style_axes(ax2, c, grid_axis="x")
    pct_axis(ax2, c, axis="x")
    seg = [("all_surv", "전원 생존", c["series_1"]),
           ("mixed", "생사 혼재", c["muted"]),
           ("all_died", "전원 사망", c["series_2"])]
    ys = np.arange(len(comp_pct))
    lefts = np.zeros(len(comp_pct))
    for key, _, color in seg:
        vals = comp_pct[key].to_numpy()
        # 조각 사이 2px 서피스 간격 — 테두리를 두르지 않는다
        ax2.barh(ys, vals, left=lefts, height=0.5, color=color,
                 edgecolor=c["surface"], linewidth=2, zorder=3)
        lefts += vals
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{labels[b].splitlines()[0]} 일행" for b in comp_pct.index],
                        fontsize=10, color=c["text_primary"])
    ax2.invert_yaxis()
    for y, b in enumerate(comp_pct.index):
        ax2.annotate(f"{int(comp.loc[b].sum())}팀", (105, y), xytext=(6, 0),
                     textcoords="offset points", va="center", ha="left",
                     fontsize=9, color=c["muted"], annotation_clip=False)
    ax2.set_title("2명 이상 일행의 결과 구성", fontsize=12, fontweight="bold",
                  color=c["text_primary"], pad=12)
    ax2.legend(handles=[Patch(facecolor=col, label=lab) for _, lab, col in seg],
               loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
               frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=1.6, fontsize=10, labelcolor=c["text_secondary"])

    titles(fig, c, "같은 티켓 = 일행인가, 그리고 일행의 생사는 함께 움직이는가 (H7)",
           f"티켓 681종 · 2명 이상 일행 {len(g)}팀 · "
           "'생사 혼재'가 드물수록 일행 단위로 운명이 갈렸다는 뜻")

    fig.subplots_adjust(left=0.06, right=0.93, top=0.81, bottom=0.17, wspace=0.28)
    return fig, (left, comp)


# ------------------------------------------------------- fig4 · H10 Cabin 결측

def fig_cabin(df, c):
    """Cabin 기록 유무별 생존율 — 등급을 통제해도 차이가 남는지가 질문.

    두 시리즈가 '있음 vs 없음'이라 강조(accent + 회색) 배색을 쓴다.
    앞선 차트의 파랑/주황(생존/사망)과 혼동되지 않게 하기 위해서다.
    """
    tab = rate_table(df, ["Pclass", "HasCabin"])
    overall = rate_table(df, "HasCabin")

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c)
    pct_axis(ax, c)

    groups = [("1등석", 1), ("2등석", 2), ("3등석", 3), ("전체", None)]
    series = [(True, "Cabin 기록 있음", c["series_1"]),
              (False, "Cabin 기록 없음", c["muted"])]
    width = 0.32
    xs = np.arange(len(groups))

    for i, (has, label, color) in enumerate(series):
        rates, ns = [], []
        for _, pclass in groups:
            row = overall.loc[has] if pclass is None else tab.loc[(pclass, has)]
            rates.append(row["rate"])
            ns.append(int(row["n"]))
        # 막대 사이 2px 간격은 width 로 확보한다
        offset = (i - 0.5) * (width + 0.02)
        ax.bar(xs + offset, rates, width=width, color=color, zorder=3)
        for x, rate, n in zip(xs + offset, rates, ns):
            ax.text(x, rate + 3, f"{rate:.0f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=c["text_primary"])
            ax.annotate(f"{n}명", (x, 0), xytext=(0, -26),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=c["muted"])

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11,
                       color=c["text_primary"])
    ax.set_xlim(-0.6, len(groups) - 0.4)

    titles(fig, c, "Cabin 기록 유무별 생존율 (H10)",
           "Cabin 결측 687명(77.1%) · 등급을 통제해도 차이가 남으면 "
           "'결측'은 버릴 값이 아니라 피처다")

    ax.legend(handles=[Patch(facecolor=col, label=lab) for _, lab, col in series],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, handlelength=0.9, handleheight=0.9,
              columnspacing=1.8, fontsize=11, labelcolor=c["text_secondary"])

    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.19)
    return fig, tab


# ----------------------------------------------------- fig5 · H4, H5 가족 규모

def fig_family(df, c):
    """가족 규모별 생존율 — 역U자인지 보려면 모양이 중요하므로 꺾은선.

    표본 크기는 축 아래 별도 행으로 깔아 둔다(임상 그래프의 risk table 방식).
    """
    band = df["FamilySize"].clip(upper=7)
    tab = rate_table(df.assign(band=band), "band")
    labels = {1: "1명\n(혼자)", 2: "2명", 3: "3명", 4: "4명",
              5: "5명", 6: "6명", 7: "7명+"}
    overall = df["Survived"].mean() * 100

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c)
    pct_axis(ax, c)

    xs = np.arange(len(tab))
    ax.plot(xs, tab["rate"], color=c["series_1"], linewidth=2, marker="o",
            markersize=9, markeredgecolor=c["surface"], markeredgewidth=2,
            zorder=3, clip_on=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[b] for b in tab.index], fontsize=10)
    ax.set_xlim(-0.4, len(tab) - 0.6)

    # 직접 라벨은 선택적으로 — 최고점과 양 끝만
    peak = int(np.argmax(tab["rate"].to_numpy()))
    for i in {0, peak, len(tab) - 1}:
        rate = tab["rate"].iloc[i]
        ax.text(xs[i], rate + 5, f"{rate:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=c["text_primary"])

    # 표본 크기 행
    for x, n in zip(xs, tab["n"]):
        ax.annotate(f"{int(n)}", (x, 0), xytext=(0, -40),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=c["muted"])
    ax.annotate("표본", (0, 0), xytext=(-34, -40), textcoords="offset points",
                ha="right", fontsize=9, color=c["muted"])

    baseline_rule(ax, c, overall, f"전체 {overall:.1f}%")

    titles(fig, c, "가족 규모별 생존율 (H4, H5)",
           "FamilySize = SibSp + Parch + 1 · 7명 이상은 묶음 · "
           "역U자면 H5 성립, 단조 증가면 H4만 성립")

    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.20)
    return fig, tab


# ------------------------------------------------------------------ 실행

FIGURES = {
    "age": ("viz_age_survival", fig_age),
    "agegrid": ("viz_age_class_grid", fig_age_grid),
    "title": ("viz_title_survival", fig_title),
    "ticket": ("viz_ticket_group", fig_ticket),
    "cabin": ("viz_cabin_missing", fig_cabin),
    "family": ("viz_family_size", fig_family),
    "fare": ("viz_fare_quartile", fig_fare),
    "embarked": ("viz_embarked_class", fig_embarked),
    "deck": ("viz_deck_survival", fig_deck),
    "mother": ("viz_mother_children", fig_mother),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--only", choices=sorted(FIGURES), help="하나만 렌더링")
    args = parser.parse_args()
    mode = "dark" if args.dark else "light"
    c = TOKENS[mode]

    use_fonts()
    df = prepare()
    names = [args.only] if args.only else list(FIGURES)

    for name in names:
        stem, builder = FIGURES[name]
        fig, table = builder(df, c)
        out = OUT_DIR / f"{stem}_{mode}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)

        # 표 형태의 동일 정보 — 값이 색에만 의존하지 않도록
        print(f"\n=== {name} → {out.name} ===")
        for t in (table if isinstance(table, tuple) else (table,)):
            print(t.round(1).to_string())


if __name__ == "__main__":
    main()
