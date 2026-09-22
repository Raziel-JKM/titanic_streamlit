"""타이타닉 생존 분석 대시보드 (Streamlit).

실행:
    streamlit run app.py

데이터: titanic.csv (Kaggle 학습용 891명 표본)
가설·차트 해설: "titanic 분석 가설.md", "titanic 데이터 설명.md"
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

BASE = Path(__file__).parent
CSV = BASE / "titanic.csv"
DOC_HYPOTHESIS = BASE / "titanic 분석 가설.md"
DOC_SCHEMA = BASE / "titanic 데이터 설명.md"

# ─────────────────────────────────────────────────────────────────────────────
# 색상 토큰 — 검증된 기본 팔레트 (survival_pie_by_sex_pclass.py와 동일한 값)
#
# 색 배정 규칙
#   · 성별은 앱 전역 고정: 여성 = 슬롯1(blue), 남성 = 슬롯2(orange)
#   · 객실등급은 순서형이므로 단일 hue 램프(1등석이 서피스에서 가장 먼 단계)
#   · 그 밖의 2~3계열은 해당 차트 안에서만 슬롯 1·2·3을 쓰고 항상 범례를 둔다
# ─────────────────────────────────────────────────────────────────────────────
TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
        "series_3": "#1baf7a",
        # 순서형 blue 램프 (라이트에서는 250단계보다 밝게 내려가지 않는다)
        "class_1": "#184f95",
        "class_2": "#2a78d6",
        "class_3": "#86b6ef",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series_1": "#3987e5",
        "series_2": "#d95926",
        "series_3": "#199e70",
        # 다크에서는 600단계보다 어둡게 내려가지 않는다
        "class_1": "#b7d3f6",
        "class_2": "#3987e5",
        "class_3": "#184f95",
    },
}

CLASS_LABELS = {1: "1등석", 2: "2등석", 3: "3등석"}
SEX_LABELS = {"female": "여성", "male": "남성"}
SEX_SLOT = {"female": "series_1", "male": "series_2"}
EMBARKED_LABELS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}

AGE_BINS = [0, 12, 18, 40, 60, 200]
AGE_LABELS = ["0–12세", "13–18세", "19–40세", "41–60세", "61세+"]

# 호칭 묶음 — "titanic 분석 가설.md" H12의 기준과 동일 (직책 18명 / 귀족 5명)
TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Mrs": "Mrs", "Mme": "Mrs",
    "Master": "Master",
    "Dr": "직책", "Rev": "직책", "Major": "직책", "Col": "직책", "Capt": "직책",
    "Don": "귀족", "Sir": "귀족", "Lady": "귀족", "the Countess": "귀족",
    "Jonkheer": "귀족",
}

# 셀이 이보다 작으면 비율을 읽지 말라는 표시를 붙인다 (문서의 공통 규칙)
SMALL_N = 20


# ─────────────────────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV)

    df["성별"] = df["Sex"].map(SEX_LABELS)
    df["등급"] = df["Pclass"].map(CLASS_LABELS)
    df["생존"] = df["Survived"].map({0: "사망", 1: "생존"})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = df["FamilySize"] == 1

    # Fare는 1인당이 아니라 티켓 단위 합계다 — 일행 수로 나눠야 비교가 된다
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("size")
    df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

    df["HasCabin"] = df["Cabin"].notna()
    df["Deck"] = df["Cabin"].str[0]

    raw_title = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    df["Title"] = raw_title.map(TITLE_MAP).fillna("기타")

    df["AgeBand"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    return df


def rate_table(df: pd.DataFrame, by) -> pd.DataFrame:
    """그룹별 생존 수 / 인원 / 생존율. 비율은 항상 n과 함께 나간다."""
    g = df.groupby(by, observed=True)["Survived"]
    out = pd.DataFrame({"생존": g.sum().astype(int), "인원": g.size().astype(int)})
    out["사망"] = out["인원"] - out["생존"]
    out["생존율"] = out["생존"] / out["인원"] * 100
    return out[["생존", "사망", "인원", "생존율"]]


def as_display_table(t: pd.DataFrame) -> pd.DataFrame:
    """표로 보기용 — 생존율을 소수 한 자리 문자열로, 작은 셀에 ⚠를 붙인다."""
    d = t.copy()
    d["생존율"] = d["생존율"].map(lambda v: f"{v:.1f}%")
    d["인원"] = [f"{n:,}명 ⚠" if n < SMALL_N else f"{n:,}명" for n in t["인원"]]
    d["생존"] = t["생존"].map(lambda v: f"{v:,}명")
    d["사망"] = t["사망"].map(lambda v: f"{v:,}명")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 차트 공통
# ─────────────────────────────────────────────────────────────────────────────
def new_fig(c, figsize, ncols=1, **kw):
    plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=140, **kw)
    fig.patch.set_facecolor(c["surface"])
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(c["surface"])
    return fig, axes


def style_axes(ax, c, ymax=100, ylabel=None):
    """격자·축은 뒤로 물리고 데이터만 남긴다."""
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=c["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(colors=c["muted"], length=0, labelsize=10)
    for lbl in ax.get_xticklabels():
        lbl.set_color(c["text_secondary"])
    if ylabel:
        ax.set_ylabel(ylabel, color=c["muted"], fontsize=10)


def titled(fig, c, title, subtitle=None, y=0.945):
    fig.text(0.5, y, title, ha="center", fontsize=14, fontweight="bold",
             color=c["text_primary"], va="top")
    if subtitle:
        fig.text(0.5, y - 0.075, subtitle, ha="center", fontsize=10,
                 color=c["text_secondary"], va="top")


def legend(fig, c, items, y=0.02, ncol=None):
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label)
                 for label, color in items],
        loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(items),
        frameon=False, handlelength=0.9, handleheight=0.9, columnspacing=1.8,
        fontsize=10, labelcolor=c["text_secondary"],
    )


def show(fig, table: pd.DataFrame, key: str):
    """차트 + 같은 정보를 담은 표 — 값이 색에만 의존하지 않게."""
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("표로 보기 (⚠ = n < 20, 비율을 읽지 말 것)"):
        st.dataframe(as_display_table(table), width="stretch")


def bar_label(ax, x, height, c, dy=2.0):
    ax.text(x, height + dy, f"{height:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=c["text_primary"])


def tick_with_n(label, n):
    """표본 크기는 눈금 라벨 둘째 줄로 — 막대 아래에 따로 쓰면 눈금과 겹친다."""
    return f"{label}\n{n:,}명" + (" ⚠" if n < SMALL_N else "")


# ─────────────────────────────────────────────────────────────────────────────
# 차트
# ─────────────────────────────────────────────────────────────────────────────
def chart_sex_class(df, c):
    """성별로 패널을 나누고, 패널 안은 등급 순서형 램프."""
    t = rate_table(df, ["Sex", "Pclass"])
    sexes = [s for s in ("female", "male") if s in df["Sex"].values]

    fig, axes = new_fig(c, (4.6 * len(sexes), 4.4), ncols=len(sexes), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, sex in zip(axes, sexes):
        classes = sorted(df.loc[df["Sex"] == sex, "Pclass"].unique())
        ticks = []
        for i, pclass in enumerate(classes):
            if (sex, pclass) not in t.index:
                ticks.append(CLASS_LABELS[pclass])
                continue
            row = t.loc[(sex, pclass)]
            # 막대 사이는 2px 서피스 간격 — 테두리를 두르지 않는다
            ax.bar(i, row["생존율"], width=0.62, color=c[f"class_{pclass}"], zorder=2)
            bar_label(ax, i, row["생존율"], c)
            ticks.append(tick_with_n(CLASS_LABELS[pclass], int(row["인원"])))
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.6, len(classes) - 0.4)
        style_axes(ax, c, ymax=112)
        ax.set_title(SEX_LABELS[sex], fontsize=12, fontweight="bold",
                     color=c["text_primary"], pad=10)

    axes[0].set_ylabel("생존율 (%)", color=c["muted"], fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.72, bottom=0.15, wspace=0.12)
    titled(fig, c, "성별 · 객실등급별 생존율",
           "성별 효과가 등급 효과보다 크다 — 3등석 여성이 1등석 남성보다 높다")
    return fig, t


def chart_age(df, c):
    """연령구간 × 성별 꺾은선. n<20은 속 빈 마커."""
    t = rate_table(df.dropna(subset=["Age"]), ["Sex", "AgeBand"])
    x = np.arange(len(AGE_LABELS))

    fig, ax = new_fig(c, (8.6, 4.6))
    for sex in ("female", "male"):
        if sex not in df["Sex"].values:
            continue
        color = c[SEX_SLOT[sex]]
        ys, ns = [], []
        for band in AGE_LABELS:
            if (sex, band) in t.index:
                ys.append(t.loc[(sex, band), "생존율"])
                ns.append(int(t.loc[(sex, band), "인원"]))
            else:
                ys.append(np.nan)
                ns.append(0)
        ax.plot(x, ys, color=color, linewidth=2, zorder=3,
                label=SEX_LABELS[sex], solid_capstyle="round")
        for xi, (y, n) in enumerate(zip(ys, ns)):
            if np.isnan(y):
                continue
            solid = n >= SMALL_N
            ax.plot(xi, y, marker="o", markersize=9, zorder=4,
                    color=color if solid else c["surface"],
                    markeredgecolor=color, markeredgewidth=2)
            if solid:  # 작은 칸은 값을 직접 쓰지 않는다 — 표에서 보게 한다
                # 0–12세에서 두 선이 붙으므로 라벨은 계열마다 반대쪽으로 뗀다
                above = sex == "female"
                ax.text(xi, y + (4 if above else -4.5), f"{y:.0f}%", ha="center",
                        va="bottom" if above else "top", fontsize=9,
                        fontweight="bold", color=c["text_primary"])

    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.set_xlim(-0.4, len(AGE_LABELS) - 0.6)
    style_axes(ax, c, ymax=110, ylabel="생존율 (%)")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.20)
    titled(fig, c, "연령구간별 생존율 — 성별",
           "속 빈 마커는 n<20 · 남성만 0–12세에서 뚜렷하게 솟는다")
    legend(fig, c, [(SEX_LABELS[s], c[SEX_SLOT[s]]) for s in ("female", "male")
                    if s in df["Sex"].values], y=0.015)
    return fig, t


def chart_family(df, c):
    """가족 규모별 생존율 — 5명 이상은 한 칸으로 묶는다."""
    d = df.copy()
    d["가족규모"] = d["FamilySize"].clip(upper=5)
    labels = {1: "혼자", 2: "2명", 3: "3명", 4: "4명", 5: "5명+"}
    t = rate_table(d, "가족규모")
    t.index = [labels[i] for i in t.index]

    fig, ax = new_fig(c, (8.6, 4.4))
    x = np.arange(len(t))
    ax.plot(x, t["생존율"], color=c["series_1"], linewidth=2, zorder=3,
            solid_capstyle="round")
    for xi, (y, n) in enumerate(zip(t["생존율"], t["인원"])):
        solid = n >= SMALL_N
        ax.plot(xi, y, marker="o", markersize=9, zorder=4,
                color=c["series_1"] if solid else c["surface"],
                markeredgecolor=c["series_1"], markeredgewidth=2)
        ax.text(xi, y + 4, f"{y:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=c["text_primary"])

    overall = df["Survived"].mean() * 100
    ax.axhline(overall, color=c["axis"], linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.text(len(t) - 0.62, overall + 2, f"현재 선택 평균 {overall:.1f}%",
            ha="right", va="bottom", fontsize=9, color=c["muted"])

    ax.set_xticks(x)
    ax.set_xticklabels([tick_with_n(label, n) for label, n in zip(t.index, t["인원"])])
    ax.set_xlim(-0.4, len(t) - 0.6)
    style_axes(ax, c, ymax=95, ylabel="생존율 (%)")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.18)
    titled(fig, c, "가족 규모별 생존율",
           "4명을 정점으로 한 역U자 — 5명부터 급락한다 (성별·등급 통제 전)")
    return fig, t


def chart_title(df, c):
    """호칭 그룹별 가로 막대. 단일 계열이므로 범례는 두지 않는다."""
    t = rate_table(df, "Title").sort_values("생존율")
    fig, ax = new_fig(c, (8.6, 0.52 * len(t) + 2.3))
    y = np.arange(len(t))
    ax.barh(y, t["생존율"], height=0.62, color=c["series_1"], zorder=2)
    for yi, (rate, n) in enumerate(zip(t["생존율"], t["인원"])):
        mark = " ⚠" if n < SMALL_N else ""
        ax.text(rate + 1.5, yi, f"{rate:.1f}%   n={n:,}{mark}", va="center",
                ha="left", fontsize=10, color=c["text_secondary"])

    ax.set_yticks(y)
    ax.set_yticklabels(t.index, color=c["text_primary"], fontsize=11)
    ax.set_xlim(0, 112)
    ax.xaxis.grid(True, color=c["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(colors=c["muted"], length=0, labelsize=10)
    ax.set_xlabel("생존율 (%)", color=c["muted"], fontsize=10)

    # 막대 수에 따라 그림 높이가 달라지므로 제목·축 여백도 인치 단위로 맞춘다
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.13, right=0.98, top=1 - 1.15 / h, bottom=0.75 / h)
    titled(fig, c, "호칭 그룹별 생존율",
           "같은 남성인데 Mr와 Master가 갈린다 — Sex 하나로는 안 잡히는 정보",
           y=1 - 0.28 / h)
    return fig, t


def grouped_bars(df, c, series_col, series_order, series_labels, title, subtitle):
    """x = 객실등급, 계열 = series_col. 2~3계열용 공통 그룹 막대."""
    t = rate_table(df, ["Pclass", series_col])
    classes = sorted(df["Pclass"].unique())
    present = [s for s in series_order if s in df[series_col].values]
    slots = [c[f"series_{i + 1}"] for i in range(len(present))]

    fig, ax = new_fig(c, (9.0, 4.8))
    width = 0.78 / max(len(present), 1)
    for si, (key, color) in enumerate(zip(present, slots)):
        for ci, pclass in enumerate(classes):
            if (pclass, key) not in t.index:
                continue
            row = t.loc[(pclass, key)]
            # 막대 폭을 계열 폭보다 살짝 줄여 인접 막대 사이에 서피스 간격을 둔다
            x = ci + (si - (len(present) - 1) / 2) * width
            ax.bar(x, row["생존율"], width=width * 0.88, color=color, zorder=2)
            n = int(row["인원"])
            # 계열이 여러 개라 n을 눈금에 넣을 수 없다 — 막대 위에 두 줄로 붙인다
            ax.text(x, row["생존율"] + 2.5,
                    f"{row['생존율']:.0f}%\n{n:,}명" + (" ⚠" if n < SMALL_N else ""),
                    ha="center", va="bottom", fontsize=8.5, linespacing=1.4,
                    color=c["text_primary"])

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([CLASS_LABELS[p] for p in classes])
    ax.set_xlim(-0.6, len(classes) - 0.4)
    style_axes(ax, c, ymax=118, ylabel="생존율 (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.17)
    titled(fig, c, title, subtitle)
    legend(fig, c, [(series_labels[k], col) for k, col in zip(present, slots)], y=0.015)
    return fig, t


def chart_cabin(df, c):
    return grouped_bars(
        df, c, "HasCabin", [True, False],
        {True: "Cabin 기록 있음", False: "기록 없음"},
        "Cabin 기록 유무별 생존율 (등급 통제)",
        "등급을 맞춰도 기록이 있는 쪽이 모든 등급에서 높다 — 결측 자체가 정보다",
    )


def chart_embarked(df, c):
    return grouped_bars(
        df, c, "Embarked", ["C", "Q", "S"], EMBARKED_LABELS,
        "승선항 × 객실등급별 생존율",
        "등급을 맞춰도 3등석에서 C와 S의 격차가 남는다 — 등급만으로는 설명 안 된다",
    )


def chart_fare(df, c):
    """등급 안에서 1인당 요금 4분위 — 요금이 등급의 대리 변수인지 본다."""
    d = df[df["FarePerPerson"] > 0].copy()
    classes = sorted(d["Pclass"].unique())
    if not classes:
        return None, pd.DataFrame()

    # 같은 요금값이 뭉쳐 4분위가 안 나오는 등급이 있어 labels=False로 받고 직접 이름 붙인다
    rank = d.groupby("Pclass")["FarePerPerson"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    d["분위"] = rank.map(lambda i: f"Q{int(i) + 1}" if pd.notna(i) else None)
    d = d.dropna(subset=["분위"])
    t = rate_table(d, ["Pclass", "분위"])

    fig, axes = new_fig(c, (3.3 * len(classes), 4.8), ncols=len(classes), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pclass in zip(axes, classes):
        quarts = [q for q in ["Q1", "Q2", "Q3", "Q4"] if (pclass, q) in t.index]
        ticks = []
        for i, q in enumerate(quarts):
            row = t.loc[(pclass, q)]
            ax.bar(i, row["생존율"], width=0.62, color=c[f"class_{pclass}"], zorder=2)
            bar_label(ax, i, row["생존율"], c)
            ticks.append(tick_with_n(q, int(row["인원"])))
        ax.set_xticks(range(len(quarts)))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_xlim(-0.6, len(quarts) - 0.4)
        style_axes(ax, c, ymax=112)
        median = d.loc[d["Pclass"] == pclass, "FarePerPerson"].median()
        ax.set_title(f"{CLASS_LABELS[pclass]}\n1인당 중앙값 £{median:,.2f}",
                     fontsize=10.5, color=c["text_primary"], pad=16, linespacing=1.5)

    axes[0].set_ylabel("생존율 (%)", color=c["muted"], fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.70, bottom=0.16, wspace=0.14)
    titled(fig, c, "등급 내 1인당 요금 4분위별 생존율",
           "Fare는 티켓 단위 합계라 일행 수로 나눈 뒤 비교했다 (£0 15건 제외)")
    return fig, t


# ─────────────────────────────────────────────────────────────────────────────
# 문서 렌더링
# ─────────────────────────────────────────────────────────────────────────────
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@st.cache_data
def read_doc(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)   # YAML frontmatter
    return re.sub(r"\[\[([^\]]+)\]\]", r"**\1**", text)          # 옵시디언 위키링크


def render_doc(path: Path, mode: str):
    """마크다운을 그대로 렌더링하되 ![](...png)만 st.image로 바꾼다.

    테마에 맞춰 _light.png ↔ _dark.png를 갈아끼우므로, 문서를 고치면
    대시보드도 같이 바뀐다 (해설을 여기에 복제하지 않는 이유).
    """
    text = read_doc(str(path))
    cursor = 0
    for m in IMG_RE.finditer(text):
        chunk = text[cursor:m.start()].strip()
        if chunk:
            st.markdown(chunk)
        src = BASE / m.group(2)
        if mode == "dark":
            dark = src.with_name(src.name.replace("_light.png", "_dark.png"))
            if dark.exists():
                src = dark
        if src.exists():
            st.image(str(src), caption=m.group(1), width="stretch")
        else:
            st.warning(f"이미지 없음: {m.group(2)}")
        cursor = m.end()
    tail = text[cursor:].strip()
    if tail:
        st.markdown(tail)


# ─────────────────────────────────────────────────────────────────────────────
# 앱
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="타이타닉 생존 분석", page_icon="🚢", layout="wide")

df_all = load_data()

with st.sidebar:
    st.header("보기 설정")
    mode = "dark" if st.radio("테마", ["라이트", "다크"], horizontal=True) == "다크" else "light"
    c = TOKENS[mode]

    st.divider()
    st.subheader("필터")
    st.caption("아래 차트는 모두 선택한 범위로 다시 계산됩니다.")

    sexes = st.multiselect("성별", ["여성", "남성"], default=["여성", "남성"])
    classes = st.multiselect("객실등급", list(CLASS_LABELS.values()),
                             default=list(CLASS_LABELS.values()))
    ports = st.multiselect("승선항", list(EMBARKED_LABELS.values()),
                           default=list(EMBARKED_LABELS.values()))
    age_range = st.slider("나이", 0, 80, (0, 80))
    keep_age_na = st.checkbox("나이 결측 177명 포함", value=True,
                              help="Age 결측은 19.9%입니다. 제외하면 표본이 714명이 됩니다.")

sex_keys = [k for k, v in SEX_LABELS.items() if v in sexes]
class_keys = [k for k, v in CLASS_LABELS.items() if v in classes]
port_keys = [k for k, v in EMBARKED_LABELS.items() if v in ports]

mask = (
    df_all["Sex"].isin(sex_keys)
    & df_all["Pclass"].isin(class_keys)
    & df_all["Embarked"].isin(port_keys)
)
in_range = df_all["Age"].between(*age_range)
mask &= in_range | (df_all["Age"].isna() if keep_age_na else False)
df = df_all[mask]

# 라이트/다크 선택이 차트 서피스와 페이지 배경 양쪽에 걸리게 한다
st.markdown(
    f"""<style>
      .stApp {{ background-color: {c['plane']}; }}
      .stApp, .stApp p, .stApp li, .stApp label,
      .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {c['text_primary']}; }}
      [data-testid="stSidebar"] {{ background-color: {c['surface']}; }}
      [data-testid="stMetricLabel"] {{ color: {c['text_secondary']}; }}
    </style>""",
    unsafe_allow_html=True,
)

st.title("🚢 타이타닉 생존 분석")
st.caption("출처: titanic.csv — Kaggle 학습용 891명 표본 · 연관은 인과가 아니며, "
           "결과는 모두 “이 표본에서”로 한정해 읽어야 합니다.")

if df.empty:
    st.warning("선택한 조건에 해당하는 승객이 없습니다. 사이드바에서 필터를 넓혀 주세요.")
    st.stop()

# 헤드라인은 차트가 아니라 숫자 타일로 — 값 하나가 답인 자리
total, survived = len(df), int(df["Survived"].sum())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("선택 인원", f"{total:,}명", f"전체 891명 중 {total / 891 * 100:.0f}%")
k2.metric("생존", f"{survived:,}명")
k3.metric("생존율", f"{survived / total * 100:.1f}%",
          f"{survived / total * 100 - 38.38:+.1f}%p vs 전체")
for col, sex in ((k4, "female"), (k5, "male")):
    sub = df[df["Sex"] == sex]
    col.metric(f"{SEX_LABELS[sex]} 생존율",
               f"{sub['Survived'].mean() * 100:.1f}%" if len(sub) else "—",
               f"{len(sub):,}명" if len(sub) else "표본 없음", delta_color="off")

tab_overview, tab_age, tab_status, tab_doc, tab_data = st.tabs(
    ["개요", "연령 · 가족", "신분 · 위치", "가설 리포트", "원자료"]
)

with tab_overview:
    st.subheader("성별과 등급, 무엇이 더 셌나")
    show(*chart_sex_class(df, c), key="sex_class")
    st.markdown(
        "**읽는 법** — 전체 표본에서 3등석 여성 50.0%가 1등석 남성 36.9%보다 높습니다. "
        "성별 효과가 등급 효과보다 크다는 뜻이고, 그래서 이후 모든 가설은 "
        "*성별과 등급을 통제한 뒤에도 효과가 남는가*를 기본 질문으로 삼습니다."
    )

with tab_age:
    st.subheader("연령")
    show(*chart_age(df, c), key="age")
    st.markdown(
        "**읽는 법** — 어린이 효과가 **남성에서만** 나타납니다. 남성은 0–12세 56.8%에서 "
        "13–18세 8.8%로 급락하고 이후 평평합니다. 여성은 전 연령 75% 내외로 평평하고 "
        "오히려 여아가 성인 여성보다 낮습니다. “여성과 어린이 먼저”의 *어린이* 조항은 "
        "사실상 남아에게만 작동한 모양새입니다."
    )
    st.divider()
    st.subheader("가족 규모")
    show(*chart_family(df, c), key="family")
    st.markdown(
        "**읽는 법** — 혼자 탄 537명은 30.4%로 평균보다 낮고, 모양은 4명을 정점으로 한 "
        "역U자입니다. 다만 혼자 탄 쪽에 남성이 몰려 있어 이 차이의 상당 부분은 성별로 "
        "설명될 수 있습니다. 사이드바에서 성별을 하나만 남겨 보면 직접 확인됩니다."
    )

with tab_status:
    st.subheader("이름에 담긴 신분 (H12)")
    show(*chart_title(df, c), key="title")
    st.markdown(
        "**읽는 법** — 핵심은 Mr 15.7% vs Master 57.5%입니다. 같은 남성인데 41.8%p 차이라, "
        "`Sex` 하나로는 잡히지 않는 정보(연령·신분)가 호칭에 들어 있습니다. "
        "귀족 n=5, 직책 n=18은 비율을 읽지 마세요."
    )
    st.divider()
    st.subheader("Cabin 결측 자체가 정보인가 (H10)")
    show(*chart_cabin(df, c), key="cabin")
    st.markdown(
        "**읽는 법** — 등급을 통제해도 기록이 있는 쪽이 모든 등급에서 높습니다. "
        "`HasCabin`은 등급의 그림자만은 아니라는 뜻이지만, 기록 여부가 무엇의 대리 변수인지는 "
        "아직 모릅니다 — 누출(leakage) 점검이 남아 있습니다."
    )
    st.divider()
    st.subheader("승선항 (H11)")
    show(*chart_embarked(df, c), key="embarked")
    st.markdown(
        "**읽는 법** — 등급을 맞추면 전체 격차는 크게 줄지만 3등석에서 C 37.9% vs S 19.0%로 "
        "19%p가 남습니다. 등급만으로는 설명되지 않아 **판정 보류**입니다. "
        "Q는 사실상 3등석 표본이라 등급별 비교를 하지 마세요."
    )
    st.divider()
    st.subheader("요금은 등급의 대리 변수인가 (H8)")
    fig_fare, t_fare = chart_fare(df, c)
    if fig_fare is None:
        st.info("선택한 범위에 유효한 요금 자료가 없습니다.")
    else:
        show(fig_fare, t_fare, key="fare")
        st.markdown(
            "**읽는 법** — 등급 안에서 4분위로 나누면 막대가 거의 평평하고 단조 증가 패턴이 "
            "없습니다. **H8 지지 방향** — 요금은 대체로 등급의 대리 변수로 보입니다."
        )

with tab_doc:
    doc = st.radio("문서", ["분석 가설 13종", "데이터 설명"], horizontal=True,
                   label_visibility="collapsed")
    st.info("이 탭은 사이드바 필터와 무관하게 **전체 891명 기준**으로 미리 만들어 둔 "
            "리포트입니다. 원본 마크다운과 PNG를 그대로 읽어오므로 문서를 고치면 여기도 바뀝니다.")
    render_doc(DOC_HYPOTHESIS if doc == "분석 가설 13종" else DOC_SCHEMA, mode)

with tab_data:
    st.subheader(f"선택된 {len(df):,}명")
    cols = ["PassengerId", "생존", "등급", "Name", "성별", "Age", "SibSp", "Parch",
            "FamilySize", "Title", "Ticket", "TicketGroupSize", "Fare",
            "FarePerPerson", "Cabin", "Deck", "Embarked"]
    st.dataframe(df[cols], width="stretch", hide_index=True, height=520)
    st.download_button("CSV로 내려받기", df[cols].to_csv(index=False, encoding="utf-8-sig"),
                       file_name="titanic_filtered.csv", mime="text/csv")
