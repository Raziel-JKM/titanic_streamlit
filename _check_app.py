"""app.py의 파생변수·집계가 문서 수치와 맞는지 확인하는 일회성 점검 스크립트."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

import app

df = app.load_data()

raw = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
print("TITLE_MAP에 없는 호칭:", sorted(set(raw) - set(app.TITLE_MAP)))
print()
print("호칭 그룹 (문서: Mrs 79.4%/126, Miss 70.3%/185, Master 57.5%/40, Mr 15.7%/517, 직책 n=18, 귀족 n=5)")
print(app.rate_table(df, "Title").round(1).to_string())
print()
print("성별 × 등급 (문서: 여 96.8/92.1/50.0, 남 36.9/15.7/13.5)")
print(app.rate_table(df, ["Sex", "Pclass"]).round(1).to_string())
print()
print("가족 규모 (문서: 혼자 30.4%/537, 4명 72.4%/29)")
d = df.copy()
d["가족규모"] = d["FamilySize"].clip(upper=5)
print(app.rate_table(d, "가족규모").round(1).to_string())
print()
print("연령구간 × 성별 (문서: 남 0-12세 56.8%, 13-18세 8.8%)")
print(app.rate_table(df.dropna(subset=["Age"]), ["Sex", "AgeBand"]).round(1).to_string())
print()
print("1인당 요금 중앙값 (문서: 1등 £35.50 / 2등 £13.00 / 3등 £7.85)")
print(df[df["FarePerPerson"] > 0].groupby("Pclass")["FarePerPerson"].median().round(2).to_string())
print()
print("Cabin 기록 유무 × 등급 (문서: 1등 66 vs 48, 2등 81 vs 44, 3등 50 vs 24)")
print(app.rate_table(df, ["Pclass", "HasCabin"]).round(1).to_string())
print()
print("승선항 × 등급 (문서: 3등석 C 37.9 vs S 19.0)")
print(app.rate_table(df.dropna(subset=["Embarked"]), ["Pclass", "Embarked"]).round(1).to_string())
