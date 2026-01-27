import pandas as pd
from pathlib import Path

ROOT     = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
SCEN_CSV = ROOT / "gdp_predictions_scenarios.csv"
OUT_CSV  = ROOT / "table_top10_economies.csv"

df = pd.read_csv(SCEN_CSV,
                 usecols=["Scenario", "Year", "Country",
                          "Pred_Median", "Pred_Lower", "Pred_Upper"])

# 既に High / Low を除外している prediction_fixed.py を走らせた前提
# → そのまま全シナリオ中央値を国別に再集計
agg = (df.groupby(["Country", "Year"], as_index=False)
         .median(numeric_only=True))

def top_n(year, n=10):
    sub = agg[agg["Year"] == year].nlargest(n, "Pred_Median").copy()
    sub["Rank"] = range(1, n+1)
    return (sub[["Rank", "Country", "Pred_Median",
                 "Pred_Lower", "Pred_Upper"]]
            .assign(Year=year))

tbl = pd.concat([top_n(2035), top_n(2040), top_n(2050), top_n(2100)], axis=0)

# 単位を兆 USD に丸める
for col in ["Pred_Median", "Pred_Lower", "Pred_Upper"]:
    tbl[col] = (tbl[col] / 1e12).round(1)

tbl.to_csv(OUT_CSV, index=False)
print(f"✓ top-10 table → {OUT_CSV}")
