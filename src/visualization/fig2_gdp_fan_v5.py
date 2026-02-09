# fig2_gdp_fan_v5.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
from pathlib import Path
import seaborn as sns

# Style settings (for publication)
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE

CSV_PATH = PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE 

# データの読み込み
df = pd.read_csv(CSV_PATH)

# シナリオによるフィルタリング（必要に応じて）
# ここでは主要なシナリオのみを使用（High/Lowなどを除く）
target_scenarios = [
    "Medium", "Constant-fertility", "Instant-replacement zero migration",
    "Momentum", "Zero-migration", "No change",
    "No fertility below age 18", "Accelerated adolescent-birth-rate decline"
]
# データセット内のシナリオ名と照合（部分一致対応）
available_scenarios = df["Scenario"].unique()
use_scenarios = []
for t in target_scenarios:
    for a in available_scenarios:
        if t in a:
            use_scenarios.append(a)
            break

df = df[df["Scenario"].isin(use_scenarios)]

# 世界計（年・シナリオごと）
world = df.groupby(["Year", "Scenario"], as_index=False)["Pred_Median"].sum()

# ピボットテーブル作成（行：年、列：シナリオ、値：GDP）
wide = world.pivot(index="Year", columns="Scenario", values="Pred_Median").sort_index()
years = wide.index.values
arr = wide.to_numpy() / 1e12  # Trillions (兆ドル) に変換

# クォンタイル（不確実性区間）の計算
# シナリオ間のばらつきを計算
med = np.median(arr, axis=1)
p95_lo = np.quantile(arr, 0.025, axis=1)
p95_hi = np.quantile(arr, 0.975, axis=1)
p80_lo = np.quantile(arr, 0.10, axis=1)
p80_hi = np.quantile(arr, 0.90, axis=1)
p50_lo = np.quantile(arr, 0.25, axis=1)
p50_hi = np.quantile(arr, 0.75, axis=1)

# プロット
fig, ax = plt.subplots(figsize=(8, 5))

# ファンチャートの描画
ax.fill_between(years, p95_lo, p95_hi, color="#1f77b4", alpha=0.15, label="95% Interval")
ax.fill_between(years, p80_lo, p80_hi, color="#1f77b4", alpha=0.25, label="80% Interval")
ax.fill_between(years, p50_lo, p50_hi, color="#1f77b4", alpha=0.35, label="50% Interval")
ax.plot(years, med, lw=3, color="#0b559f", label="Pooled Median")

# 装飾
ax.set_title("Global GDP Projection (Adjusted for Age Structure & Time Drift)", fontsize=14, pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Global GDP (USD Trillions, 2023 prices)", fontsize=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
ax.set_xlim(2025, 2100)
ax.legend(loc="upper left", frameon=False, fontsize=10)

# 余白調整と保存
plt.tight_layout()
SAVE_PATH = ROOT / "Figure2_Global_GDP_Fan_v5.png"
fig.savefig(SAVE_PATH, dpi=600)
print(f"✓ Saved Figure 2 to {SAVE_PATH}")
plt.show()