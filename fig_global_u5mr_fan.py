# fig_global_u5mr_fan.py — robust, pivot + quantiles (population-weighted)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")

# 1) load forecasts and population cache (aligned to scenarios)
u5  = pd.read_csv(ROOT/"u5mr_predictions_scenarios_rcs.csv",
                  usecols=["ISO3","Year","Scenario","U5MR_median"])
pop = pd.read_csv(ROOT/"pop_predictions_scenarios.csv",
                  usecols=["ISO3","Year","Scenario","Population"])

# force numeric
for d, cols in [(u5, ["Year","U5MR_median"]), (pop, ["Year","Population"])]:
    for c in cols: d[c] = pd.to_numeric(d[c], errors="coerce")
u5 = u5.dropna(subset=["Year","U5MR_median"])
pop = pop.dropna(subset=["Year","Population"])

# 2) population-weighted world U5MR per scenario-year
u5w = (u5.merge(pop, on=["ISO3","Year","Scenario"], how="left")
         .dropna(subset=["Population"]))

# compute weighted average without groupby.apply
wx = (u5w.assign(wx = u5w["U5MR_median"] * u5w["Population"])
         .groupby(["Year","Scenario"], as_index=False)["wx"].sum())
W  = (u5w.groupby(["Year","Scenario"], as_index=False)["Population"].sum()
         .rename(columns={"Population":"W"}))
world = (wx.merge(W, on=["Year","Scenario"])
           .assign(U5MR_world=lambda d: d["wx"]/d["W"])
           .loc[:, ["Year","Scenario","U5MR_world"]]
           .sort_values(["Year","Scenario"]))

# 3) pivot to wide and compute quantiles across scenarios
wide = world.pivot(index="Year", columns="Scenario", values="U5MR_world").sort_index()
arr  = wide.to_numpy()

fan = pd.DataFrame({
    "Year": wide.index.astype(int),
    "Median":   np.nanmedian(arr, axis=1),
    "p50_lo":   np.nanquantile(arr, 0.25, axis=1),
    "p50_hi":   np.nanquantile(arr, 0.75, axis=1),
    "p80_lo":   np.nanquantile(arr, 0.10, axis=1),
    "p80_hi":   np.nanquantile(arr, 0.90, axis=1),
    "p95_lo":   np.nanquantile(arr, 0.025, axis=1),
    "p95_hi":   np.nanquantile(arr, 0.975, axis=1),
})

# 4) plot
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot(fan["Year"], fan["Median"], lw=2, label="Median")
ax.fill_between(fan["Year"], fan["p95_lo"], fan["p95_hi"], alpha=0.15, label="95%")
ax.fill_between(fan["Year"], fan["p80_lo"], fan["p80_hi"], alpha=0.25, label="80%")
ax.fill_between(fan["Year"], fan["p50_lo"], fan["p50_hi"], alpha=0.35, label="50%")
ax.set_xlim(int(fan["Year"].min()), int(fan["Year"].max()))
ax.set_xlabel("Year"); ax.set_ylabel("Global U5MR (per 1,000 live births)")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(ROOT/"fig_global_u5mr_fan.png", dpi=600)
print("✓ fig_global_u5mr_fan.png")
