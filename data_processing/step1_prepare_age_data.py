# step1_fix_age_data.py
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATH_MERGED_AGE, PATH_POP_PREDICTIONS, PATH_AGE_PREDICTIONS

HIST_CSV = PATH_MERGED_AGE
POP_CSV = PATH_POP_PREDICTIONS
OUT_AGE_CSV = PATH_AGE_PREDICTIONS

def main():
    print("--- Step 1: generating consistent age data ---")
    # 1. Load 2023 Age Structure from History
    df_hist = pd.read_csv(HIST_CSV)
    # Get 2023 values for each country
    df_2023 = df_hist[df_hist['Year'] == 2023][['ISO3', 'WAshare', 'OldDep']].copy()
    
    if df_2023.empty:
        # Fallback to latest available year if 2023 is missing
        print("2023 data missing, using latest available year per country...")
        df_2023 = df_hist.sort_values('Year').groupby('ISO3').tail(1)[['ISO3', 'WAshare', 'OldDep']]

    print(f"Loaded Age Structure for {len(df_2023)} countries.")

    # 2. Load Future Population Scenarios (to get ISO3-Year-Scenario keys)
    df_pop = pd.read_csv(POP_CSV)
    print(f"Loaded Population Scenarios: {df_pop['Scenario'].unique()}")

    # 3. Merge to broadcast 2023 structure to all future years/scenarios
    # Left join ensures we have an Age row for every Population row
    df_age_future = pd.merge(df_pop[['ISO3', 'Year', 'Scenario']], df_2023, on='ISO3', how='left')
    
    # 4. Save
    df_age_future = df_age_future.dropna(subset=['WAshare', 'OldDep'])
    df_age_future.to_csv(OUT_AGE_CSV, index=False)
    print(f"✓ Saved {OUT_AGE_CSV} ({len(df_age_future)} rows)")

if __name__ == "__main__":
    main()