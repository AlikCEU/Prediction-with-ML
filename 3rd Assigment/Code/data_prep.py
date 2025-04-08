# -------------- data_prep.py --------------
import pandas as pd
import numpy as np

def create_fast_growth_label(df):
    # Ensure data sorted by year
    df = df.sort_values(by=["comp_id", "year"])

    # Calculate percentage changes between 2012 and 2014
    df_2012 = df[df.year == 2012].set_index("comp_id")
    df_2014 = df[df.year == 2014].set_index("comp_id")

    # Merge data for 2012 and 2014
    growth_df = df_2012[["fage", "industry", "region"]].copy()
    growth_df["revenue_growth"] = (df_2014["sales"] - df_2012["sales"]) / df_2012["sales"]
    growth_df["employee_growth"] = (df_2014["empl"] - df_2012["empl"]) / df_2012["empl"]
    growth_df["personnel_cost_growth"] = (df_2014["pers_exp"] - df_2012["pers_exp"]) / df_2012["pers_exp"]
    growth_df["profit_growth"] = (df_2014["profit"] - df_2012["profit"]) / df_2012["profit"]
    growth_df["fixed_assets_growth"] = (df_2014["tang"] - df_2012["tang"]) / df_2012["tang"]

    # Conditions for fast growth
    cond_revenue = growth_df["revenue_growth"] >= 0.44

    cond_employees = (growth_df["employee_growth"] >= 0.21) & \
                     (growth_df["personnel_cost_growth"] <= 0.32)

    cond_profit = (growth_df["profit_growth"] >= 0.32) | ((df_2012["profit"] < 0) & (df_2014["profit"] > 0))

    cond_assets = growth_df["fixed_assets_growth"] >= 0.21

    # Final fast growth condition
    growth_df["fast_growth"] = cond_revenue & (cond_employees | cond_profit | cond_assets)
    growth_df["fast_growth"] = growth_df["fast_growth"].astype(int)

    return growth_df.reset_index()

def main():
    df = pd.read_csv("3rd Assignment/cs_bisnode_panel.csv")
    df = df[df["year"].between(2010, 2015)]

    label_df = create_fast_growth_label(df)

    # Merge back for modeling
    data = df[df.year == 2012].merge(label_df[["comp_id", "fast_growth"]], on="comp_id", how="inner")

    # Drop rows with missing values
    data = data.dropna()

    # Save cleaned dataset
    data.to_csv("3rd Assignment/bisnode_firms_fastgrowth.csv", index=False)

if __name__ == "__main__":
    main()
