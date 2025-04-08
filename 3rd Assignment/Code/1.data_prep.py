# -------------- data_prep.py --------------
import pandas as pd
import numpy as np

def create_fast_growth_label(df):
    # Ensure data sorted by year
    df = df.sort_values(by=["comp_id", "year"])

    # Filter out firms with missing 2012 or 2014 sales
    df_2012 = df[df.year == 2012].dropna(subset=["sales"]).set_index("comp_id")
    df_2014 = df[df.year == 2014].dropna(subset=["sales"]).set_index("comp_id")

    common_ids = df_2012.index.intersection(df_2014.index)
    df_2012 = df_2012.loc[common_ids]
    df_2014 = df_2014.loc[common_ids]

    # Calculate fage and extract relevant columns
    growth_df = pd.DataFrame(index=df_2012.index)
    growth_df["fage"] = 2012 - df_2012["founded_year"]
    growth_df["industry"] = df_2012["ind"]
    growth_df["region"] = df_2012["region_m"]

    # Growth calculations with fillna to avoid NaN propagation
    growth_df["revenue_growth"] = (df_2014["sales"] - df_2012["sales"]) / df_2012["sales"]
    growth_df["employee_growth"] = (df_2014["labor_avg"] - df_2012["labor_avg"]) / df_2012["labor_avg"]
    growth_df["personnel_cost_growth"] = (df_2014["personnel_exp"] - df_2012["personnel_exp"]) / df_2012["personnel_exp"]
    growth_df["profit_growth"] = (df_2014["profit_loss_year"] - df_2012["profit_loss_year"]) / df_2012["profit_loss_year"]
    growth_df["fixed_assets_growth"] = (df_2014["tang_assets"] - df_2012["tang_assets"]) / df_2012["tang_assets"]

    # Replace infinite or missing values
    growth_df = growth_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Conditions for fast growth
    cond_revenue = growth_df["revenue_growth"] >= 0.44

    cond_employees = (growth_df["employee_growth"] >= 0.21) & \
                     (growth_df["personnel_cost_growth"] <= 0.32)

    cond_profit = (growth_df["profit_growth"] >= 0.32) | \
                  ((df_2012.loc[growth_df.index, "profit_loss_year"] < 0) &
                   (df_2014.loc[growth_df.index, "profit_loss_year"] > 0))

    cond_assets = growth_df["fixed_assets_growth"] >= 0.21

    # Final fast growth condition
    growth_df["fast_growth"] = cond_revenue & (cond_employees | cond_profit | cond_assets)
    growth_df["fast_growth"] = growth_df["fast_growth"].astype(int)

    print("Total firms after filtering:", len(growth_df))
    print("Number of fast-growing firms:", growth_df["fast_growth"].sum())

    return growth_df.reset_index()

def main():
    df = pd.read_csv("3rd Assignment/cs_bisnode_panel.csv")
    df = df[df["year"].between(2010, 2015)]

    label_df = create_fast_growth_label(df)

    # Merge back for modeling
    data = df[df.year == 2012].merge(label_df[["comp_id", "fast_growth"]], on="comp_id", how="inner")

    model_features = [
        'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
        'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
        'curr_liab', 'share_eq', 'subscribed_cap', 'founded_year', 'foreign'
    ]

    data = data.dropna(subset=model_features)
    data['age'] = 2012 - data['founded_year']

    print("Final modeling dataset shape:", data.shape)
    print("Fast growth label distribution:\n", data["fast_growth"].value_counts())

    # Save cleaned dataset
    data.to_csv("3rd Assignment/bisnode_firms_fastgrowth.csv", index=False)

if __name__ == "__main__":
    main()