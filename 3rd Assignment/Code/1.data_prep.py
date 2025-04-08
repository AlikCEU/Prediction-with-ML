# -------------- data_prep.py --------------
# This script prepares the data for fast growth prediction
# It creates a binary label for fast-growing firms based on multiple growth metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_fast_growth_label(df):
    """
    Creates a binary label for fast-growing firms between 2012 and 2014.
    
    Fast growth is defined as meeting revenue growth criteria AND at least one of:
    - Employee growth with controlled personnel costs
    - Profit growth or turnaround from negative to positive profit
    - Fixed assets growth
    
    Growth thresholds were selected to identify top-performing firms (approximately top 20%)
    while ensuring multiple dimensions of growth are considered.
    """
    # Ensure data sorted by year
    df = df.sort_values(by=["comp_id", "year"])

    # Filter out firms with missing 2012 or 2014 sales
    df_2012 = df[df.year == 2012].dropna(subset=["sales"]).set_index("comp_id")
    df_2014 = df[df.year == 2014].dropna(subset=["sales"]).set_index("comp_id")

    common_ids = df_2012.index.intersection(df_2014.index)
    df_2012 = df_2012.loc[common_ids]
    df_2014 = df_2014.loc[common_ids]

    # Calculate firm age and extract relevant columns
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

    # Calculate growth percentiles to inform threshold selection
    for col in [c for c in growth_df.columns if c.endswith('_growth')]:
        p25, p50, p75, p90 = np.percentile(growth_df[col], [25, 50, 75, 90])
        print(f"{col} percentiles: 25%={p25:.2f}, 50%={p50:.2f}, 75%={p75:.2f}, 90%={p90:.2f}")

    # Conditions for fast growth - thresholds target approximately top 20-25% of firms
    # Revenue growth is primary - must have at least 44% growth over 2 years (~20% per year)
    cond_revenue = growth_df["revenue_growth"] >= 0.44

    # Employee growth must be substantial (>21% over 2 years) while keeping personnel costs under control
    cond_employees = (growth_df["employee_growth"] >= 0.21) & \
                     (growth_df["personnel_cost_growth"] <= 0.32)

    # Profit growth of 32% or more, OR a turnaround from negative to positive profit
    cond_profit = (growth_df["profit_growth"] >= 0.32) | \
                  ((df_2012.loc[growth_df.index, "profit_loss_year"] < 0) &
                   (df_2014.loc[growth_df.index, "profit_loss_year"] > 0))

    # Fixed assets growth of at least 21% indicating investment in production capacity
    cond_assets = growth_df["fixed_assets_growth"] >= 0.21

    # Final fast growth condition: Revenue growth AND at least one secondary growth indicator
    growth_df["fast_growth"] = cond_revenue & (cond_employees | cond_profit | cond_assets)
    growth_df["fast_growth"] = growth_df["fast_growth"].astype(int)

    print("Total firms after filtering:", len(growth_df))
    print("Number of fast-growing firms:", growth_df["fast_growth"].sum())
    print(f"Percentage of fast-growing firms: {growth_df['fast_growth'].mean()*100:.2f}%")

    return growth_df.reset_index()

def perform_eda(df, label_df):
    """
    Exploratory Data Analysis to understand the data and fast growth patterns
    """
    # Merge original data with fast growth labels
    merged_df = df[df.year == 2012].merge(label_df[["comp_id", "fast_growth"]], on="comp_id", how="inner")
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists("3rd Assignment/figures"):
        os.makedirs("3rd Assignment/figures")
    
    # Distribution of numeric features by fast growth status
    features = [
        'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
        'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
        'curr_liab', 'share_eq', 'subscribed_cap'
    ]
    
    # Plot histograms for key metrics
    for feature in features[:5]:  # Limit to first 5 features to avoid too many plots
        plt.figure(figsize=(10, 6))
        
        # Log transform to handle skewed distributions, but handle negative values
        merged_df[f'log_{feature}'] = np.log1p(merged_df[feature].clip(lower=0))
        
        # Plot distributions by group
        sns.histplot(data=merged_df, x=f'log_{feature}', hue='fast_growth', element='step', common_norm=False)
        plt.title(f'Distribution of log({feature}) by Fast Growth Status')
        plt.tight_layout()
        plt.savefig(f"3rd Assignment/figures/dist_{feature}.png")
        plt.close()
    
    # Fast growth by industry
    industry_growth = merged_df.groupby('ind')['fast_growth'].agg(['mean', 'count'])
    industry_growth.columns = ['Fast Growth Rate', 'Count']
    industry_growth['Fast Growth Rate'] *= 100  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    ax = industry_growth['Fast Growth Rate'].plot(kind='bar')
    plt.title('Fast Growth Rate by Industry')
    plt.ylabel('Percentage of Fast Growing Firms')
    plt.xlabel('Industry (1=Manufacturing, 2=Services)')
    
    # Add count labels - using index values, not integer positions
    for i, idx in enumerate(industry_growth.index):
        ax.text(i, industry_growth.loc[idx, 'Fast Growth Rate'] + 1, 
                f"n={industry_growth.loc[idx, 'Count']}", ha='center')
    
    plt.tight_layout()
    plt.savefig("3rd Assignment/figures/growth_by_industry.png")
    plt.close()
    
    # Age vs. Fast Growth
    merged_df['age'] = 2012 - merged_df['founded_year']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='fast_growth', y='age', data=merged_df)
    plt.title('Firm Age by Fast Growth Status')
    plt.xlabel('Fast Growth (1=Yes, 0=No)')
    plt.ylabel('Age (years)')
    plt.tight_layout()
    plt.savefig("3rd Assignment/figures/age_vs_growth.png")
    plt.close()
    
    # Correlation matrix of features
    corr_matrix = merged_df[features + ['fast_growth']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig("3rd Assignment/figures/correlation_matrix.png")
    plt.close()
    
    print("EDA completed. Figures saved to '3rd Assignment/figures/' directory.")

def main():
    """
    Main function to prepare data for modeling
    """
    # Load panel data and filter to relevant years
    print("Loading and filtering data...")
    df = pd.read_csv("3rd Assignment/cs_bisnode_panel.csv")
    df = df[df["year"].between(2010, 2015)]
    
    # Create fast growth label based on 2012-2014 growth
    print("\nCreating fast growth labels...")
    label_df = create_fast_growth_label(df)
    
    # Perform exploratory data analysis
    print("\nPerforming exploratory data analysis...")
    perform_eda(df, label_df)

    # Merge back for modeling - using 2012 data to predict 2012-2014 growth
    print("\nPreparing final modeling dataset...")
    data = df[df.year == 2012].merge(label_df[["comp_id", "fast_growth"]], on="comp_id", how="inner")

    # Define model features
    model_features = [
        'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
        'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
        'curr_liab', 'share_eq', 'subscribed_cap', 'founded_year', 'foreign'
    ]

    # Drop rows with missing values for modeling features
    data = data.dropna(subset=model_features)
    data['age'] = 2012 - data['founded_year']  # Create age feature

    print("Final modeling dataset shape:", data.shape)
    print("Fast growth label distribution:")
    print(data["fast_growth"].value_counts())
    print(f"Percentage of fast-growing firms: {data['fast_growth'].mean()*100:.2f}%")

    # Save cleaned dataset for modeling
    data.to_csv("3rd Assignment/bisnode_firms_fastgrowth.csv", index=False)
    print("Prepared data saved to '3rd Assignment/bisnode_firms_fastgrowth.csv'")

if __name__ == "__main__":
    main()