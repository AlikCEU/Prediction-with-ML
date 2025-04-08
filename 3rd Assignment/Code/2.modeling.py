# -------------- modeling.py --------------
# This script performs the data preparation for modeling and feature engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Creates additional features for modeling and handles data preparation
    """
    print("Performing feature engineering...")
    
    # Create financial ratios that might indicate growth potential
    df['profit_margin'] = df['profit_loss_year'] / df['sales']
    df['current_ratio'] = df['curr_assets'] / df['curr_liab']
    df['asset_turnover'] = df['sales'] / (df['tang_assets'] + df['intang_assets'])
    df['debt_to_equity'] = df['curr_liab'] / df['share_eq']
    df['personnel_per_employee'] = df['personnel_exp'] / df['labor_avg']
    
    # Calculate firm age
    df['age'] = 2012 - df['founded_year']  
    
    # Log transform skewed features to normalize distributions
    for col in ['sales', 'labor_avg', 'personnel_exp', 'tang_assets', 'curr_assets']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Create size classes as numeric categories instead of strings
    df['size_category'] = pd.cut(
        df['labor_avg'], 
        bins=[0, 10, 50, 250, float('inf')],
        labels=[1, 2, 3, 4]  # Numeric labels instead of strings
    ).astype(int)  # Convert to integer
    
    # Handle missing values - fill with medians for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    print(f"Created {len(df.columns) - 50} new features")
    
    return df

def feature_selection(df):
    """
    Select the most important features for modeling
    """
    print("Selecting features for modeling...")
    
    # Base features (original + engineered that we want to keep)
    selected_features = [
        # Original financial features
        'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
        'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
        'curr_liab', 'share_eq', 'subscribed_cap',
        
        # Engineered features
        'age', 'foreign', 'profit_margin', 'current_ratio', 'asset_turnover',
        'debt_to_equity', 'personnel_per_employee',
        
        # Log-transformed features
        'log_sales', 'log_labor_avg', 'log_personnel_exp', 
        'log_tang_assets', 'log_curr_assets',
        
        # Categorical features (as numeric)
        'size_category', 'ind', 'region_m'
    ]
    
    # Check if all selected features exist in the dataframe
    existing_features = [f for f in selected_features if f in df.columns]
    missing_features = set(selected_features) - set(existing_features)
    
    if missing_features:
        print(f"Warning: Some selected features are missing from the dataframe: {missing_features}")
    
    # Include target variable
    modeling_df = df[existing_features + ['fast_growth']].copy()
    
    # Make sure all features are numeric for correlation calculations
    # Convert any remaining object columns to dummy variables
    object_cols = modeling_df.select_dtypes(include='object').columns
    if not object_cols.empty:
        print(f"Converting categorical columns to dummies: {list(object_cols)}")
        modeling_df = pd.get_dummies(modeling_df, columns=object_cols, drop_first=True)
    
    print(f"Final dataset has {modeling_df.shape[1]-1} features and {modeling_df.shape[0]} observations")
    return modeling_df

def visualize_features(df):
    """
    Create visualizations for important features
    """
    print("Creating feature visualizations...")
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists("3rd Assignment/figures"):
        os.makedirs("3rd Assignment/figures")
    
    # 1. Feature importance by correlation with target
    # First ensure we only use numeric columns for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if 'fast_growth' in numeric_df.columns:
        corr_with_target = numeric_df.corr()['fast_growth'].sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        # Skip the first one (correlation with itself) and show top 10
        top_correlations = corr_with_target[1:11] if len(corr_with_target) > 10 else corr_with_target[1:]
        top_correlations.plot(kind='bar')
        plt.title('Top Features by Correlation with Fast Growth')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.tight_layout()
        plt.savefig("3rd Assignment/figures/feature_correlation.png")
        plt.close()
    
    # 2. Distribution of key numeric features by fast growth status
    for feature in ['log_sales', 'profit_margin', 'age', 'current_ratio']:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, hue='fast_growth', element='step', common_norm=False)
            plt.title(f'Distribution of {feature} by Fast Growth Status')
            plt.tight_layout()
            plt.savefig(f"3rd Assignment/figures/dist_{feature}_by_growth.png")
            plt.close()
    
    # 3. Size category distribution by fast growth (using numeric size category)
    if 'size_category' in df.columns:
        plt.figure(figsize=(10, 6))
        # Create a crosstab with size categories
        size_labels = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
        crosstab = pd.crosstab(
            pd.Categorical(df['size_category'].map(size_labels), 
                          categories=['Micro', 'Small', 'Medium', 'Large'], 
                          ordered=True),
            df['fast_growth'], 
            normalize='index'
        )
        crosstab.plot(kind='bar')
        plt.title('Fast Growth Rate by Firm Size')
        plt.xlabel('Firm Size')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig("3rd Assignment/figures/size_vs_growth.png")
        plt.close()
    
    print("Visualizations saved to '3rd Assignment/figures/' directory")

def main():
    """
    Main function to prepare features for modeling
    """
    # Load data prepared in the previous step
    print("Loading prepared data...")
    try:
        df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")
    except FileNotFoundError:
        print("Error: Input file not found. Please run data_prep.py first.")
        return
    
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Select features for modeling
    modeling_df = feature_selection(df_engineered)
    
    # Visualize important features
    visualize_features(modeling_df)
    
    # Save prepared data for classification
    modeling_df.to_csv("3rd Assignment/bisnode_modeling_features.csv", index=False)
    print("Enhanced feature set saved to '3rd Assignment/bisnode_modeling_features.csv'")

if __name__ == "__main__":
    main()