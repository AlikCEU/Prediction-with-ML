import pandas as pd

# -------------------------------------------------------
# 🔹 Part A: Montreal Q1 2025
# -------------------------------------------------------

print(" Cleaning Montreal Q1 2025...")

# Load dataset from GitHub
df_mtl = pd.read_csv("https://raw.githubusercontent.com/AlikCEU/Prediction-with-ML-2-HW/main/montreal%20q1.2025.csv")

# Drop completely empty columns (if any)
drop_cols_empty = ["neighbourhood_group_cleansed", "calendar_updated"]
df_mtl.drop(columns=[col for col in drop_cols_empty if col in df_mtl.columns], inplace=True)

# Convert price to float
df_mtl['price'] = df_mtl['price'].replace('[\$,]', '', regex=True).astype(float)
df_mtl = df_mtl[df_mtl['price'].notnull()]

# Clean and extract top amenities
df_mtl['amenities_clean'] = df_mtl['amenities'].str.replace('[{}"]', '', regex=True)
all_amenities = df_mtl['amenities_clean'].str.get_dummies(sep=', ')
top_amenities = all_amenities.sum().sort_values(ascending=False).head(10).index
df_amenities = all_amenities[top_amenities]
df_mtl = pd.concat([df_mtl, df_amenities], axis=1)

# Drop unnecessary columns
drop_cols_common = [
    'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name',
    'description', 'picture_url', 'host_id', 'host_url', 'host_name',
    'host_since', 'host_about', 'host_response_time', 'host_thumbnail_url',
    'host_picture_url', 'calendar_last_scraped', 'first_review', 'last_review',
    'amenities', 'amenities_clean'
]
df_mtl.drop(columns=[col for col in drop_cols_common if col in df_mtl.columns], inplace=True)

# Save cleaned data
df_mtl.to_csv("montreal_q1_cleaned.csv", index=False)
print("✅ Saved: montreal_q1_cleaned.csv")
print("Shape:", df_mtl.shape)
print("Top 10 amenities:", list(top_amenities))


# -------------------------------------------------------
# 🔹 Part B: Ottawa Q1 2025
# -------------------------------------------------------

print("\n Cleaning Ottawa Q1 2025...")

# Load dataset from GitHub
df_ott = pd.read_csv("https://raw.githubusercontent.com/AlikCEU/Prediction-with-ML-2-HW/main/ottawa%20q1.2025.csv")

# Drop empty columns (if any)
df_ott.drop(columns=[col for col in drop_cols_empty if col in df_ott.columns], inplace=True)

# Convert price to float
df_ott['price'] = df_ott['price'].replace('[\$,]', '', regex=True).astype(float)
df_ott = df_ott[df_ott['price'].notnull()]

# Clean and extract top amenities
df_ott['amenities_clean'] = df_ott['amenities'].str.replace('[{}"]', '', regex=True)
all_amenities = df_ott['amenities_clean'].str.get_dummies(sep=', ')
top_amenities = all_amenities.sum().sort_values(ascending=False).head(10).index
df_amenities = all_amenities[top_amenities]
df_ott = pd.concat([df_ott, df_amenities], axis=1)

# Drop unnecessary columns
df_ott.drop(columns=[col for col in drop_cols_common if col in df_ott.columns], inplace=True)

# Save cleaned data
df_ott.to_csv("ottawa_q1_cleaned.csv", index=False)
print("✅ Saved: ottawa_q1_cleaned.csv")
print("Shape:", df_ott.shape)
print("Top 10 amenities:", list(top_amenities))
