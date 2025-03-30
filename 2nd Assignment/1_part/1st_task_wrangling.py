import pandas as pd
import numpy as np

# -------------------------------------
# 📥 Load dataset
# -------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/AlikCEU/Prediction-with-ML-2-HW/main/montreal%20q4.2024.csv"
)

# -------------------------------------
# 🧹 Drop columns with 100% missing values
# -------------------------------------
df.drop(columns=["neighbourhood_group_cleansed", "calendar_updated"], inplace=True)

# -------------------------------------
# 💵 Convert price from string to float
# (e.g. "$1,234.00" → 1234.00)
# -------------------------------------
df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)

# Drop rows with missing prices
df = df[df["price"].notnull()]

# -------------------------------------
# 🛠️ Parse amenities into binary features
# -------------------------------------

# Clean string: remove braces and quotes
df["amenities_clean"] = df["amenities"].str.replace(r'[{}"]', "", regex=True)

# Split string and get dummy variables
all_amenities = df["amenities_clean"].str.get_dummies(sep=", ")

# Keep only top 10 most frequent amenities
top_amenities = all_amenities.sum().sort_values(ascending=False).head(10).index
df_amenities = all_amenities[top_amenities]

# Merge into main dataframe
df = pd.concat([df, df_amenities], axis=1)

# -------------------------------------
# 🗑️ Drop useless columns
# -------------------------------------
drop_cols = [
    "id", "listing_url", "scrape_id", "last_scraped", "source", "name",
    "description", "picture_url", "host_id", "host_url", "host_name",
    "host_since", "host_about", "host_response_time", "host_thumbnail_url",
    "host_picture_url", "calendar_last_scraped", "first_review", "last_review",
    "amenities", "amenities_clean"
]

df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# -------------------------------------
# ✅ Final check & save
# -------------------------------------
print("✅ Shape after cleaning:", df.shape)
print("✅ Top 10 amenities added as binary features:", list(top_amenities))

df.to_csv("montreal_q4_cleaned.csv", index=False)
print("✅ Cleaned file saved as 'montreal_q4_cleaned.csv'")

# Preview
print(df.head(10))
