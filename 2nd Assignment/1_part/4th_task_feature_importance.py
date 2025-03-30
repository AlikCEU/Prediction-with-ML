import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------------------------
# 📥 Load dataset
# -------------------------------------
df = pd.read_csv("montreal_q4_cleaned.csv")
y = df["price"]
X = df.drop(columns=["price"])

# -------------------------------------
# 🔎 Define column types
# -------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -------------------------------------
# 🧼 Preprocessing
# -------------------------------------
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# -------------------------------------
# ✂️ Train-test split and transform
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_transformed = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()

# -------------------------------------
# 🌳 Train models
# -------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
top_rf = rf_importances.head(10)

xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train_transformed, y_train)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
top_xgb = xgb_importances.head(10)

# -------------------------------------
# 🏷️ Rename features for readability
# -------------------------------------
name_map = {
    'num__maximum_nights': 'Maximum Nights',
    'num__accommodates': 'Accommodates',
    'num__bathrooms': 'Bathrooms',
    'num__bedrooms': 'Bedrooms',
    'num__beds': 'Beds',
    'num__latitude': 'Latitude',
    'num__longitude': 'Longitude',
    'num__review_scores_accuracy': 'Review Accuracy',
    'cat__host_neighbourhood_Notre-Dame-de-Grace': 'Host Neighborhood: NDG',
    'cat__room_type_Entire home/apt': 'Room Type: Entire',
    'cat__room_type_Private room': 'Room Type: Private',
    'cat__license_271191, expires: 2025-10-31': 'License: 2025',
    'cat__host_acceptance_rate_59%': 'Acceptance Rate: 59%',
    'cat__neighborhood_overview_Welcome to PLATEAU. It\'s one of the trendiest neighborhood\'s in Montreal. This area is a fully bilingual area. Feel free to speak English or French. PLATEAU area is a very trendy and friendly area to meet friends or make new ones. Lots of cafe\'s, restaurants, & drinking establishments. PLATEAU is a very family oriented neighborhood.':
        'Neighborhood: Plateau',
    'cat__neighborhood_overview_Perfectly located in Old Montreal, a corner from Square Victoria, Notre-Dame, St-Paul Street and the Old Port. Enjoy the beautiful architecture!<br />A short 5 minutes walk and you are Downtown, on Ste-Catherine Street<br />The space is surrounded by coffee shop, nice bars, amazing restaurant, museums and shopping!':
        'Neighborhood: Old Montreal'
}

short_rf = top_rf.rename(index=lambda x: name_map.get(x, x))
short_xgb = top_xgb.rename(index=lambda x: name_map.get(x, x))

# -------------------------------------
# 📊 Print results
# -------------------------------------
print("\n🔍 Top 10 Feature Importances — Random Forest:")
print(short_rf)

print("\n🔍 Top 10 Feature Importances — XGBoost:")
print(short_xgb)

# -------------------------------------
# 📈 Plot feature importances
# -------------------------------------
sns.set(font_scale=1.1)

plt.figure(figsize=(10, 6))
sns.barplot(x=short_rf.values, y=short_rf.index)
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=short_xgb.values, y=short_xgb.index)
plt.title("Top 10 Feature Importances - XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
