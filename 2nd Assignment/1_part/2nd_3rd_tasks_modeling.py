import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------------------------
# 📥 Load data and prepare features
# -------------------------------------
df = pd.read_csv("montreal_q4_cleaned.csv")

y = df["price"]
X = df.drop(columns=["price"])

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -------------------------------------
# 🧼 Preprocessing pipelines
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
# ✂️ Split data into train and test
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data loaded and split")
print("   X_train shape:", X_train.shape)
print("   X_test  shape:", X_test.shape)

# -------------------------------------
# ⚙️ Define models
# -------------------------------------
models = {
    "OLS": LinearRegression(),
    "LASSO": LassoCV(cv=5),
    "CART": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

# -------------------------------------
# 🧪 Train and evaluate models
# -------------------------------------
results = []

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    start = time.time()

    # Create full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - start

    results.append({
        "Model": name,
        "RMSE": round(rmse, 2),
        "R²": round(r2, 4),
        "MAE": round(mae, 2),
        "Time (s)": round(elapsed, 2)
    })

# -------------------------------------
# 📊 Print summary table
# -------------------------------------
print("\n📊 Model Comparison:")
for r in results:
    print(f"{r['Model']:15} | RMSE: {r['RMSE']:7} | R²: {r['R²']:6} | MAE: {r['MAE']:7} | Time: {r['Time (s)']}s")
