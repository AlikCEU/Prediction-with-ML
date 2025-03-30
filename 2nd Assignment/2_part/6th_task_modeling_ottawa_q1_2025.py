import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------------
# 📥 Load cleaned Ottawa Q1 2025 data
# -------------------------------------
df = pd.read_csv("ottawa_q1_cleaned.csv")
y = df["price"]
X = df.drop(columns=["price"])

# -------------------------------------
# 🔎 Identify column types
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
X_test_transformed = preprocessor.transform(X_test)

# -------------------------------------
# ⚙️ Define models
# -------------------------------------
models = {
    "OLS": LinearRegression(),
    "LASSO": Lasso(alpha=0.1),
    "CART": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

# -------------------------------------
# 🧪 Evaluate models
# -------------------------------------
print("\n🏙️ Ottawa Q1 2025 — Model Evaluation")
results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    end = time.time()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    duration = end - start

    print(f"{name:<15} | RMSE: {rmse:.2f} | R²: {r2:.4f} | MAE: {mae:.2f} | Time: {duration:.2f}s")
    results.append((name, rmse, r2, mae, duration))
