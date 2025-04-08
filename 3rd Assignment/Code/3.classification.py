# -------------- classification.py --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

FP_COST = 1000
FN_COST = 5000

model_dict = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

def expected_loss(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * FP_COST + fn * FN_COST

def plot_conf_matrix(cm, model_name):
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"3rd Assignment/confusion_{model_name.lower()}.png")
    plt.close()

def main():
    df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")

    X = df[[
        'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
        'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
        'curr_liab', 'share_eq', 'subscribed_cap', 'age', 'foreign'
    ]]
    y = df["fast_growth"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in model_dict.items():
        losses = []
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_loss = float("inf")

        for train_idx, test_idx in skf.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs = model.predict_proba(X.iloc[test_idx])[:, 1]
            fold_y = y.iloc[test_idx]

            for t in thresholds:
                loss = expected_loss(fold_y, probs, t)
                if loss < best_loss:
                    best_loss = loss
                    best_threshold = t

        # Final model + confusion matrix
        model.fit(X, y)
        final_probs = model.predict_proba(X)[:, 1]
        final_preds = (final_probs >= best_threshold).astype(int)
        cm = confusion_matrix(y, final_preds)

        print(f"Model: {name}")
        print(f"Best threshold: {best_threshold:.2f}")
        print("Confusion matrix:")
        print(cm)

        plot_conf_matrix(cm, name)

if __name__ == "__main__":
    main()
