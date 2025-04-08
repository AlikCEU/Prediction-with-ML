# -------------- industry_comparison.py --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

FP_COST = 1000
FN_COST = 5000

FEATURES = [
    'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
    'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
    'curr_liab', 'share_eq', 'subscribed_cap', 'age', 'foreign'
]

def expected_loss(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * FP_COST + fn * FN_COST

def find_best_threshold(model, X, y):
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)

    best_thresh = 0.5
    best_loss = float("inf")
    for t in thresholds:
        loss = expected_loss(y, probs, t)
        if loss < best_loss:
            best_loss = loss
            best_thresh = t
    return best_thresh, probs

def plot_conf_matrix(cm, name):
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f"3rd Assignment/confusion_{name.lower()}.png")
    plt.close()

def run_for_industry(df, industry_label, name):
    df = df[df["ind"] == industry_label].dropna(subset=FEATURES)
    X = df[FEATURES]
    y = df["fast_growth"]

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    threshold, probs = find_best_threshold(model, X, y)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds)

    print(f"Industry: {name}")
    print(f"Best threshold: {threshold:.2f}")
    print("Confusion matrix:")
    print(cm)

    plot_conf_matrix(cm, name)

def main():
    df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")

    print("--- Manufacturing ---")
    run_for_industry(df, 1, "Manufacturing")

    print("\n--- Services ---")
    run_for_industry(df, 2, "Services")

if __name__ == "__main__":
    main()