# -------------- data_prep.py --------------
# (оставляем как есть, уже завершено)


# -------------- modeling.py --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc


def main():
    df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")

    y = df["fast_growth"]
    features = [
        'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
        'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
        'curr_liab', 'share_eq', 'subscribed_cap', 'age', 'foreign'
    ]

    X = df[features]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    plt.figure()
    for name, model in models.items():
        y_scores = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("3rd Assignment/roc_all_models.png")
    plt.close()
    print("ROC curves saved as roc_all_models.png")

if __name__ == "__main__":
    main()
