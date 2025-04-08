# -------------- classification.py --------------
# This script performs classification modeling for fast growth prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Define costs for misclassification
FP_COST = 1000  # Cost of false positive (wrongly predicting fast growth)
FN_COST = 5000  # Cost of false negative (missing a fast growing firm)

# Dictionary of models to evaluate
model_dict = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

def expected_loss(y_true, y_probs, threshold):
    """
    Calculate expected loss based on misclassification costs
    """
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * FP_COST + fn * FN_COST

def find_optimal_threshold(y_true, y_probs):
    """
    Find the threshold that minimizes expected loss
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    losses = []
    
    for t in thresholds:
        loss = expected_loss(y_true, y_probs, t)
        losses.append(loss)
    
    best_idx = np.argmin(losses)
    best_threshold = thresholds[best_idx]
    best_loss = losses[best_idx]
    
    return best_threshold, best_loss

def plot_conf_matrix(cm, model_name):
    """
    Plot confusion matrix for a model
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"3rd Assignment/figures/confusion_{model_name.lower()}.png")
    plt.close()

def plot_roc_curves(models_results):
    """
    Plot ROC curves for all models
    """
    plt.figure(figsize=(10, 8))
    
    for name, results in models_results.items():
        fpr = results['fpr']
        tpr = results['tpr']
        roc_auc = results['auc']
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Fast Growth Classification')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("3rd Assignment/figures/roc_curves.png")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Features - {model_name}')
        plt.tight_layout()
        plt.savefig(f"3rd Assignment/figures/feature_importance_{model_name.lower()}.png")
        plt.close()

def main():
    """
    Main function to train and evaluate models
    """
    print("Loading modeling data...")
    try:
        # Try to load the enhanced feature set first
        df = pd.read_csv("3rd Assignment/bisnode_modeling_features.csv")
        print("Using enhanced feature set from modeling.py")
    except FileNotFoundError:
        # Fall back to the original dataset if enhanced features aren't available
        print("Enhanced feature set not found. Using original dataset...")
        df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Separate target and features
    y = df["fast_growth"]
    
    # Drop non-feature columns if they exist
    non_features = ["comp_id", "year", "fast_growth"]
    X = df.drop(columns=[col for col in non_features if col in df.columns])
    
    # Print feature set summary
    print(f"Number of features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Percentage of fast-growing firms: {y.mean()*100:.2f}%")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create stratified k-fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results for comparison
    models_results = {}
    
    print("\nEvaluating classification models...")
    
    # Evaluate each model
    for name, model in model_dict.items():
        print(f"\nTraining {name}...")
        
        # Get cross-validation accuracy scores
        cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model on all data
        from sklearn.model_selection import cross_val_predict
        
        # Get probabilities
        y_probs = cross_val_predict(model, X_scaled, y, cv=skf, method='predict_proba')[:, 1]
        
        # Find optimal threshold
        best_threshold, best_loss = find_optimal_threshold(y, y_probs)
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"Expected loss: {best_loss:.2f}")
        
        # Make predictions with optimal threshold
        y_pred = (y_probs >= best_threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        plot_conf_matrix(cm, name)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_probs)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Store results for comparison
        models_results[name] = {
            'accuracy': cv_scores.mean(),
            'threshold': best_threshold,
            'loss': best_loss,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'model': model
        }
        
        # Plot feature importance for tree-based models
        if name in ['RandomForest', 'GradientBoosting']:
            plot_feature_importance(model, X.columns, name)
    
    # Plot ROC curves for all models
    plot_roc_curves(models_results)
    
    # Determine best model based on expected loss
    best_model_name = min(models_results, key=lambda k: models_results[k]['loss'])
    print(f"\nBest model: {best_model_name}")
    print(f"Expected loss: {models_results[best_model_name]['loss']:.2f}")
    print(f"Optimal threshold: {models_results[best_model_name]['threshold']:.4f}")
    print(f"ROC AUC: {models_results[best_model_name]['auc']:.4f}")
    
    # Save best model results
    best_results = {
        'model_name': best_model_name,
        'accuracy': models_results[best_model_name]['accuracy'],
        'threshold': models_results[best_model_name]['threshold'],
        'loss': models_results[best_model_name]['loss'],
        'auc': models_results[best_model_name]['auc']
    }
    
    # Save results to file for industry comparison script
    best_results_df = pd.DataFrame([best_results])
    best_results_df.to_csv("3rd Assignment/best_model_results.csv", index=False)
    print("\nBest model results saved to '3rd Assignment/best_model_results.csv'")

if __name__ == "__main__":
    main()
    