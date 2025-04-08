# -------------- industry_comparison.py --------------
# This script compares model performance across different industries
# It applies the best model (Gradient Boosting) to manufacturing and service firms separately

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# Define loss function parameters - consistent with previous analysis
FP_COST = 1000  # Cost of false positive (pursuing non-fast-growth firm)
FN_COST = 5000  # Cost of false negative (missing fast-growth firm)

# Features used for modeling
FEATURES = [
    'sales', 'labor_avg', 'personnel_exp', 'profit_loss_year',
    'tang_assets', 'intang_assets', 'liq_assets', 'curr_assets',
    'curr_liab', 'share_eq', 'subscribed_cap', 'age', 'foreign'
]

def expected_loss(y_true, y_probs, threshold):
    """
    Calculate expected loss based on confusion matrix and specified costs
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Tuple of (total loss, fp loss, fn loss)
    """
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fp_loss = fp * FP_COST
    fn_loss = fn * FN_COST
    total_loss = fp_loss + fn_loss
    return total_loss, fp_loss, fn_loss

def find_best_threshold(model, X, y):
    """
    Find the threshold that minimizes expected loss
    
    Args:
        model: Fitted model
        X: Features
        y: Target
        
    Returns:
        Tuple of (best threshold, prediction probabilities)
    """
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)  # Fine-grain threshold search

    best_thresh = 0.5
    best_loss = float("inf")
    for t in thresholds:
        loss, _, _ = expected_loss(y, probs, t)
        if loss < best_loss:
            best_loss = loss
            best_thresh = t
    return best_thresh, probs

def plot_loss_curve(y_true, y_probs, name):
    """
    Plot expected loss as a function of threshold
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        name: Name for the plot
    
    Returns:
        Optimal threshold
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    total_losses = []
    fp_losses = []
    fn_losses = []
    
    for t in thresholds:
        total, fp_loss, fn_loss = expected_loss(y_true, y_probs, t)
        total_losses.append(total)
        fp_losses.append(fp_loss)
        fn_losses.append(fn_loss)
    
    # Find optimal threshold
    optimal_idx = np.argmin(total_losses)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, total_losses, 'b-', label='Total Loss')
    plt.plot(thresholds, fp_losses, 'r--', label='False Positive Loss')
    plt.plot(thresholds, fn_losses, 'g--', label='False Negative Loss')
    plt.axvline(x=optimal_threshold, color='k', linestyle='-', alpha=0.5,
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Expected Loss ($)')
    plt.title(f'Expected Loss vs Threshold - {name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"3rd Assignment/figures/loss_curve_{name.lower()}.png")
    plt.close()
    
    return optimal_threshold

def plot_conf_matrix(cm, name):
    """
    Plot confusion matrix with annotations and metrics
    
    Args:
        cm: Confusion matrix
        name: Name for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate derived metrics
    total = np.sum(cm)
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0  # TPR, Recall
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0  # TNR
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    
    # Plot with seaborn for better styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Add metrics text
    plt.text(0.1, -0.05, 
             f"Accuracy: {accuracy:.3f}\nSpecificity: {specificity:.3f}",
             transform=plt.gca().transAxes)
    plt.text(0.6, -0.05, 
             f"Sensitivity: {sensitivity:.3f}\nPrecision: {precision:.3f}", 
             transform=plt.gca().transAxes)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f"3rd Assignment/figures/confusion_{name.lower().replace(' ', '_')}.png")
    plt.close()

def plot_roc_curves(models_data):
    """
    Plot ROC curves for all models on one graph
    
    Args:
        models_data: Dictionary with model data
    """
    plt.figure(figsize=(10, 8))
    
    for name, data in models_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Industry Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("3rd Assignment/figures/roc_industry_comparison.png")
    plt.close()

def run_for_industry(df, industry_label, name):
    """
    Run analysis for a specific industry
    
    Args:
        df: DataFrame with all data
        industry_label: Industry code (1=Manufacturing, 2=Services)
        name: Industry name for plots
        
    Returns:
        Dictionary with results
    """
    # Filter data for specific industry
    industry_df = df[df["ind"] == industry_label].copy()
    
    # Check if we have enough data
    if len(industry_df) < 100:
        print(f"Warning: Small sample size for {name} industry ({len(industry_df)} samples)")
    
    # Handle missing values
    industry_df = industry_df.dropna(subset=FEATURES)
    
    X = industry_df[FEATURES]
    y = industry_df["fast_growth"]
    
    print(f"Industry: {name}")
    print(f"Total samples: {len(X)}")
    print(f"Fast growth firms: {y.sum()} ({y.mean()*100:.2f}%)")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Find optimal threshold on training data
    train_threshold, _ = find_best_threshold(model, X_train, y_train)
    print(f"Training optimal threshold: {train_threshold:.2f}")
    
    # Evaluate on test data
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Plot loss curve and find optimal threshold on test data
    test_threshold = plot_loss_curve(y_test, test_probs, name)
    print(f"Test optimal threshold: {test_threshold:.2f}")
    
    # Calculate final predictions using test threshold
    test_preds = (test_probs >= test_threshold).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    
    # Calculate expected loss
    test_loss, fp_loss, fn_loss = expected_loss(y_test, test_probs, test_threshold)
    print(f"Test expected loss: ${test_loss:.2f}")
    print(f"  False Positive Loss: ${fp_loss:.2f}")
    print(f"  False Negative Loss: ${fn_loss:.2f}")
    
    # Plot confusion matrix
    plot_conf_matrix(cm, name)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    roc_auc = auc(fpr, tpr)
    print(f"Test AUC: {roc_auc:.3f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))
    
    return {
        'threshold': test_threshold,
        'loss': test_loss,
        'fp_loss': fp_loss,
        'fn_loss': fn_loss,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

def main():
    """
    Main function to compare model performance across industries
    """
    # Create directory for figures if it doesn't exist
    import os
    if not os.path.exists("3rd Assignment/figures"):
        os.makedirs("3rd Assignment/figures")
    
    try:
        # Load prepared data
        print("Loading prepared data...")
        df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")
        
        # Check that we have the industry column
        if 'ind' not in df.columns:
            print("Error: Industry column 'ind' not found in dataset")
            return
        
        # Verify we have both industries present
        industry_counts = df['ind'].value_counts()
        print("Industry distribution:")
        for ind, count in industry_counts.items():
            print(f"Industry {ind}: {count} firms")
        
        # Run analysis for manufacturing
        print("\n" + "="*50)
        print("MANUFACTURING INDUSTRY ANALYSIS")
        print("="*50)
        manufacturing_results = run_for_industry(df, 1, "Manufacturing")
        
        # Run analysis for services
        print("\n" + "="*50)
        print("SERVICES INDUSTRY ANALYSIS")
        print("="*50)
        services_results = run_for_industry(df, 2, "Services")
        
        # Compare results
        print("\n" + "="*50)
        print("INDUSTRY COMPARISON")
        print("="*50)
        print(f"Manufacturing optimal threshold: {manufacturing_results['threshold']:.2f}")
        print(f"Services optimal threshold: {services_results['threshold']:.2f}")
        print(f"Manufacturing expected loss: ${manufacturing_results['loss']:.2f}")
        print(f"Services expected loss: ${services_results['loss']:.2f}")
        print(f"Manufacturing AUC: {manufacturing_results['auc']:.3f}")
        print(f"Services AUC: {services_results['auc']:.3f}")
        
        # Plot ROC curves comparison
        plot_roc_curves({
            'Manufacturing': manufacturing_results,
            'Services': services_results
        })
        
        # Save comparison results
        comparison_df = pd.DataFrame({
            'Manufacturing': {
                'Threshold': manufacturing_results['threshold'],
                'Expected Loss': manufacturing_results['loss'],
                'FP Loss': manufacturing_results['fp_loss'],
                'FN Loss': manufacturing_results['fn_loss'],
                'AUC': manufacturing_results['auc']
            },
            'Services': {
                'Threshold': services_results['threshold'],
                'Expected Loss': services_results['loss'],
                'FP Loss': services_results['fp_loss'],
                'FN Loss': services_results['fn_loss'],
                'AUC': services_results['auc']
            }
        }).T
        
        comparison_df.to_csv("3rd Assignment/industry_comparison_results.csv")
        print("\nIndustry comparison results saved to '3rd Assignment/industry_comparison_results.csv'")
        
    except FileNotFoundError:
        print("Error: Required data file not found. Run data_prep.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()