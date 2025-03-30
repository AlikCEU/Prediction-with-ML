# 🏠 Airbnb Price Prediction Project

This project builds and evaluates a predictive pricing model for Airbnb listings in Montreal , with additional validation on Ottawa data. The work is part of a university-level machine learning assignment focused on modeling, feature engineering, and model generalization.

---

## 📁 Project Structure

.
├── airbnb_report.md         # Main report (Part I + Part II)
├── README.txt
├── requirements.txt         # Python environment
├── images/                  # Model feature importance plots
│   ├── Figure_1.png
│   └── Figure_2.png
├── 1_part/
│   ├── 1st_task_wrangling.py
│   ├── 2nd_3rd_tasks_modeling.py
│   └── 4th_task_feature_importance.py
├── 2_part/
│   ├── 5th_task_wrangling.py
│   ├── 6th_task_modeling_montreal_q1_2025.py
│   └── 6th_task_modeling_ottawa_q1_2025.py

---

## 📊 Models Used

- Linear Regression (OLS)
- LASSO Regression
- CART (Decision Trees)
- Random Forest Regressor
- XGBoost Regressor

---

## 🔁 Reproducibility

- Python 3.9+
- All packages listed in `requirements.txt`
- All datasets are pulled directly from public GitHub links

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📅 Datasets

- Main/Training: Montreal Q4 2024  
- Validation A: Montreal Q1 2025  
- Validation B: Ottawa Q1 2025  
- Data source: [InsideAirbnb](http://insideairbnb.com/get-the-data.html)

---

## 💡 Key Findings

- Tree-based ensemble models (Random Forest, XGBoost) consistently outperform linear models
- Maximum Nights, Accommodates, and Bathrooms are among the top price predictors
- Random Forest shows strong generalization across time and space

---

## ⚠️ macOS Users — XGBoost Note

If you encounter the following error on Mac:

```
XGBoostError: libomp.dylib not found
```

Fix it with:

```bash
brew install libomp
```

Then restart your terminal or IDE and re-run the code.

---

## 👤 Author
Allakhverdi Agakishiev
