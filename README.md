# LOAN-DEFAULT-PREDICTION

# 💰 Loan Default Prediction – Machine Learning Project

## 🧠 Overview
This project predicts whether a **loan applicant will default** or **successfully repay** using advanced **tree-based machine learning models** like **Decision Trees**, **Random Forest**, and **Gradient Boosting (XGBoost / LightGBM)**.

The goal is to help banks and financial institutions make smarter, data-driven lending decisions and reduce credit risk.  

---

## 🎯 Objectives
- Analyze loan applicant data to find patterns in defaults.  
- Build and compare models (Decision Tree, Random Forest, Gradient Boosting).  
- Evaluate each model using accuracy, precision, recall, F1-score, and ROC-AUC.  
- Rank feature importance to understand the key drivers of loan default.  

---

## 📊 Dataset Description
The dataset used is `loan_default.csv` (or can be simulated using a public dataset from Kaggle).  

| Feature | Description |
|:--|:--|
| `Loan_ID` | Unique loan identifier |
| `Gender` | Male / Female |
| `Married` | Applicant’s marital status |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Non-Graduate |
| `Self_Employed` | Applicant’s employment type |
| `ApplicantIncome` | Income of applicant |
| `CoapplicantIncome` | Income of co-applicant |
| `LoanAmount` | Loan amount in thousands |
| `Loan_Amount_Term` | Term of loan in months |
| `Credit_History` | 1 = good history, 0 = bad |
| `Property_Area` | Urban / Rural / Semiurban |
| `Loan_Status` | Target (Y = Approved, N = Defaulted) |

---

## ⚙️ Technologies Used
| Category | Libraries / Tools |
|:--|:--|
| Language | Python 🐍 |
| IDE | Jupyter Notebook / VS Code |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Model Evaluation | `roc_auc_score`, `confusion_matrix`, `classification_report` |
| Deployment (optional) | `streamlit` |

---

## 🔍 Workflow

### 1️⃣ Data Preprocessing
- Handle missing values (`SimpleImputer`).  
- Encode categorical columns (`OneHotEncoder`).  
- Scale numerical data (`StandardScaler`).  
- Split data into train-test sets (80/20).

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
