# 🔮 Customer Churn Risk Modeling

> **A complete beginner-friendly data science project** for predicting customer churn with machine learning.

---

## 📌 What is This Project?

This project predicts which customers are likely to **leave a telecom company** (churn). By identifying at-risk customers early, businesses can take action to retain them.

**Perfect for:**
- 🎓 Beginners learning data science
- 💼 Interview preparation
- 📚 Understanding end-to-end ML workflow

---

## 🎯 Quick Overview

| Item | Details |
|------|---------|
| **Problem Type** | Binary Classification |
| **Dataset** | Telco Customer Churn (Kaggle) |
| **Size** | 7,043 customers, 21 features |
| **Best Model** | Random Forest |
| **ROC-AUC** | ~0.93 |

---

## 📊 Key Findings

### From Data Analysis:
- 📉 **27% of customers churned** (imbalanced data)
- 📅 **New customers churn more** (first 12 months are critical)
- 📝 **Month-to-month contracts = 43% churn** (vs 3% for 2-year contracts)
- 🌐 **Fiber optic has highest churn** (possible service issues)

### From Model:
- 🤖 **Random Forest outperformed Logistic Regression**
- 🎯 **Optimal threshold: 0.3-0.4** (lower than default 0.5)
- 💰 **Threshold selection based on business costs**

---

## 📁 Project Structure

```
customer-churn-project/
│
├── churn_analysis.ipynb      # 📓 Main notebook (ALL CODE HERE)
│   └── Part 1-7: Complete analysis with visualizations
│
├── PROJECT_LEARNING_GUIDE.md # 📚 DETAILED learning guide
│   └── Beginner explanations, interview Q&A, glossary
│
├── README.md                 # 📋 This file
│
├── requirements.txt          # 📦 Python dependencies
│
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # 📊 Dataset
```

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Open the Notebook
- Open `churn_analysis.ipynb` in VS Code or Jupyter
- Click "Run All" to execute all cells

### Step 3: Learn from the Guide
- Read `PROJECT_LEARNING_GUIDE.md` for detailed explanations

---

## 📓 What's in the Notebook?

| Part | Content |
|------|---------|
| **Part 1** | Load data, explore structure, understand target variable |
| **Part 2** | Clean data, fix data types, handle missing values |
| **Part 3** | EDA with visualizations and insights |
| **Part 4** | Handle class imbalance with SMOTE |
| **Part 5** | Train Logistic Regression + Random Forest |
| **Part 6** | Optimize threshold based on business costs |
| **Part 7** | Analyze errors (confusion matrix, false negatives) |


---

## 💼 Business Recommendations

Based on this analysis, the company should:

1. **Focus on new customers** - Retention efforts in the first 12 months
2. **Incentivize longer contracts** - Discounts for annual/2-year plans
3. **Investigate fiber optic** - Why are these customers leaving?
4. **Deploy the model** - Proactively contact high-risk customers

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Visualizations
- **scikit-learn** - Machine learning models
- **imbalanced-learn** - SMOTE for class imbalance

---

## 📈 Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | ~75% | ~86% |
| Precision | ~73% | ~85% |
| Recall | ~78% | ~87% |
| F1-Score | ~75% | ~86% |
| ROC-AUC | ~0.89 | ~0.93 |

---

## 📝 License

This project is for educational purposes. Dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

*Created with 💙 for learning data science*
