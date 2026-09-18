# 💳 Credit Card Fraud Detection using XGBoost & SMOTE

This project demonstrates a machine learning workflow for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance and **XGBoost** for classification. It also evaluates different prediction thresholds to understand the trade-off between precision and recall.

## 📌 Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a very small percentage of all transactions.

This project focuses on:

* Loading the credit card transaction dataset
* Examining the class distribution
* Standardizing `Time` and `Amount`
* Splitting the data into training and testing sets
* Handling class imbalance using **SMOTE**
* Training an **XGBoost classifier**
* Evaluating the model at different classification thresholds
* Generating confusion matrices and classification reports
* Analyzing the most important features used by the model

## 🧠 Technologies Used

* **Python**
* **Pandas** - Data manipulation
* **NumPy** - Numerical operations
* **Scikit-learn** - Data preprocessing and evaluation
* **imbalanced-learn** - SMOTE oversampling
* **XGBoost** - Machine learning model
* **Matplotlib** - Visualization
* **Seaborn** - Feature importance visualization
* **Jupyter Notebook**

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset.

The dataset contains credit card transactions with a target column called `Class`:

* `0` → Legitimate transaction
* `1` → Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing a very small percentage of the total transactions.

The notebook automatically downloads the dataset from a public mirror if `creditcard.csv` is not already available.

## 🔄 Machine Learning Workflow

```text
Credit Card Transaction Dataset
              ↓
        Load Dataset
              ↓
     Examine Class Balance
              ↓
     Feature Preprocessing
              ↓
     Train-Test Split
              ↓
        Apply SMOTE
       (Training Data Only)
              ↓
       Train XGBoost
              ↓
     Predict Probabilities
              ↓
      Threshold Tuning
              ↓
 Classification Reports
 & Confusion Matrices
              ↓
   Feature Importance
```

## ⚙️ Data Preprocessing

The target variable `Class` is separated from the input features.

The `Time` and `Amount` features are standardized using `StandardScaler`.

The dataset is then divided into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to maintain the class distribution.

## ⚖️ Handling Class Imbalance with SMOTE

Because fraudulent transactions are heavily underrepresented, directly training a model on the original dataset can make it difficult for the model to identify fraud.

The project uses **SMOTE** to generate synthetic samples for the minority class.

Importantly, SMOTE is applied **only to the training data** after the train-test split. This prevents synthetic information from leaking into the test set.

```python
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)
```

## 🤖 XGBoost Model

The classification model used is **XGBoost (Extreme Gradient Boosting)**.

```python
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_clf.fit(X_train_res, y_train_res)
```

XGBoost is trained using the SMOTE-resampled training dataset.

## 🎯 Threshold Tuning

Instead of relying only on the default classification threshold of `0.5`, the notebook evaluates multiple thresholds:

```text
0.1
0.3
0.5
0.7
0.9
```

A lower threshold can classify more transactions as potentially fraudulent, which can increase recall but may also increase false positives.

For every threshold, the notebook generates:

* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

This helps examine how changing the threshold affects fraud detection.

## 📈 Feature Importance

After training, XGBoost's feature importance values are extracted.

The notebook displays the **top 15 most important features** used by the model.

```python
importances = xgb_clf.feature_importances_
```

These values provide an indication of which input features contributed most to the model's predictions.

## 📁 Project Structure

```text
CREDIT-CARD-FRAUD/
│
├── CREDIT_CARD_FRAUD.ipynb
├── creditcard.csv
└── README.md
```

> `creditcard.csv` may be downloaded automatically by the notebook if it is not present locally.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd CREDIT-CARD-FRAUD
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
CREDIT_CARD_FRAUD.ipynb
```

Run the cells from top to bottom.

The notebook will download the dataset automatically if `creditcard.csv` is not found.

## 📌 Key Concepts Demonstrated

### Class Imbalance

Fraud detection datasets commonly contain far fewer fraudulent transactions than legitimate ones.

### SMOTE

Creates synthetic minority-class samples to improve model learning on imbalanced data.

### XGBoost

A powerful gradient-boosting algorithm used for classification.

### Precision

Measures how many transactions predicted as fraud are actually fraudulent.

### Recall

Measures how many actual fraudulent transactions are successfully detected.

### F1-Score

Combines precision and recall into a single metric.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Threshold Tuning

Changes the probability threshold used to classify a transaction as fraudulent.

## ⚠️ Important Note

This project is intended for **educational and demonstration purposes**. The model should not be treated as a production-ready financial fraud detection system without additional validation, feature engineering, calibration, monitoring, and testing on real-world data.

## 🔮 Possible Future Improvements

* Hyperparameter tuning for XGBoost
* Compare XGBoost with Random Forest and Logistic Regression
* Evaluate ROC-AUC and PR-AUC
* Perform cross-validation
* Optimize the fraud detection threshold using a specific business cost function
* Add SHAP-based model explainability
* Handle temporal characteristics more carefully
* Build a real-time fraud prediction API
* Create a dashboard for monitoring predictions

## 👨‍💻 Author

**Hanzala**

B.Tech - Artificial Intelligence & Machine Learning
