# Project: Predicting Customer Churn in Telco Dataset

## Objective
The goal of this project is to predict whether a customer will leave the company (Churn = Yes) or stay (Churn = No) using machine learning techniques.

## Dataset
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Contains 7043 customers and 21 features including demographics, services used, monthly charges, etc.

## Implementation Steps
1. **Data Loading & Initial Analysis**
2. **Data Preprocessing**
   - Convert `TotalCharges` to numeric type
   - Handle missing values
   - One-Hot Encoding for categorical variables
3. **Train/Test Split (80/20)**
4. **Feature Scaling with StandardScaler**
5. **Train KNN model**
6. **Hyperparameter Tuning with GridSearchCV**
7. **Model Evaluation using accuracy, precision, recall, and F1-score**

## Model Performance
| Metric       | Score |
|--------------|-------|
| **Accuracy**   | 0.79 |
| **Precision**  | 0.59 |
| **Recall**     | 0.59 |
| **F1-Score**   | 0.59 |

## Conclusion
Using the KNN algorithm and optimizing `k=17`, the model achieved acceptable performance. The improved recall score indicates that the model can better detect churned customers, which is crucial for business retention strategies.