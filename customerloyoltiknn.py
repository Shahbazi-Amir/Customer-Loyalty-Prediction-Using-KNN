# Import necessary libraries
import pandas as pd

# Load the dataset
# Assuming the file "WA_Fn-UseC_-Telco-Customer-Churn.csv" is in the project directory
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

# Drop the 'customerID' column as it is not useful for modeling
data = data.drop(columns=['customerID'])

# Display the updated dataset
print("Dataset after dropping 'customerID':")
print(data.head())

# Convert 'TotalCharges' from object to numeric
# Some values might be non-numeric (e.g., empty strings), so we handle them
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Check for missing values after conversion
print("\nMissing values in 'TotalCharges' after conversion:")
print(data['TotalCharges'].isnull().sum())

# Fill missing values with the median of 'TotalCharges'
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Verify the changes
print("\nData types after converting 'TotalCharges':")
print(data.dtypes)

# Separate categorical and numerical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)

# Apply One-Hot Encoding to categorical columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Verify the changes
print("\nDataset after One-Hot Encoding:")
print(data.head())

# Convert 'Churn_Yes' column to numeric (1 for Yes, 0 for No)
data['Churn'] = data['Churn_Yes']

# Drop the original 'Churn_Yes' column as it is no longer needed
data = data.drop(columns=['Churn_Yes'])

# Verify the changes
print("\nDataset after converting 'Churn' to numeric:")
print(data['Churn'].value_counts())

# Import the train_test_split function from sklearn
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = data.drop(columns=['Churn'])  # Features (all columns except 'Churn')
y = data['Churn']  # Target (the 'Churn' column)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# Verify the split
print("\nNumber of samples in each set:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Import the KNeighborsClassifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model with k=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)

# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN model again with n_neighbors=5
knn_scaled = KNeighborsClassifier(n_neighbors=5)

# Train the model on scaled data
knn_scaled.fit(X_train_scaled, y_train)

# Predict on the scaled test data
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Calculate evaluation metrics again
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
precision_scaled = precision_score(y_test, y_pred_scaled)
recall_scaled = recall_score(y_test, y_pred_scaled)
f1_scaled = f1_score(y_test, y_pred_scaled)

# Print the results after scaling
print("\nModel Evaluation Metrics After Scaling:")
print(f"Accuracy: {accuracy_scaled:.2f}")
print(f"Precision: {precision_scaled:.2f}")
print(f"Recall: {recall_scaled:.2f}")
print(f"F1-Score: {f1_scaled:.2f}")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# محاسبه ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred_scaled)

# رسم نمودار
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix after Scaling')
plt.show()


from sklearn.model_selection import GridSearchCV

# لیست مقادیر k که می‌خوایم تست کنیم
param_grid = {'n_neighbors': list(range(1, 20))}

# ساخت GridSearchCV
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),   # مدل ما (KNN)
    param_grid=param_grid,              # مقادیر k
    cv=5,                               # 5-fold cross validation
    scoring='f1',                       # معیار بهینه‌سازی: F1-Score
    verbose=1,                          # لاگ توسط لاگ اجرا رو نشون بده
    n_jobs=-1                           # از تمام هسته‌های CPU استفاده کن
)

# اجرای GridSearch روی داده‌های استاندارد شده
grid_search.fit(X_train_scaled, y_train)

# نمایش بهترین k
print("بهترین k:", grid_search.best_params_['n_neighbors'])
print("بهترین F1-Score در آموزش:", grid_search.best_score_)



from sklearn.model_selection import GridSearchCV

# لیست مقادیر k که می‌خوایم تست کنیم
param_grid = {'n_neighbors': list(range(1, 20))}

# ساخت GridSearchCV
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),   # مدل ما (KNN)
    param_grid=param_grid,              # مقادیر k
    cv=5,                               # 5-fold cross validation
    scoring='f1',                       # معیار بهینه‌سازی: F1-Score
    verbose=1,                          # لاگ توسط لاگ اجرا رو نشون بده
    n_jobs=-1                           # از تمام هسته‌های CPU استفاده کن
)

# اجرای GridSearch روی داده‌های استاندارد شده
grid_search.fit(X_train_scaled, y_train)

# نمایش بهترین k
print("بهترین k:", grid_search.best_params_['n_neighbors'])
print("بهترین F1-Score در آموزش:", grid_search.best_score_)