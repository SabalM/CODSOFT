import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load the preprocessed test data
test_data_processed = pd.read_csv('dataset/preprocessed_test_data.csv')

# Separate features and target variable
X_test = test_data_processed.drop('is_fraud', axis=1)
y_test = test_data_processed['is_fraud']

# Load the chosen model
# Uncomment the model you have saved and want to load

# Load the Logistic Regression model
# best_model = joblib.load('models/best_model_logistic_regression.joblib')

# Load the Decision Tree model
# best_model = joblib.load('models/best_model_decision_tree.joblib')

# Load the Random Forest model
best_model = joblib.load('models/best_model_random_forest.joblib')

# Evaluate the chosen model on the test set
y_test_pred = best_model.predict(X_test)

# Calculate test set evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_pred)

print("Test Set Metrics for the Chosen Model:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
