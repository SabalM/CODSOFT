import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load the preprocessed training data
train_data_processed = pd.read_csv('dataset/preprocessed_train_data.csv')

# Separate features and target variable
X_train = train_data_processed.drop('is_fraud', axis=1)
y_train = train_data_processed['is_fraud']

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_split, y_train_split)
    
    # Validate the model
    y_val_pred = model.predict(X_val_split)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val_split, y_val_pred)
    precision = precision_score(y_val_split, y_val_pred)
    recall = recall_score(y_val_split, y_val_pred)
    f1 = f1_score(y_val_split, y_val_pred)
    roc_auc = roc_auc_score(y_val_split, y_val_pred)
    
    print(f"Validation Metrics for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print()

# Manually choose the best model
# Uncomment the model you want to save

# Save the Logistic Regression model
best_model = models['Logistic Regression']
joblib.dump(best_model, 'models/best_model_logistic_regression.joblib')

# Save the Decision Tree model
best_model = models['Decision Tree']
joblib.dump(best_model, 'models/best_model_decision_tree.joblib')

# Save the Random Forest model
best_model = models['Random Forest']
joblib.dump(best_model, 'models/best_model_random_forest.joblib')

print("Model training completed. Uncomment the model you want to save and rerun the script.")
