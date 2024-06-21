import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def load_and_prepare_data(file_path):
    return pd.read_csv(file_path, encoding='latin1')

if __name__ == "__main__":
    test_file_path = 'dataset/test_dataset.csv'

    # Load and prepare test data
    test_data = load_and_prepare_data(test_file_path)
    
    X_test = test_data['v2'].values.astype('U')
    y_test = test_data['v1']

    # Load the trained model (choose one according to classifier)
    model = joblib.load('models/trained_model_NBC.pkl')
    # model = joblib.load('models/trained_model_LGC.pkl')
    # model = joblib.load('models/trained_model_SVM.pkl')

    y_test_pred = model.predict(X_test)

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
