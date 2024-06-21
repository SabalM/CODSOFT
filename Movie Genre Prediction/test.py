import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_and_prepare_data(file_path):
    mydata = pd.read_csv(file_path, encoding='utf-8')
    return mydata

def load_solution_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip().split(' ::: ') for line in lines]
    ids = [row[0] for row in data]
    genres = [row[2] for row in data]
    return pd.DataFrame({'ID': ids, 'GENRE': genres})

if __name__ == "__main__":
    test_file_path = 'dataset/processed_test_data.csv'
    solution_file_path = 'dataset/test_data_solution.txt'

    # Load and prepare test data
    test_data = load_and_prepare_data(test_file_path)

    # Load the solution file to get the ground truth labels
    solution_data = load_solution_file(solution_file_path)

    # Ensure both 'ID' columns have the same data type
    test_data['ID'] = test_data['ID'].astype(str)
    solution_data['ID'] = solution_data['ID'].astype(str)

    # Ensure test data and solution data align properly
    test_data = test_data.merge(solution_data, on='ID')

    # Load the trained model (choose one based on model name)
    model = joblib.load('models/trained_model_NBC.pkl')
    # model = joblib.load('models/trained_model_LGC.pkl')
    # model = joblib.load('models/trained_model_SVM.pkl')

    # Predict on test data
    X_test = test_data['CLEANED_DESCRIPTION']
    y_test = test_data['GENRE']  # Use the correct column name 'GENRE' from the solution file

    y_test_pred = model.predict(X_test)

    # Calculate test set evaluation metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    print("Test Set Metrics for the Chosen Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
