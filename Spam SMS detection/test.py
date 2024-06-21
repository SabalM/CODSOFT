import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
