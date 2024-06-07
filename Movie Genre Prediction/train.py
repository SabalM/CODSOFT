import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data(file_path):
    mydata = pd.read_csv(file_path)
    return mydata

if __name__ == "__main__":
    train_file_path = 'dataset/processed_train_data.csv'

    # Load and prepare training data
    train_data = load_and_prepare_data(train_file_path)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(train_data['CLEANED_DESCRIPTION'])
    y_train = train_data['GENRE']

    # Train Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Optionally, you can save the trained model for later use
    # Save the trained model
    import joblib
    joblib.dump(model, 'trained_model.pkl')

    print("Model training completed.")
