import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_prepare_data(file_path):
    mydata = pd.read_csv(file_path)
    return mydata

if __name__ == "__main__":
    train_file_path = 'dataset/processed_train_data.csv'

    # Load and prepare training data
    train_data = load_and_prepare_data(train_file_path)

    # TF-IDF Vectorization and Naive Bayes Classifier in a pipeline
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    # Train classifier
    model.fit(train_data['CLEANED_DESCRIPTION'], train_data['GENRE'])

    # Save the trained model
    joblib.dump(model, 'trained_model.pkl')
    print("Model training completed and saved as 'trained_model.pkl'.")
