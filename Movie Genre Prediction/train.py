import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

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

    # Train Model with progress bar
    model = RandomForestClassifier()

    epochs = 10  # Define number of epochs
    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            model.fit(X_train, y_train)
            pbar.update(1)

    print("Model training completed.")
