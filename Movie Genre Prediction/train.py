import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_and_prepare_data(file_path):
    mydata = pd.read_csv(file_path, encoding='utf-8')
    return mydata

if __name__ == "__main__":
    train_file_path = 'dataset/processed_train_data.csv'

    # Load and prepare training data
    train_data = load_and_prepare_data(train_file_path)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(train_data['CLEANED_DESCRIPTION'])
    y_train = train_data['GENRE']

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=pd.unique(y_train), y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(pd.unique(y_train), class_weights)}

    # Choose a classifier (Naive Bayes, Logistic Regression, or SVM)
    # Uncomment the desired classifier and comment out the others

    # Naive Bayes classifier (obtained accuracy= 0.44)
    classifier = MultinomialNB()

    # Logistic Regression classifier (obtained accuracy= 0.50)
    # classifier = LogisticRegression(class_weight=class_weight_dict, max_iter=1000)

    # Support Vector Machine (SVM) classifier (obtained accuracy= 0.56)
    # classifier = LinearSVC(class_weight=class_weight_dict, max_iter=1000)

    # Pipeline: train tf-vec and pass input to classifier
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", classifier)
    ])

    # Train the classifier
    model.fit(train_data['CLEANED_DESCRIPTION'], y_train)

    # Save the trained model (choose one according to classifier)
    joblib.dump(model, 'trained_model_NBC.pkl')
    # joblib.dump(model, 'trained_model_LGC.pkl')
    # joblib.dump(model, 'trained_model_SVM.pkl')
