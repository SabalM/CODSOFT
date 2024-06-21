import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# Ensure you have the necessary NLTK data files
nltk.download('stopwords')

# Load the dataset, dropping unnamed columns
def load_dataset(file_path, encoding='latin1'):
    df = pd.read_csv(file_path, encoding=encoding)
    unnamed_columns = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=unnamed_columns, errors='ignore')
    return df

# Check for missing values and remove duplicates
def clean_data(df):
    print("Checking for missing values:")
    print(df.isnull().sum())
    
    df = df.dropna(subset=['v1', 'v2'])
    print("\nNumber of duplicate rows before removal:")
    print(df.duplicated().sum())
    df = df.drop_duplicates()
    print("\nNumber of duplicate rows after removal:")
    print(df.duplicated().sum())
    return df

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
def preprocess_dataset(df):
    df['v2'] = df['v2'].apply(preprocess_text)
    df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
    return df

if __name__ == "__main__":
    file_path = 'dataset/spam.csv'
    df = load_dataset(file_path)
    df = preprocess_dataset(df)
    df = clean_data(df)

    X = df['v2']
    y = df['v1'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.DataFrame({'v1': y_train, 'v2': X_train})
    test_df = pd.DataFrame({'v1': y_test, 'v2': X_test})

    train_df.to_csv('dataset/train_dataset.csv', index=False)
    test_df.to_csv('dataset/test_dataset.csv', index=False)

    print("Train and test datasets saved successfully.")
