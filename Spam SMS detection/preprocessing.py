import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure you have the necessary NLTK data files
nltk.download('stopwords')

# Load the dataset, dropping unnamed columns
def load_dataset(file_path, encoding='latin1'):
    # Load the dataset
    df = pd.read_csv(file_path, encoding=encoding)

    # Drop unnamed columns (assuming they are entirely empty)
    unnamed_columns = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=unnamed_columns, errors='ignore')

    return df

# Check for missing values and remove duplicates
def clean_data(df):
    print("Checking for missing values:")
    print(df.isnull().sum())
    
    # Drop rows with any missing values in 'v1' or 'v2'
    df = df.dropna(subset=['v1', 'v2'])

    print("\nNumber of duplicate rows before removal:")
    print(df.duplicated().sum())

    # Drop duplicate rows based on all columns
    df = df.drop_duplicates()

    print("\nNumber of duplicate rows after removal:")
    print(df.duplicated().sum())

    return df

# Preprocess text: Tokenize, lowercase, remove special characters and stop words, and apply stemming
def preprocess_text(text):
    # Tokenize and lowercase
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Apply preprocessing to the relevant columns in the dataset
def preprocess_dataset(df):
    if 'v2' in df.columns:
        df['v2'] = df['v2'].apply(preprocess_text)
    else:
        raise KeyError("Column 'v2' not found in the DataFrame. Adjust preprocessing logic.")

    return df

# Main function to load, clean, preprocess, and save the dataset
file_path = 'dataset/spam.csv'  # Adjust the path to your dataset file

# Load and clean the dataset
df = load_dataset(file_path)
df = clean_data(df)

# Preprocess the text data
df = preprocess_dataset(df)

# Save the cleaned dataset for later use
df.to_csv('dataset/processed_dataset.csv', index=False)

