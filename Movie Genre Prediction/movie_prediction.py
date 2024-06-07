import pandas as pd
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

def tokenize_overview(mydata, overview_col):
    # Function to clean show overview
    # Return: each row as a list of tokens

    # removes punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = stopwords.words("English")

    # split text
    tokens = mydata[overview_col].map(lambda x: tokenizer.tokenize(str(x)))

    # strip white spaces & lower case
    tokens = tokens.map(lambda x: [i.lower().strip("_") for i in x])

    # remove stop words
    tokens = tokens.map(lambda x: [i for i in x if i not in stop_words])

    # remove empty strings
    tokens = tokens.map(lambda x: [i for i in x if i != ''])

    return tokens

file_path = 'dataset/train_data.txt'  

# Reading the dataset line by line with specified encoding
data = []
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip the line to remove leading/trailing whitespace and split by the custom delimiter
            split_line = line.strip().split(' ::: ')
            data.append(split_line)
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")

# Manually set the column names
columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
rows = data[1:]  # The rest are data rows
mydata = pd.DataFrame(rows, columns=columns)

# Print the first few rows and columns to debug
print("Data Columns:", mydata.columns)
print("First few rows of data:")
print(mydata.head())

# Ensure 'DESCRIPTION' is the column to process, adjust if necessary
description_column_name = 'DESCRIPTION'

if description_column_name in mydata.columns:
    tokenized_overviews = tokenize_overview(mydata, description_column_name)
    print(tokenized_overviews.head())
else:
    print(f"The specified column '{description_column_name}' does not exist in the dataset.")
