import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Define the paths to the dataset files
train_file = 'dataset/fraudTrain.csv'
test_file = 'dataset/fraudTest.csv'

# Load the datasets
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data])

# Convert date columns to datetime and extract useful features
combined_data['trans_date_trans_time'] = pd.to_datetime(combined_data['trans_date_trans_time'])
combined_data['trans_hour'] = combined_data['trans_date_trans_time'].dt.hour
combined_data['trans_day'] = combined_data['trans_date_trans_time'].dt.day
combined_data['trans_month'] = combined_data['trans_date_trans_time'].dt.month

# Drop unnecessary columns
drop_columns = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'trans_num', 'unix_time', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'merchant']
combined_data.drop(columns=drop_columns, inplace=True)

# Impute missing values for numerical features
numerical_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
imputer = SimpleImputer(strategy='median')
combined_data[numerical_features] = imputer.fit_transform(combined_data[numerical_features])

# Encode categorical variables
categorical_features = ['category', 'gender']
for feature in categorical_features:
    combined_data[feature] = LabelEncoder().fit_transform(combined_data[feature])

# Ensure all columns are numeric
combined_data = pd.get_dummies(combined_data, columns=categorical_features, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
combined_data[numerical_features] = scaler.fit_transform(combined_data[numerical_features])

# Split back into train and test data
train_data_processed = combined_data[:len(train_data)]
test_data_processed = combined_data[len(train_data):]

# Define the paths to save the preprocessed files
output_train_file = os.path.join('dataset', 'preprocessed_train_data.csv')
output_test_file = os.path.join('dataset', 'preprocessed_test_data.csv')

# Save preprocessed training data to a new CSV file
train_data_processed.to_csv(output_train_file, index=False)

# Save preprocessed test data to a new CSV file
test_data_processed.to_csv(output_test_file, index=False)

print("Preprocessed files saved successfully.")
