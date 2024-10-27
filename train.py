import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load and preprocess data
data = pd.read_csv('synthetic_fraud_data_human_readable.csv')

# Select features and target variable
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['merchant_category', 'transaction_location', 'device_type', 'payment_type'])

# Split the data for training and testing (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('fraud_model.pkl', 'wb') as file:
    pickle.dump(model, file)
