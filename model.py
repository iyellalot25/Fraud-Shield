import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

data=pd.read_csv('creditcard.csv')

A = data.drop(columns=['Class'])
B = data['Class']

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.3, random_state=42)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(A_train, B_train)

B_pred=model.predict(A_test)
print("Model Performance:")
print("Accuracy:", accuracy_score(B_test, B_pred))
print("Precision:", precision_score(B_test, B_pred))
print("Recall:", recall_score(B_test, B_pred))

def generate_transaction():
    transaction = {
        "Time": input("Enter Time in Seconds: "),  #*assuming max time based on dataset
        "Amount": input("Enter Transaction Amount: ")  #*assuming max amount based on dataset,
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = round(random.uniform(-5, 5), 2)  
    return transaction

def monitor_transaction(transaction):
    transaction_df = pd.DataFrame([transaction], columns=A.columns)
    
    # fraudulent (1) or legitimate (0)
    prediction = model.predict(transaction_df)[0]
    return "Transaction Blocked: Potential Fraud Detected" if prediction == 1 else "Transaction Approved"

new_transaction = generate_transaction()
print("Initializing FraudShield")
print("New Transaction:", new_transaction)
print("Monitoring Result:", monitor_transaction(new_transaction))