from flask import Flask, render_template, request
import pandas as pd
import pickle
from twilio.rest import Client

# Load the trained model
with open('fraud_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = 'ACcadc5b6e303eb7781943f768e420ae8d'  # Your Twilio Account SID
TWILIO_AUTH_TOKEN = 'd1f1a3597c85e27b671ac1063e8ed0c7'      # Your Twilio Auth Token
TWILIO_PHONE_NUMBER = '+1 715 203 4737'
RECIPIENT_PHONE_NUMBER = '+91 8597441881'   # Recipient's phone number

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_sms_alert(transaction_details):
    message = f"ALERT! A potentially fraudulent transaction was detected:\n\n{transaction_details}"
    twilio_client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')  # Render index.html

# Route to display the form
@app.route('/form')
def form():
    return render_template('form.html')  # Render form.html when this route is accessed

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    features = {
        'timestamp': 0.0,
        'amount': 0.0,
        'customer_age': 0,
        'days_since_last_transaction': 0,
        'transaction_frequency': 0,
        'average_transaction_amount': 0,
        'is_weekend': 0,
        'is_night': 0,
        'distance_from_home': 0,
        'previous_fraud_flag': 0,
        'merchant_category': '',  # Categorical field
        'transaction_location': '',  # Categorical field
        'device_type': '',  # Categorical field
        'payment_type': ''  # Categorical field
    }

    # Retrieve form data
    features['timestamp'] = float(request.form['timestamp'])
    features['amount'] = float(request.form['amount'])
    features['customer_age'] = int(request.form['customer_age'])
    features['days_since_last_transaction'] = int(request.form['days_since_last_transaction'])
    features['transaction_frequency'] = int(request.form['transaction_frequency'])
    features['average_transaction_amount'] = float(request.form['average_transaction_amount'])
    features['is_weekend'] = int(request.form['is_weekend'])
    features['is_night'] = int(request.form['is_night'])
    features['distance_from_home'] = float(request.form['distance_from_home'])
    features['previous_fraud_flag'] = int(request.form['previous_fraud_flag'])
    features['merchant_category'] = request.form['merchant_category']
    features['transaction_location'] = request.form['transaction_location']
    features['device_type'] = request.form['device_type']
    features['payment_type'] = request.form['payment_type']
    
    # Convert to DataFrame and one-hot encode categorical variables
    transaction = pd.DataFrame([features])
    transaction_encoded = pd.get_dummies(transaction, columns=['merchant_category', 'transaction_location', 'device_type', 'payment_type'])
    
    # Align the columns with the model's expected input
    missing_cols = set(model.feature_names_in_) - set(transaction_encoded.columns)
    for c in missing_cols:
        transaction_encoded[c] = 0
    transaction_encoded = transaction_encoded[model.feature_names_in_]

    # Make a prediction
    prediction = model.predict(transaction_encoded)[0]
    result = "Transaction Approved" if prediction == 0 else "Transaction Blocked: Potential Fraud Detected"
    
    # Send an SMS alert if the transaction is flagged as fraudulent
    if prediction == 1:
        transaction_details = f"Transaction Details: {features}"
        send_sms_alert(transaction_details)
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)