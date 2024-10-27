import pandas as pd
import numpy as np

# Parameters
num_rows = 1000  # Number of transactions
fraud_ratio = 0.1  # Ratio of fraudulent transactions

# Seed for reproducibility
np.random.seed(42)

# Transaction ID
transaction_ids = np.arange(1, num_rows + 1)

# Timestamp (simulating time of day in seconds)
timestamps = np.random.randint(1, 86400, size=num_rows)  # 86400 seconds in a day

# Amounts (random amounts between $1 and $1000)
amounts = np.round(np.random.uniform(1, 1000, num_rows), 2)

# Merchant category (random selection from common categories)
merchant_categories = np.random.choice(
    ["Grocery", "Electronics", "Dining", "Travel", "Entertainment", "Clothing"],
    num_rows
)

# Transaction location (random US cities)
transaction_locations = np.random.choice(
    ["New York", "San Francisco", "Los Angeles", "Chicago", "Houston", "Seattle"],
    num_rows
)

# Device type (randomly assigned)
device_types = np.random.choice(["Mobile", "Web", "In-store"], num_rows)

# Payment type (randomly assigned)
payment_types = np.random.choice(["Credit Card", "Debit Card", "Bank Transfer"], num_rows)

# Customer age (random ages between 18 and 70)
customer_ages = np.random.randint(18, 71, num_rows)

# Days since last transaction (random values between 0 and 30)
days_since_last_transaction = np.random.randint(0, 31, num_rows)

# Transaction frequency (random count of transactions in the last 30 days)
transaction_frequencies = np.random.randint(1, 20, num_rows)

# Average transaction amount in the last 30 days
average_transaction_amounts = np.round(np.random.uniform(20, 500, num_rows), 2)

# Weekend transaction indicator (0 for weekday, 1 for weekend)
is_weekend = np.random.choice([0, 1], num_rows)

# Night transaction indicator (0 for day, 1 for night)
is_night = np.random.choice([0, 1], num_rows)

# Distance from home (random values between 0 and 100 miles)
distance_from_home = np.round(np.random.uniform(0, 100, num_rows), 2)

# Previous fraud flag (0 if no history, 1 if history)
previous_fraud_flag = np.random.choice([0, 1], num_rows)

# Fraud label (10% fraud cases)
is_fraud = np.random.choice([0, 1], size=num_rows, p=[1 - fraud_ratio, fraud_ratio])

# Create DataFrame
data = pd.DataFrame({
    'transaction_id': transaction_ids,
    'timestamp': timestamps,
    'amount': amounts,
    'merchant_category': merchant_categories,
    'transaction_location': transaction_locations,
    'device_type': device_types,
    'payment_type': payment_types,
    'customer_age': customer_ages,
    'days_since_last_transaction': days_since_last_transaction,
    'transaction_frequency': transaction_frequencies,
    'average_transaction_amount': average_transaction_amounts,
    'is_weekend': is_weekend,
    'is_night': is_night,
    'distance_from_home': distance_from_home,
    'previous_fraud_flag': previous_fraud_flag,
    'is_fraud': is_fraud
})

# Save to CSV
data.to_csv('synthetic_fraud_data_human_readable.csv', index=False)
print("Synthetic human-readable fraud data generated and saved as 'synthetic_fraud_data_human_readable.csv'.")
