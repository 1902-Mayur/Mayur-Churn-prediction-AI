import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 7043

# ── Dynamic path — works on Windows, Mac, Linux ──────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, 'telecom_churn.csv')

gender = np.random.choice(['Male', 'Female'], n)
senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n)
dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
tenure = np.random.randint(1, 73, n)
phone_service = np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1])
multiple_lines = np.where(phone_service == 'Yes', np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]), 'No phone service')
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
online_security = np.where(internet_service != 'No', np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.50, 0.21]), 'No internet service')
online_backup = np.where(internet_service != 'No', np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]), 'No internet service')
device_protection = np.where(internet_service != 'No', np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]), 'No internet service')
tech_support = np.where(internet_service != 'No', np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22]), 'No internet service')
streaming_tv = np.where(internet_service != 'No', np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.40, 0.22]), 'No internet service')
streaming_movies = np.where(internet_service != 'No', np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.39, 0.39, 0.22]), 'No internet service')
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n, p=[0.34, 0.23, 0.22, 0.21])
monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.9, 1.1, n), 2)

# Churn logic - realistic probabilities
churn_prob = np.zeros(n)
churn_prob += np.where(contract == 'Month-to-month', 0.25, 0)
churn_prob += np.where(contract == 'One year', 0.05, 0)
churn_prob += np.where(internet_service == 'Fiber optic', 0.1, 0)
churn_prob += np.where(tenure < 12, 0.15, 0)
churn_prob += np.where(monthly_charges > 80, 0.1, 0)
churn_prob += np.where(tech_support == 'No', 0.07, 0)
churn_prob += np.where(online_security == 'No', 0.07, 0)
churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0)
churn_prob += np.where(senior == 1, 0.05, 0)
churn_prob = np.clip(churn_prob, 0.02, 0.92)
churn = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')

df = pd.DataFrame({
    'customerID': [f'CUST-{i:05d}' for i in range(1, n+1)],
    'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
    'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
    'MultipleLines': multiple_lines, 'InternetService': internet_service,
    'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
    'DeviceProtection': device_protection, 'TechSupport': tech_support,
    'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
    'Contract': contract, 'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges, 'Churn': churn
})

# Inject some missing values
missing_idx = np.random.choice(df.index, 11, replace=False)
df.loc[missing_idx, 'TotalCharges'] = np.nan

df.to_csv(OUTPUT, index=False)
print(f"Dataset created: {df.shape}")
print(f"Saved to: {OUTPUT}")
print(df['Churn'].value_counts())
