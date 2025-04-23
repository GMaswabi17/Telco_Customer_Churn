import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("_data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing data
df.dropna(inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# 👉 Extract Churn target
y = df['Churn']
df.drop('Churn', axis=1, inplace=True)

# Feature Engineering
service_cols = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies'
]

df['TotalServicesUsed'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)
df['HasInternetPhoneCombo'] = ((df['InternetService'] != 'No') & (df['PhoneService'] == 'Yes')).astype(int)

def tenure_group(tenure):
    if tenure <= 12:
        return '0-1 year'
    elif tenure <= 24:
        return '1-2 years'
    elif tenure <= 48:
        return '2-4 years'
    elif tenure <= 60:
        return '4-5 years'
    else:
        return '5+ years'

df['tenure_group'] = df['tenure'].apply(tenure_group)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Scale numerical features
scaler = MinMaxScaler()
df[['TotalCharges', 'tenure', 'MonthlyCharges']] = scaler.fit_transform(df[['TotalCharges', 'tenure', 'MonthlyCharges']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Save to CSVs
X_train.to_csv("_data/cleaned_X_train.csv", index=False)
X_test.to_csv("_data/cleaned_X_test.csv", index=False)
y_train.to_csv("_data/cleaned_y_train.csv", index=False)
y_test.to_csv("_data/cleaned_y_test.csv", index=False)