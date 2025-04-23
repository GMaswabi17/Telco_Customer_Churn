#Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Telco-Customer-Churn.csv")

#Convert TotalCharges to type numeric since we want a boxplot for it
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Drop all rows with missing data
df.dropna(inplace=True)

#Drop customerID column since it is only an identifier and not important to the analytics of the dataset
df.drop('customerID', axis=1, inplace=True)

## Feature Engineering
# Total Services Used
service_cols = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies'
]
df['TotalServicesUsed'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

# Has Internet and Phone Combo
df['HasInternetPhoneCombo'] = ((df['InternetService'] != 'No') & (df['PhoneService'] == 'Yes')).astype(int)

# Tenure Group
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

# One-hot encode all categorical variables
df = pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['TotalCharges', 'tenure', 'MonthlyCharges']] = scaler.fit_transform(df[['TotalCharges', 'tenure', 'MonthlyCharges']]) #rescales for better performance

#Make Churn the target column
X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv("cleaned_X_train.csv", index=False)
X_test.to_csv("cleaned_X_test.csv", index=False)
y_train.to_csv("cleaned_y_train.csv", index=False)
y_test.to_csv("cleaned_y_test.csv", index=False)



