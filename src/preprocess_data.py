#Import necessary libraries
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#Read dataset and attach it to the dataframe
df = pd.read_csv("Telco-Customer-Churn.csv")
#Print a part of the dataframe to get an idea of the data
print(df.head())

#Check if there are any blank or null values in the dataset
print(df.isnull().sum())
print(df.describe())
import seaborn as sns
import matplotlib.pyplot as plt
print(df.dtypes)

#Convert TotalCharges to type numeric since we want a boxplot for it
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Check to see if there are missing values
if df.isnull().values.any():
    print("There are missing values in the dataset.")
else:
    print("No missing values found!")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns   
    #print numeric columns
    print(f"Numeric columns detected: {list(numeric_cols)}") 

    #Print rows with missing data (We had 11 rows and will drop them in the following indentation)
print(df[df.isnull().any(axis=1)])

#Drop all rows with missing data
df.dropna(inplace=True)

#Drop customerID column since it is only an identifier and not important to the analytics of the dataset
df.drop('customerID', axis=1, inplace=True)

#Print data types of each row to try and understand what needs to be converted so that we can work with it
print(df.dtypes)

# Check which columns are categorical
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", categorical_cols)

#Apply factorization on all categorical columns. (All columns with type of object are mapped to become numerical)
for col in df.select_dtypes(include='object').columns:
    df[col], mapping = pd.factorize(df[col])
    print(f"{col} mapping: {dict(enumerate(mapping))}")

for col in df.columns:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    #Check if mapping worked
print(df.info())
print(df.isnull().sum())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Contract', 'tenure', 'InternetService']] = scaler.fit_transform(df[['Contract', 'tenure', 'InternetService']]) #rescales for better performance

#Make Churn the target column
X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(df['Churn'].value_counts(normalize=True))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))

importances = rf_model.feature_importances_
feature_names = X_train.columns
# Plot
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', figsize=(10,6), title='Feature Importances')
plt.tight_layout()
plt.savefig("FeautureImportance.png")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

importances = rf_model.feature_importances_
feature_names = X_train.columns

# Convert to Series and sort
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

# Better visual with seaborn
plt.figure(figsize=(10, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("FeatureImportance.png")
plt.show()

numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlations = numeric_df.corr()['Churn'].sort_values(ascending=False)
print(correlations)

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Create the model
xgb_model = XGBClassifier(eval_metric='logloss')  # suppress warning

# Train the model
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_xgb))

# Accuracy
accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.show()
