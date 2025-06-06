import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("_data/Telco-Customer-Churn.csv")

# Load cleaned data
X_train = pd.read_csv("cleaned_X_train.csv")
X_test = pd.read_csv("cleaned_X_test.csv")
y_train = pd.read_csv("cleaned_y_train.csv").squeeze().map({"No": 0, "Yes": 1})
y_test = pd.read_csv("cleaned_y_test.csv").squeeze().map({"No": 0, "Yes": 1})

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


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=3000)
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

from sklearn.metrics import classification_report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(df['Churn'].value_counts(normalize=True))

from sklearn.metrics import confusion_matrix


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
# Feature Importance Plot
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', figsize=(10,6), title='Feature Importances')
plt.tight_layout()
plt.savefig("FeautureImportance.png")
plt.show()

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

#XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


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

import joblib
joblib.dump(rf_model, 'artifacts_random_forest_model.pkl')
joblib.dump(xgb_model, 'artifacts_xgb_model.pkl')
joblib.dump(logistic_model, 'artifacts_logisticalregression_model.pkl')