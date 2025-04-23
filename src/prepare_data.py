import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

#Load dataset and get a summary of the info
df = pd.read_csv("Telco-Customer-Churn.csv")
df.info()

#Checking to see if there are any missing values in the dataset
df.isnull().sum()

#Univariate Data Analyis
#Starting with Demographics

#Gender Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='gender')
plt.title('Gender Distribution')
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('../gender_distribution.png')
plt.show()

#Distribution of Senior Customers
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='SeniorCitizen')
plt.title('Distribution of Senior Customers')
plt.xlabel("Senior Citizen")
plt.ylabel("Frequency")
plt.xticks([0, 1], ['False', 'True'])
plt.tight_layout()
plt.savefig('../senior_distribution.png')
plt.show()

#Distribution of Customers with Partners
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Partner')
plt.title('Distribution of Customers with Partners')
plt.xlabel("Partners")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig('../partner_distribution.png')
plt.show()

#Distribution of Customers with Dependents
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Dependents', palette='flare')
plt.title('Distribution of Customers with Dependents')
plt.xlabel("Dependent")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig('../dependent_distribution.png')
plt.show()

#Churn Distribution
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Churn Distribution')
plt.savefig("../churn_distribution")
plt.show()

#Bivaritate Analysis
#Group the services and compare their Relationship with the Churn Rate
services_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']

plt.figure(figsize=(20, 20))
for i, col in enumerate(services_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=df, x=col, hue='Churn', palette='Set2')
    plt.title(f'{col} vs Churn')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../services_subscribed_vs_churn.png')
plt.show()

sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Churn Distribution')
plt.savefig("../churn_distribution")
plt.show()



num_cols = ['tenure', 'MonthlyCharges']

# Histograms
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# Boxplots vs Churn
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Churn', y=col, data=df, palette='coolwarm')
    plt.title(f'{col} by Churn')
plt.tight_layout()
plt.show()


# InternetService
internet_features = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

for feature in internet_features:
    g = sns.catplot(
        data=df_internet,
        x=feature,
        hue="InternetService",
        col="Churn",
        kind="count",
        height=4,
        aspect=1
    )
    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle(f"Churn by InternetService and {feature}")
    plt.savefig(f"../Churn by InternetService and {feature}")
    plt.show()

#Convert TotalCharges to type numeric since we want a boxplot for it
#df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')



#Phone Service
#df_phone_only = df[(df['PhoneService'] == 'Yes') & (df['InternetService'] == 'No')]

#df['HasInternetAndPhone'] = ((df['InternetService'] != 'No') & (df['PhoneService'] == 'Yes')).astype(int)
