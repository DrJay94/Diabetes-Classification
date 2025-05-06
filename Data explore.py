import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load the dataset ===
file_path = r"C:\Users\dylan\Desktop\Data Science Final\diabetes_prediction_dataset.csv"
df = pd.read_csv(file_path)

# === Basic information ===
print("=== Basic Info ===")
print(df.info())
print("\n=== First 5 Rows ===")
print(df.head())

# === Summary statistics ===
print("\n=== Summary Statistics ===")
print(df.describe(include='all'))

# === Missing values ===
print("\n=== Missing Values ===")
print(df.isnull().sum())

# === Unique values per column ===
print("\n=== Unique Values per Column ===")
print(df.nunique())

# === Class balance (assumes target column is named 'diabetes' or similar) ===
if 'diabetes' in df.columns:
    print("\n=== Target Class Balance ===")
    print(df['diabetes'].value_counts(normalize=True))
    sns.countplot(x='diabetes', data=df)
    plt.title("Target Class Distribution")
    plt.show()

# === Correlation Matrix for numeric features (safe version) ===
numeric_df = df.select_dtypes(include=['int64', 'float64'])
numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]  # Remove constant cols

corr_matrix = numeric_df.corr().dropna(axis=1, how='all').dropna(axis=0, how='all')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm')  # Remove annot=True for speed
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# === Distributions for numerical columns ===
for column in numeric_df.columns:
    plt.figure()
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.show()

# === Boxplots to detect outliers ===
for column in numeric_df.columns:
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

# === Pairwise relationships ===
if numeric_df.shape[1] <= 10:  # Prevent overload
    sns.pairplot(df[numeric_df.columns])
    plt.show()

# === Value counts and target relationships for categorical features ===
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n=== Value Counts for {col} ===")
    print(df[col].value_counts())
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()
    
    if 'diabetes' in df.columns:
        plt.figure()
        sns.countplot(x=col, hue='diabetes', data=df)
        plt.title(f"{col} vs Diabetes")
        plt.xticks(rotation=45)
        plt.show()

#Distribution of Categorical Features
for col in ['gender', 'smoking_history']:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Boxplots of Numeric Features by Diabetes
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='diabetes', y=col, data=df)
    plt.title(f"{col} by Diabetes")
    plt.tight_layout()
    plt.show()

#Group Statistics by Diabetes
grouped = df.groupby('diabetes')[numerical_cols].mean()
print("\n=== Mean Values by Diabetes Status ===")
print(grouped)

#Violin Plots for Skewed Distributions
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='diabetes', y=col, data=df)
    plt.title(f"{col} Distribution by Diabetes")
    plt.tight_layout()
    plt.show()

#Chi-Square Test for Categorical Variables vs Target
from scipy.stats import chi2_contingency

for col in ['gender', 'smoking_history']:
    contingency = pd.crosstab(df[col], df['diabetes'])
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"\nChi-Square test for {col} vs Diabetes:")
    print(f"Chi2 = {chi2:.2f}, p-value = {p:.5f}")
