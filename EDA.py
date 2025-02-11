import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Strip spaces from categorical columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert categorical variables to proper types
df["loan_status"] = df["loan_status"].astype(str)
df["education"] = df["education"].astype(str)
df["self_employed"] = df["self_employed"].astype(str)

# Create a folder to save images
output_folder = "EDA_Visuals"
os.makedirs(output_folder, exist_ok=True)

# 1. Distribution of Loan Amount
plt.figure(figsize=(8,5))
sns.histplot(df["loan_amount"], bins=30, kde=True)
plt.title("Distribution of Loan Amount")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.savefig(f"{output_folder}/loan_amount_distribution.png")

# 2. Count of Loan Status (Approved vs. Rejected)
plt.figure(figsize=(6,5))
sns.countplot(x=df["loan_status"], hue=df["loan_status"], palette="coolwarm", legend=False)
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.savefig(f"{output_folder}/loan_status_count.png")

# 3. Income vs. Loan Amount (Scatter Plot)
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["income_annum"], y=df["loan_amount"], hue=df["loan_status"])
plt.title("Income vs Loan Amount")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.savefig(f"{output_folder}/income_vs_loan.png")

# 4. CIBIL Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["cibil_score"], bins=20, kde=True, color="purple")
plt.title("CIBIL Score Distribution")
plt.xlabel("CIBIL Score")
plt.ylabel("Frequency")
plt.savefig(f"{output_folder}/cibil_score_distribution.png")

# 5. Loan Term Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["loan_term"], bins=15, kde=True, color="green")
plt.title("Loan Term Distribution")
plt.xlabel("Loan Term (Months)")
plt.ylabel("Frequency")
plt.savefig(f"{output_folder}/loan_term_distribution.png")

# 6. Loan Status by Education Level
plt.figure(figsize=(7,5))
sns.countplot(x=df["education"], hue=df["loan_status"], palette="Set2")
plt.title("Loan Status by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.savefig(f"{output_folder}/loan_status_by_education.png")

# 7. Self-Employed vs. Loan Approval Rate
plt.figure(figsize=(6,5))
sns.countplot(x=df["self_employed"], hue=df["loan_status"], palette="coolwarm")
plt.title("Self-Employed vs Loan Approval Rate")
plt.xlabel("Self-Employed")
plt.ylabel("Count")
plt.savefig(f"{output_folder}/self_employed_vs_loan.png")

# 8. Loan Amount by Loan Status (Boxplot)
plt.figure(figsize=(7,5))
sns.boxplot(x=df["loan_status"], y=df["loan_amount"], palette="pastel")
plt.title("Loan Amount by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.savefig(f"{output_folder}/loan_amount_by_status.png")

# 9. Pairplot for Key Numerical Features
sns.pairplot(df[["income_annum", "loan_amount", "cibil_score", "loan_term", "no_of_dependents"]], diag_kind="kde")
plt.savefig(f"{output_folder}/pairplot.png")

# 10. Heatmap of Correlation Matrix (Only Numeric Columns)
numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(10,7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{output_folder}/correlation_heatmap.png")

print(f"EDA visuals saved in '{output_folder}' folder.")
