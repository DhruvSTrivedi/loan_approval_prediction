import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Preprocessing
df['Movable_assets'] = df[' bank_asset_value'] + df[' luxury_assets_value']
df['Immovable_assets'] = df[' residential_assets_value'] + df[' commercial_assets_value']
df.drop(columns=['loan_id', ' bank_asset_value', ' luxury_assets_value', 
                 ' residential_assets_value', ' commercial_assets_value'], inplace=True)
df['Loan_to_Income_Ratio'] = df[' loan_amount'] / df[' income_annum']
df['Dependents_Adjusted_Income'] = df[' income_annum'] / (1 + df[' no_of_dependents'])
df['CIBIL_Loan_Term_Interaction'] = df[' cibil_score'] * df[' loan_term']
df[' education'] = df[' education'].map({' Not Graduate': 0, ' Graduate': 1})
df[' self_employed'] = df[' self_employed'].map({' No': 0, ' Yes': 1})
df[' loan_status'] = df[' loan_status'].map({' Rejected': 0, ' Approved': 1})

# Features and target
X = df.drop(' loan_status', axis=1)
y = df[' loan_status']

# Augmentation 1: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Augmentation 2: Add Gaussian Noise
def add_noise(data, noise_level=0.01):
    noisy_data = data.copy()
    for col in data.select_dtypes(include=np.number).columns:
        noisy_data[col] += noise_level * np.random.randn(len(data))
    return noisy_data

X_augmented = add_noise(X_resampled)

# Combine original and augmented datasets
X_combined = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(X_augmented)], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_resampled, y_resampled], axis=0).reset_index(drop=True)

# Cross-Validation with Augmented Data
params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, test_index in kf.split(X_combined):
    X_train, X_test = X_combined.iloc[train_index], X_combined.iloc[test_index]
    y_train, y_test = y_combined.iloc[train_index], y_combined.iloc[test_index]

    xgb_model = XGBClassifier(**params, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    fold_accuracies.append(accuracy_score(y_test, y_pred))

# Cross-Validation Results
mean_accuracy = np.mean(fold_accuracies)
print(f"Mean CV Accuracy (Augmented): {mean_accuracy:.4f}")

# Train Final Model
final_xgb = XGBClassifier(**params, eval_metric='logloss')
final_xgb.fit(X_combined, y_combined)

# Final Model Predictions
y_pred = final_xgb.predict(X_combined)

# Metrics
final_r2 = r2_score(y_combined, y_pred)
final_mse = mean_squared_error(y_combined, y_pred)
final_mae = mean_absolute_error(y_combined, y_pred)

print("\nFinal Model Performance:")
print(f"R2: {final_r2:.4f}")
print(f"MSE: {final_mse:.4f}")
print(f"MAE: {final_mae:.4f}")

# Feature Importance Analysis

## XGBoost Built-In Plot
plt.figure(figsize=(10, 6))
plot_importance(final_xgb, importance_type='weight', max_num_features=10, title="Feature Importance (Weight)")
plt.show()

plot_importance(final_xgb, importance_type='gain', max_num_features=10, title="Feature Importance (Gain)")
plt.show()

## SHAP Analysis
explainer = shap.Explainer(final_xgb, X_combined)
shap_values = explainer(X_combined)

# Global Feature Importance Summary Plot
shap.summary_plot(shap_values, X_combined)

# Bar Plot for Global Feature Importance
shap.summary_plot(shap_values, X_combined, plot_type="bar")
