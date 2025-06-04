#Najse Foster
#CIS625 - Machine Learning for Business
#Professor JC Martel
#April 16, 2025

#Required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay

# Set MacBook path to save plots
Desktop_path = os.path.expanduser("~/Desktop")

#Load dataset
dataset_path = "Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv"

if not os.path.exists(dataset_path):
    print("[ERROR]: Dataset not found! Please ensure the CSV file is in the correct directory.")
    exit()

print("[INFO]: Loading dataset...")
df = pd.read_csv(dataset_path)
df = df[df['Data_Value'].notnull()]
print("[INFO]: Dataset loaded successfully!")

#Select relevant columns
columns = ['YearStart', 'LocationAbbr', 'Class', 'Topic', 'Question',
           'StratificationCategory1', 'Stratification1', 'Data_Value']
df = df[columns].dropna()
print("[INFO]: Columns filtered successfully!")

#Encode categorical columns
print("[INFO]: Encoding categorical columns...")
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("[INFO]: Categorical encoding completed!")

# Define features and target
X = df.drop('Data_Value', axis=1)
y = df['Data_Value']
feature_names = [
    "Start Year",
    "State Abbreviation",
    "Health Category",
    "Health Topic",
    "Survey Question",
    "Stratification Type",
    "Stratification Detail"
]

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("[INFO]: Data split completed!")

#Train models
print("[INFO]: Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=30, random_state=42)
rf_model.fit(X_train, y_train)
print("[INFO]: Random Forest training completed!")

print("[INFO]: Training XGBoost model...")
xgb_model = XGBRegressor(n_estimators=30, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
print("[INFO]: XGBoost training completed!")

#SHAP analysis using TreeExplainer
print("[INFO]: Computing SHAP values...")
X_sample = X_test.sample(n=100, random_state=42)

explainer_rf = shap.Explainer(rf_model)
shap_values_rf = explainer_rf(X_sample)
shap_values_rf.feature_names = feature_names

explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_sample)
shap_values_xgb.feature_names = feature_names

#SHAP Summary Plot - Random Forest
print("[INFO]: Generating SHAP Summary Plot - Random Forest...")
plt.figure()
shap.plots.bar(shap_values_rf, max_display=7, show=False)
plt.title("Top Predictors Impacting Health Outcomes (Random Forest)")
plt.xlabel("Average Contribution to Prediction")
plt.ylabel("Feature Description")
plt.savefig(os.path.join(Desktop_path, "shap_summary_random_forest.png"))
plt.show()

#SHAP Summary Plot - XGBoost
print("[INFO]: Generating SHAP Summary Plot - XGBoost...")
plt.figure()
shap.plots.bar(shap_values_xgb, max_display=7, show=False)
plt.title("Most Influential Factors in XGBoost Model")
plt.xlabel("Average Contribution to Prediction")
plt.ylabel("Feature Description")
plt.savefig(os.path.join(Desktop_path, "shap_summary_xgboost.png"))
plt.show()

#Partial Dependence Plots for 'Stratification1'
feature = 'Stratification1'

#Random Forest
print("[INFO]: Generating Partial Dependence Plot - Random Forest...")
fig, ax = plt.subplots()
PartialDependenceDisplay.from_estimator(rf_model, X_test, [feature], ax=ax)
plt.title("How Stratification Detail Affects Prediction (Random Forest)")
plt.xlabel("Stratification Detail (Encoded)")
plt.ylabel("Predicted Health Metric")
plt.savefig(os.path.join(Desktop_path, "pdp_random_forest_stratification1.png"))
plt.show()

#PDP - XGBoost
print("[INFO]: Generating Partial Dependence Plot - XGBoost...")
fig, ax = plt.subplots()
PartialDependenceDisplay.from_estimator(xgb_model, X_test, [feature], ax=ax)
plt.title("Stratification Detail's Impact in XGBoost")
plt.xlabel("Stratification Detail (Encoded)")
plt.ylabel("Predicted Health Metric")
plt.savefig(os.path.join(Desktop_path, "pdp_xgboost_stratification1.png"))
plt.show()

print("[SUCCESS]: Script execution completed! Check your plots on the MacBook.")
