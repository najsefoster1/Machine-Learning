#CIS625 - Machine Learning For Business Assignment
#Obesity and Physical Inactivity Analysis
#Author: Najse Foster
#Dataset Source: https://www.kaggle.com/datasets/spittman1248/cdc-data-nutrition-physical-activity-obesity

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load dataset
df = pd.read_csv('Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')

#Filter to only include total population entries
df_total = df[df['StratificationCategory1'] == 'Total']

#Focus on only relevant questions
relevant_questions = {
    'Percent of adults aged 18 years and older who have obesity': 'obesity_rate',
    'Percent of adults who engage in no leisure-time physical activity': 'physical_inactivity'
}

df_filtered = df_total[df_total['Question'].isin(relevant_questions.keys())]
df_filtered.loc[:, 'variable'] = df_filtered['Question'].map(relevant_questions)

#Pivot data
df_pivot = df_filtered.pivot_table(index=['LocationDesc', 'YearStart'],
                                   columns='variable',
                                   values='Data_Value').reset_index()

#Prepare model dataset
df_model = df_pivot.dropna()
X = df_model[['physical_inactivity']]
y = df_model['obesity_rate']

#1. Linear Regression
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()
print('--- Linear Regression Summary ---')
print(ols_model.summary())

#2. Lasso Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)
print('\n--- Lasso Coefficient ---')
print(dict(zip(X.columns, lasso.coef_)))

# 3. Quantile Regression (75th Percentile)
quant_model = smf.quantreg('obesity_rate ~ physical_inactivity', df_model)
quant_fit = quant_model.fit(q=0.75, max_iter=5000)
print('\n--- Quantile Regression Summary (75th percentile) ---')
print(quant_fit.summary())


#4. Logistic Regression
median_obesity = df_model['obesity_rate'].median()
df_model.loc[:, 'obesity_class'] = (df_model['obesity_rate'] > median_obesity).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df_model['obesity_class'], test_size=0.3, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print('--- Logistic Regression Coefficient ---')
for name, coef in zip(X.columns, log_reg.coef_[0]):
    print(f"{name}: {coef:.4f}")

