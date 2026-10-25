import pandas as pd
import numpy as np
from collections import defaultdict
import statsmodels.api as sm

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.head()

data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data = data.drop(columns = ['id', 'work_type'], errors='ignore')
num_cols = data.select_dtypes(include = ['number']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
data = data[data['gender'] != 'Other']
data['gender'] = data['gender'].replace({'Male': 1, 'Female': 0}) #Gender: Male = 1; Female = 0
data['ever_married'] = data['ever_married'].replace({'Yes': 1, 'No': 0}) #Ever married? Yes = 1; No = 0
data['Residence_type'] = data['Residence_type'].replace({'Urban': 1, 'Rural': 0}) #Residence type Urban = 1; Rural = 0
data = data[data['smoking_status'] != 'Unknown']
data.head()

data = pd.get_dummies(data, columns = ['smoking_status'], drop_first = True, dtype = int) #One hot encoding for smokes and never smoked
data.head()

#Scale the continuous covariates to follow ~ N(0,1) (i.e. standardize)
data['age'] = ((data['age'] - data['age'].mean())/data['age'].std())
data['avg_glucose_level'] = ((data['avg_glucose_level'] - data['avg_glucose_level'].mean())/data['avg_glucose_level'].std())
data['bmi'] = ((data['bmi'] - data['bmi'].mean())/data['bmi'].std())
data.head()

y = data['stroke']
Y = y.to_numpy

X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status_never smoked', 'smoking_status_smokes']]
X = sm.add_constant(X)

#Let's try OLS regression, just for the sake of it, how good a fit does it give?
X = X.astype(float)
model_OLS = sm.OLS(y,X).fit()
print(model_OLS.summary())
