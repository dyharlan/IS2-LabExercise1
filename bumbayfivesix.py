import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
### Loading the Dataset
data = pd.read_csv("House_Rent_Dataset.csv")
### Engineer a new Feature. In this case, Rent/Sq. Ft.
area = data['Size']
rent = data['Rent']
print(area.size == rent.size)
rent_per_sq_ft = []
x = 0
while x < rent.size:
    rent_per_sq_ft.append(rent.iloc[x]/area.iloc[x])
    x+=1
### Append the new feature as a new column
data['Rent per Square Feet'] = rent_per_sq_ft
### Pre-processing
#data = data[['Size', 'Rent per Square Feet','Area Locality', 'Area Type', 'City', 'Tenant Preferred', 'Rent']]
data = data[['Size', 'Rent per Square Feet', 'Area Type', 'City', 'Tenant Preferred', 'Rent']]
def one_hot_encode(data, column):
    encoded = pd.get_dummies(data[column], drop_first= True)
    data = data.drop(column, axis = 1)
    data = data.join(encoded)
    return data
def target_encode(data, column, column_label, target):
    target_mean = data.groupby(column)[target].mean()
    data[column_label] = data[column].map(target_mean)
    data = data.drop(column, axis = 1)
    return data

#data = one_hot_encode(data, 'Furnishing Status')
#data = target_encode(data, 'Area Locality', 'Area Locality Target', 'Rent')
data = one_hot_encode(data, 'Area Type')
data = one_hot_encode(data, 'Tenant Preferred')
data = one_hot_encode(data, 'City')
## Remove Rent from the X axis and put into the Y-axis
X = data.drop('Rent', axis= 1)
y = data['Rent']
## Split the dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)
## Standardize the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
## Fitting via gradient descent
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print("Results of the Gradient Descent: ",model.coef_)

### Quantitative Evaluation
y_preds = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
# The coefficients
print("Coefficients: \n", model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_preds))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_preds))

### Qualitative Evaluation

#sample_index = 456 #Negative Value?
sample_index = 384
sample_data = X.iloc[sample_index]
print("Sample Data: \n", sample_data)
print("\nRent: ", y.iloc[sample_index])
sample_data_standardized = sc.transform(X.iloc[sample_index].values.reshape(1,-1))
model_rent_forecast = model.predict(sample_data_standardized)[0]
print("Forecasted Rent: ",model_rent_forecast)
print("Actual Rent: ", y.iloc[sample_index])