import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weatherIn = pd.read_csv('weatherCSV.csv')
weatherIn['Precip'].replace('T', np.nan, inplace=True)
weatherIn['Precip'] = pd.to_numeric(weatherIn['Precip'], errors='coerce')
mean_precip = weatherIn['Precip'].mean()
weatherIn['Precip'].fillna(mean_precip, inplace=True)
# Create data from only 2018
weatherIn['Date'] = pd.to_datetime(weatherIn['Date'])
weatherIn2018 = weatherIn[weatherIn['Date'].dt.year == 2018]


print(weatherIn.head())

datCols = ['MaxTmp', 'MinTmp', 'AvgTmp', 'Precip']

for cols in datCols:
    plt.figure(figsize = (8, 8))
    if cols == 'Precip':
        sns.histplot(weatherIn[cols], kde = True, color = 'skyblue')
        plt.xticks(rotation = 70)
    else:
        sns.histplot(weatherIn[cols], kde = True, color = 'skyblue')
    plt.xlabel(cols)
    plt.title("Frequency For: " + cols)
    plt.legend()
    plt.show()

plt.tight_layout()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = weatherIn[['MaxTmp', 'MinTmp', 'Depart', 'HDD', 'CDD', 'Precip']]
y = weatherIn['AvgTmp']

# Creating the train and test sets and starting the model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.165, random_state = 42, shuffle = True)
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(y_pred)
print(y_pred.size)
print("Mean Squared Error:", mse)