import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weatherIn = pd.read_csv('weatherCSV.csv')
weatherIn.dropna(inplace = True)
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

X = weatherIn[['MaxTmp', 'MinTmp', 'Depart', 'HDD', 'CDD']]
y = weatherIn['AvgTmp']