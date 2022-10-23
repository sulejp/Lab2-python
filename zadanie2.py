import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("practice_lab_2.xlsx")
correlationArray = file.corr()

x = file.iloc[:, :file.shape[1] - 1]
y = file.iloc[:, -1]

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def testModel(repetitions):
    s = 0
    for i in range(repetitions):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(x_train, y_train)
        y_pred = linReg.predict(x_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        s += mape
    return s / repetitions

print(testModel(100))
