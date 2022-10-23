import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

data = load_diabetes()
dataFrameData = pd.DataFrame(data.data, columns=data.feature_names)
correlationArray = dataFrameData.corr()
columns = list(dataFrameData.columns)


def testModelDiabetes(repetitions):
    x = dataFrameData.values
    y = data.target
    s = 0
    for i in range(repetitions):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(x_train, y_train)
        y_pred = linReg.predict(x_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        s += mape
    return s / repetitions

print(testModelDiabetes(100))


