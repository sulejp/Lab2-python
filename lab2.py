import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("practice_lab_2.xlsx")
correlationArray = file.corr()
x = file.iloc[:, :file.shape[1] - 1]
y = file.iloc[:, -1]

"""
Zadanie 2.1 Pierwsze spojrzenie
Pobierz plik „housing.xlsx” ze strony kursu.
Podobnie do zadań z poprzednich zajęć (Zadanie 1.4) wygeneruj macierz korelacji dla wczytanego zbioru.
Przeanalizuj macierz korelacji. Jakie zależności mogą mieć związek, a jakie są przypadkowe?
Wygeneruj wykresy korelacji pomiędzy cechami niezależnymi a cecha zależną (medianową ceną mieszkania).
"""

def zadanie1():
    fig, ax = plt.subplots(x.shape[1], 1, figsize=(10, 10))
    for i, col in enumerate(x.columns):
        ax[i].scatter(x[col], y)
    plt.show()


"""
Zadanie 2.2.Wielokrotne testowanie modelu
Zmodyfikuj skrypt, który pokazuje Listing 2.4, tak,
żeby wyszedł skrypt pozwalający na wielokrotne przetestowanie modelu regresji liniowej.
Skrypt umieść w oddzielnej funkcji, która jako argument przyjmie liczbę powtórzeń, które trzeba wykonać.
Podpowiedź: w funkcji zastosuj pętlę for.
Za każdym razem zbiór danych ma zostać podzielony na podzbiory: uczący oraz testowy w sposób losowy,
w tym celu nie podawaj argumentu random_state, aby wyniki za każdym razem się różniły.
Jako wynik eksperymentu ma być zwrócona średnia wartość miary mean_absolute_percentage_error – czyli średni procent błędu regresji.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def zadanie2():
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

"""
Zadanie 2.3. Uwzględnienie wartości odstających
Wykonaj Zadanie 2.2 dodając do niego procedurę usuwania/zastępowania wartości odstających.
Porównaj wyniki uzyskane w poprzednim zadaniu z nowymi wynikami.
"""

def zadanie3():
    def deleteOutliners(n):
        x = file.iloc[:, :file.shape[1] - 1]
        y = file.iloc[:, -1]
        s = 0
        for i in range(n):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
            outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
            x_train_no_outliers = x_train.loc[~outliers, :]
            y_train_no_outliers = y_train.loc[~outliers]
            linReg = LinearRegression()
            linReg.fit(x_train_no_outliers, y_train_no_outliers)
            y_pred = linReg.predict(x_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            s += mape
        return s / n

    def replaceOutliners(n):
        x = file.iloc[:, :file.shape[1] - 1]
        y = file.iloc[:, -1]
        s = 0
        for i in range(n):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
            outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
            x_train_no_outliers = x_train.loc[~outliers, :]
            y_train_no_outliers = y_train.loc[~outliers]
            y_train_mean = y_train.copy()
            y_train_mean[outliers] = y_train.mean()
            linReg = LinearRegression()
            linReg.fit(x_train_no_outliers, y_train_no_outliers)
            linReg.fit(x_train, y_train_mean)
            y_pred = linReg.predict(x_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            s += mape
        return s / n
    print(deleteOutliners(3))
    print(replaceOutliners(3))

"""
Zadanie 2.5. Samodzielne badanie danych.
Za pomocą kodu, który przedstawia Listing 2.9,
załaduj zbiór danych Diabetes. Przeanalizuj go podobnie do tego,
jak zrobiliśmy ze zbiorem danych „Boston Housing”.
"""

from sklearn.datasets import load_diabetes
def zadanie5():
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