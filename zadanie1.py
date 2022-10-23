import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("practice_lab_2.xlsx")
correlationArray = file.corr()

x = file.iloc[:, :file.shape[1]-1]
y = file.iloc[:, -1]

fig, ax = plt.subplots(x.shape[1], 1, figsize=(10, 10))
for i, col in enumerate(x.columns):
    ax[i].scatter(x[col], y)
plt.show()
