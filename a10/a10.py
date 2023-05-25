import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("IRIS.csv")
print(df)

df = pd.read_csv("IRIS.csv", header=1)
column_name = ['sepal_length','sepal_width','petal_length','petal_width','species']

df.columns = column_name

print(column_name)
print(df.head())

sns.histplot(data = df, x='sepal_length',hue='species')
sns.boxplot(x='species',y='petal_length',data=df)
plt.show()