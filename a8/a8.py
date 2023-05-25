import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/media/estanghost/XD/sublimes/a8/Titanic.csv')
print(df)

sns.histplot(data=df, x="Fare", binwidth=3)
plt.show()

sns.histplot(data=df, x="Age", kde=True)
plt.show()

sns.histplot(data=df, x="Age", hue="Survived")
plt.show()

sns.histplot(data=df, x="Sex", kde=True)
plt.show()

sns.distplot(df['Age'], kde=False, bins=10)
plt.show()

sns.distplot(df['Fare'], kde=False, bins=10)
plt.show()

sns.scatterplot(data=df, x='Fare', y='Age')
plt.show()

sns.kdeplot(data=df, x='Fare')
plt.show()
sns.kdeplot(data=df, x='Age')
plt.show()

sns.stripplot(data=df, x='Fare', y='Age')
plt.show()

sns.catplot(kind='bar',data=df,x='Fare',y='Age',aspect=2)
plt.show()