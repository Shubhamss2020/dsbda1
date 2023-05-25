import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

df = pd.read_csv('/media/estanghost/XD/sublimes/a9/Titanic.csv' )
print(df)

sns.countplot(df['Survived'])
plt.show()

df['Sex'].value_counts().plot(kind='pie', autopct="%.2f")
plt.show()

# plt.hist(date=df, df['Age'], bins=5)
# plt.show()

# sns.displot(date=df, df['Age'])
# plt.show()

sns.scatterplot(data=df, x='Fare', y='Age')

plt.show()

sns.scatterplot(data=df, x=df['Fare'], y=df['Age'], hue=df["Sex"])

plt.show()

sns.scatterplot(data=df, x=df['Fare'], y=df['Age'], hue=df["Sex"], style=df['Survived'])
# plt.show()

sns.barplot(data=df, x=df['Pclass'], y=df['Age'])
plt.show()

sns.barplot(data=df, x=df['Pclass'], y=df['Age'], hue=df['Sex'])
plt.show()	

sns.boxplot(data=df, x=df['Sex'], y=df['Age'])
plt.show()

sns.boxplot(data=df, x=df['Sex'], y=df['Age'], hue=df["Survived"])
plt.show()

sns.displot(df[df['Survived'] == 0]['Age'], hist=False, color="blue")
sns.displot(df,df[df['Survived'] == 1]['Age'], hist=False, color="orange")
plt.show()


a=pd.crosstab(df['Pclass'], df['Survived'])
print(a)
pd.crosstab(df['Pclass'], df['Survived'])
sns.heatmap(pd.crosstab(df['Pclass'], df['Survived']))
plt.show()

sns.clustermap(pd.crosstab(df['Parch'], df['Survived']))
plt.show()