import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ds1 = pd.read_csv("dataset.csv")
print('Null values before updating - ')
print(ds1.isnull().sum())
ds1.fillna(ds1.mean())
print('Null values After updating - ')
print(ds1.isnull().sum())

#Boxplot
sns.boxplot(data = ds1, x = ds1["Discussion"], y = ds1["NationalITy"])
sns.boxplot(data = ds1, x = ds1['Discussion'], y = ds1["NationalITy"], hue = ds1['gender'])
plt.show()

#Scatterplot
sns.scatterplot(data=ds1, x="raisedhands", y="VisITedResources")
plt.show()

#Heatmap
print(ds1.corr())
sns.heatmap(ds1.corr(), annot = True)
plt.show()

#Print Outliers using z-score
lower_limit = ds1["raisedhands"].mean() - 3 * ds1["raisedhands"].std()
upper_limit = ds1["raisedhands"].mean() + 3 * ds1["raisedhands"].std()
ds1 = [(ds1["raisedhands"] >= lower_limit) & (ds1['raisedhands'] <= upper_limit)]
print(ds1)
def outliers_z_score(ys):
    threshold = 3*ys - np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y-np.mean(ys)) / stdev_y for y in ys]
    return np.where(np.abs(z_scores)>0)
    outliers_z_score(ds1['raisedhands'])

# Print outliers using modified z-score 
mean = np.mean(ds1['raisedhands'])
std = np.std(ds1['raisedhands'])
print("The mean of dataset is ", mean)
print("The standard deviation of dataset is ", std)
minthreshold = -4
maxthreshold = 4
outlier = []
for i in ds1['raisedhands']:
    z = (i - mean) / std
    if z > maxthreshold or z < minthreshold:
        outlier.append(i)
if outlier:
    print("The outliers in dataset are ", outlier)
else:
    print("There are no outliers in the dataset.")

#Print outliers using IQR(Interquartile Range)
q1 = np.percentile(ds1['raisedhands'], 25)
q3 = np.percentile(ds1['raisedhands'], 75)
IQR = q3 - q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
outlier = []
for i in ds1['raisedhands']:
    if i<lower_bound or i>upper_bound:
        outlier.append(i)
if outlier:
    print("The outliers in dataset are ", outlier)
else:
    print("There are no outliers in the dataset.")