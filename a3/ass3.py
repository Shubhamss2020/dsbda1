import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Employee_Salary.csv")
print(df)
print("Information in dataset - ")
print(df.info())
print("size of the dataset - \n", df.shape)
print("total number of variables", df.size)
print("Columns are - ", df.columns)
print("datatypes are - \n", df.dtypes)
print("First 5 rows - \n", df.head(5).T)
print("Last 5 rows in the dataset - \n", df.tail(5).T)
print("Any 5 rows in dataset - \n", df.sample(5).T)
print("Null values are - \n", df.isna().sum())

print("statistical information of numerical columns")
columns = ['Experience_Years', 'Age', 'Salary']
print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("Columns","Min","Max","Mean","Median","Std"))
for i in columns:
    m1 = df[i].min() 
    m2 = df[i].max()
    m3 = df[i].mean()
    m4 = df[i].median()
    m5 = df[i].std()
    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format(i,m1,m2,m3,m4,m5))
    
print("Groupwise statistical information of numerical columns")
columns = ['Experience_Years', 'Age', 'Salary']

print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("Columns","Min","Max","Mean","Median","Std"))
for i in columns:
    print("\n --------------------",i,"--------------------------")
    m1 = df[i].groupby(df['Gender']).min() 
    m2 = df[i].groupby(df['Gender']).max()
    m3 = df[i].groupby(df['Gender']).mean()
    m4 = df[i].groupby(df['Gender']).median()
    m5 = df[i].groupby(df['Gender']).std()
    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Female',m1[0],m2[0],m3[0],m4[0],m5[0]))
    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Male',m1[1],m2[1],m3[1],m4[1],m5[1]))
    
#groupwise barplotS
X = ['min', 'max', 'mean', 'median', 'std']
features = ['Salary', 'Age', 'Experience_Years']
df1 = pd.DataFrame(columns=['min','max','mean','median','std'])

for var in features:
    df1['min'] = df[var].groupby(df['Gender']).min()
    df1['max'] = df[var].groupby(df['Gender']).max()
    df1['mean'] = df[var].groupby(df['Gender']).mean()
    df1['median'] = df[var].groupby(df['Gender']).median()
    df1['std'] = df[var].groupby(df['Gender']).std()

    X_axis = np.arange(len(X))
    plt.bar(X_axis + 0.0, df1.loc['Female'], 0.4, label='Female')
    plt.bar(X_axis + 0.4, df1.loc['Male'], 0.4, label='Male')
    plt.xticks(X_axis, X)
    plt.ylabel(var)
    plt.legend()
    plt.show()
