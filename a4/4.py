def RemoveOutlier(df, var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

def DisplayOutlier(df, msg):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(msg)
    sns.boxplot(data = df, x = 'RM', ax = axes[0])
    sns.boxplot(data = df, x = 'LSTAT', ax = axes[1])
    fig.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read dataset
df = pd.read_csv('/home/estanghost/Desktop/sublimes/a4/HousingData.csv')
print('Boston dataset loaded.')

#Display information about dataset
print('Information of dataset:\n', df.info)
print('Shape of dataset (row x column):', df.shape)
print('Column names: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatypes of attributes:\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

#Find missing values
print('Missing values:\n', df.isnull().sum())

#Find correlation matrix
print('Finding correlation matrix using heatmap: ')
sns.heatmap(df.corr(), annot=True)
plt.show()



#Finding and removing outliers
print('Finding and removing outliers:')
DisplayOutlier(df, 'Before removing outliers')
print('Identifying outliers')
df = RemoveOutlier(df, 'LSTAT')
df = RemoveOutlier(df, 'RM')
DisplayOutlier(df, 'After removing outliers')

#Split the data into inputs and outputs
x = df[['RM', 'LSTAT']] #input data
y = df['MEDV']          #output data

#Training and testing data
from sklearn.model_selection import train_test_split

#Assign test data size 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

#Apply linear regression model on training data
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)


y_pred = LinearRegression().fit(x_train, y_train).predict(x_test)



#Display accuracy of the model
from sklearn.metrics import mean_absolute_error
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('Model score: ', model.score(x_test, y_test))

#Test the model using user input
print('Predict house price by giving user input: ')
features = np.array([[6, 19]])
prediction = model.predict(features)
print("\n\n\n")
print("**************************************************")
print(' Final Prediction: {}'.format(prediction))
print("**************************************************")
