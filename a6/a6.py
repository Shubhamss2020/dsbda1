import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("/home/estanghost/Desktop/sublimes/a6/IRIS.csv")
print(df)

print("information of dataset is - ")
print(df.info)
print("shape of datset - ")
print(df.shape)
print("columns of dataset - ")
print(df.columns)
print("size of dataset - ")
print(df.size)
print("datatype of dataset - ")
print(df.dtypes)
print("heads of dataset - ")
print(df.head())
print("tails of dataset - ")
print(df.tail())
print("samples of dataset - ")
print(df.sample(5))
print("description of dataset - ")
print(df.describe())

print("Null values are - ")
print(df.isna().sum())

#OULIER DETECTION
data = np.random.rand(100)
sns.boxplot(df['sepal_length'])
plt.show()

# outliers present
data = np.random.rand(100)
sns.boxplot(df['sepal_width'])
plt.show()

# #outlier removal
#df = df[(df["sepal_width"] < 4.05) | (df["sepal_width"] > 2.05)]
df = df[(df['sepal_width'] <= 4.05)]
df = df[(df['sepal_width'] >= 2.05)]
sns.boxplot(df["sepal_width"])
plt.show()

data = np.random.rand(100)
sns.boxplot(df['petal_length'])
plt.show()

data = np.random.rand(100)
sns.boxplot(df['petal_width'])
plt.show()

#character encoding
df['species']=df['species'].astype('category')
print(df.dtypes)
df['species']=df['species'].cat.codes
print(df.dtypes)
print(df)

#correlation matrix
plt.figure(figsize=(16, 6))
sns.heatmap(df.corr(),annot=True)
plt.show()

#split data into input and output
x = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))

from sklearn.metrics import confusion_matrix

#Importing the required modules
 
from sklearn.metrics import confusion_matrix 
#Passing actual and predicted values

cm = confusion_matrix(y_test, y_pred) 
print(cm)

#True write data values in each cell of the matrix
sns.heatmap(cm, annot=True) 
plt.savefig('364.png') 
plt.show()

# #Importing classification report 
from sklearn.metrics import classification_report 

# #Printing the report
print(classification_report(y_test, y_pred)) 


# #Predict iris flower Varity by giving user input: 
 
features = np.array([[5,2.9,1,0.2]]) 
prediction = classifier.predict(features) 
print('Prediction: {}'.format(prediction)) 
