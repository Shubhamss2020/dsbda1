import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support 
df = pd.read_csv("/home/estanghost/Desktop/sublimes/a5/Social_Network_Ads.csv")
sns.heatmap(df.corr(), annot=True) 
plt.show()
X = df[['Age', 'EstimatedSalary']] 
Y = df['Purchased']  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
norm = MinMaxScaler().fit(X_train) #X_train=MinMaxScaler().fit(X_train).transform(X_train) 
X_train = norm.transform(X_train) 
norm = MinMaxScaler().fit(X_test) 
X_test = norm.transform(X_test) 
model = LogisticRegression().fit(X_train,Y_train) 
print('Model Score: ', model.score(X_test, Y_test))
 
# X = df[['Age', 'EstimatedSalary']] 
# Y = df['Purchased']  
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
# print(X_train) 
# print(X_test) 
# norm = MinMaxScaler().fit(X_train) 
# X_train = norm.transform(X_train) 
# norm = MinMaxScaler().fit(X_test) 
# X_test = norm.transform(X_test) 
# model = LogisticRegression() 
# model.fit(X_train,Y_train) 
y_pred = model.predict(X_test) 
# print('Model Score: ', model.score(X_test, Y_test))
cf_matrix = confusion_matrix(Y_test, y_pred) 
print(cf_matrix) 

score = precision_recall_fscore_support(Y_test, y_pred, average='micro') 
print('Precision of Model: ', score[0]) 
print('Recall of Model: ', score[1]) 
print('F-Score of Model: ', score[2])
