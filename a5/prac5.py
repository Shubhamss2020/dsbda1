import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('/home/estanghost/Desktop/sublimes/a5/Social_Network_Ads.csv')

#heatmap or corr matrix
sns.heatmap(df.corr(), annot=True) 
plt.show()

# Split the data into inputs (X) and output/target variable (y)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Norm

# scr=StandardScaler()
# X_train=scr.fit_transform(X_train)
# X_test=scr.fit_transform(X_test)

mm=MinMaxScaler()
X_train=mm.fit_transform(X_train)
X_test=mm.fit_transform(X_test)

# Fit the logistic regression model
model = LogisticRegression().fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True)
plt.show()
print("Confusion Matrix:")
print(confusion)

# Extract TP, FP, TN, FN from the confusion matrix
TP = confusion[1, 1]
FP = confusion[0, 1]
TN = confusion[0, 0]
FN = confusion[1, 0]

# Compute accuracy : overall correctness of the model's predictions
accuracy = (TP + TN) / (TP + FP + TN + FN)

# Compute error rate : The complement of accuracy
error_rate = 1 - accuracy

# Compute precision : It indicates the accuracy of positive predictions
if TP + FP > 0:
    precision = TP / (TP + FP)
else:
    precision = 0.0

# Compute recall : It represents the proportion of correctly predicted positive instances out of all actual positive instances.
recall = TP / (TP + FN)

# Print the computed metrics
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)