import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#User defined function for detecting outliers and substituting them with the mean values
def detect_outliers_iqr(df, column_name):
    # Calculate the first and third quartiles
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Identify outliers using the IQR
    outliers = df[(df[column_name] < (Q1 - 1.5 * IQR)) | (df[column_name] > (Q3 + 1.5 * IQR))]
    # Replace outliers with the mean
    df[column_name] = np.where((df[column_name] < (Q1 - 1.5 * IQR)) | (df[column_name] > (Q3 + 1.5 * IQR)), df[column_name].mean(), df[column_name])
    return outliers

# Create the DataFrame using the dictionary
df = pd.read_csv("Project_Data.csv")
# Display Basic Information of Data Frame
print(df.info()) 
df =  df[['Age', 'Gender', 'DaysDrink', 'Overdrawn']]
# Display the first 5 rows of the data frame
print(df.head())
# Display the last 5 rows of the data frame
print(df.tail())
# Display the size of the data frame 
print(df.shape)
# Display the unique values for every column in the data frame / Looks for any features with data inconsitency
print(df.nunique())
# Display the unique values of a specific column / Identify Inconsist Data
print(df['Gender'].unique()) 
# Check the data description
print(df.describe())
# Check for missing data
print(df.isnull().sum())
# Printing records containing missing data in entire data frame
missing_data=df.isnull()
missing_data_record=df[missing_data.any(axis=1)]
print(missing_data_record)

# Drop rows with missing values
df= df.dropna()
# Check if Missing values are dropped
print(df.describe())
# Check for missing data in data frame 
missing_data_record=df[missing_data.any(axis=1)]
# Define dictionary to map inconsistent values to consistent ones in a data columns M and F of a data frame.
mapping = {'M':'Male', 'F': 'Female' }
# Replace inconsistent values in the data frame for Gender
df['Gender'] = df["Gender"].replace(mapping)
print(df['Gender'].unique())
# Handling outliers in one column of data in the data frame
outliers = detect_outliers_iqr(df, 'DaysDrink') 
print("Outliers are: ", outliers)
# Transform categorical data to numerical data containing 0s and 1s and adding it to the data frame for Gender
df_encoded = pd.get_dummies(df['Gender'], prefix='G')
df = pd.concat([df, df_encoded], axis=1)
print(df.head)

# Rearrange the data frame in the following format: Gender, Gender_Male, Gender_Female
df=df[['Gender', 'Age', 'G_Male' , 'G_Female', 'DaysDrink', 'Overdrawn']]
print(df.head())
# Split the data into features and target 
X=df[['G_Male', 'G_Female','Age', 'DaysDrink']]
y=df['Overdrawn']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Logistic Regression
# Create Logistic Regression
logreg = LogisticRegression()
# Train Logistic Regression
logreg.fit(X_train, y_train)
# Predict on test data
y_pred = logreg.predict(X_test)
# Evaluate the performance of Logistic Regression on the testing data
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# KNN
k = 10
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=k)
# Train KNN classifier
knn.fit(X_train, y_train)
# Predict on test data
y_pred = knn.predict(X_test)
# Evaluate the performance of KNN on the testing data
print("The prediected labels are :" , y_pred)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Decision Tree
# Create decision tree classifier
classifier = tree.DecisionTreeClassifier()
# Train decision tree classifier
classifier = classifier.fit(X_test , y_test)
# Predict on test data
y_pred = classifier.predict(X_test)
# Evaluate the performance Decision Tree on the testing data
print("The prediected labels are :" , y_pred)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# RandomForest
# Create random forest classifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
# Train random forest classifier
classifer = classifier.fit(X_test,y_test)
# Predict on test data
y_pred = classifier.predict(X_test)
# Evaluate the performance of RandomForest on the testing data
print("The prediected labels are :" , y_pred)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#SVM
svm = SVC()
# Train SVM
svm.fit(X_train, y_train)
# Predict on test data
y_pred = svm.predict(X_test)
# Evaluate the performance of SVM on the testing data
print("The prediected labels are :" , y_pred)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#GaussianNB
# Create GaussianNB classifier
classifier = GaussianNB()  
# Train GaussianNB classifier
classifier.fit(X_train, y_train) 
# Predict on test data 
y_pred = classifier.predict(X_test)  
# Evaluate the performance of GaussianNB on the testing data
print("The prediected labels are :" , y_pred)
print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred))
print("GaussianNB Matrix:")
print(confusion_matrix(y_test, y_pred))
