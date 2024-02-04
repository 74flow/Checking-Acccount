import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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


df = pd.read_csv("Project_Data.csv")
column_headers = list(df.columns.values)
print("The Column Header :", column_headers)

print(df)
print(df.info()) #Identify basic information about the data
df = df[['Age', 'Gender', 'DaysDrink','Overdrawn']]
print(df.head())
print(df.tail())
print(df.shape)
# Display the unique values for every column in the data frame
print(df.nunique())
# Display the unique values of a specific column
print(df['Gender'].unique())
print(df.describe())

# Check for missing data
print(df.isnull().sum())

#Printing records containing missing data in entire data frame
missing_data=df.isnull()
missing_data_record=df[missing_data.any(axis=1)]
print(missing_data_record)

# Drop rows with missing values
df= df.dropna()
print(df)
# Check for missing data
print(df.isnull().sum())
print(df.describe())

# Define dictionary to map inconsistent values to consistent ones in a data column of a data frame.
mapping = {'F': 'Female', 'M': 'Male'}
# Replace inconsistent values in the data frame
df['Gender'] = df['Gender'].replace(mapping)
# Display the unique values of a specific column
print(df['Gender'].unique())

outliers = detect_outliers_iqr(df, "Age")
print("outliers are: ",outliers )

outliers = detect_outliers_iqr(df, "DaysDrink")
print("outliers are: ",outliers )

# Transform categorical data to numerical data containing 0s and 1s and adding it to the data frame
df_encoded = pd.get_dummies(df['Gender'], prefix='G') # The name of each numeric column starts with what ever is set for prefix followed by
df = pd.concat([df, df_encoded], axis=1)
print(df_encoded)
print(df.head)

#Rearrange the data frame
df = df[['Gender', 'G_Male', 'G_Female', 'Age', 'DaysDrink', 'Overdrawn']]
print(df.head)
#Split the data into features and target
X =  df[['G_Male', 'G_Female', 'Age', 'DaysDrink']]
y = df['Overdrawn']
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)


#feature Scaling  
#st_x= StandardScaler()    
#X_train = st_x.fit_transform(X_train)    
#X_test = st_x.transform(X_test)  


# Fit the K-NN Model
k = 10
#Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=k)
#Train KNN classifier
knn.fit(X_train, y_train)
# Predict the label of the test instance
y_pred = knn.predict(X_test)
print('The predicted label is: ', y_pred)
print("KNN Accuracy:", accuracy_score (y_test, y_pred)) 
print("KNN Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred))

# Logistic Regression 
# Create Logistic Regression 
logreg = LogisticRegression() 
# Train Logistic Regression 
logreg. fit(X_train, y_train) 
# Predict on test data 
predicted_label = logreg.predict (X_test) 
# Evaluate the performance of Logistic Regression on the testing data 
print('The predicted label is: ', y_pred)
print("Logistic Regression Accuracy:", accuracy_score (y_test, y_pred)) 
print("Logistic Regression Confusion Matrix:", confusion_matrix(y_test, y_pred)) 

# Decision Tree 
# Create decision tree classifier 
classifier = tree.DecisionTreeClassifier() 
# Train decision tree classifier
classifier = classifier.fit(X_test , y_test) 
# Predict on test data 
y_pred = classifier.predict(X_test) 
# Evaluate the performance Decision Tree on the testing data 
print("Decision Tree Accuracy:", accuracy_score (y_test, y_pred)) 
print("Decision Tree Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred))

# RandomForest 
# Create random forest classifier 
classifier= RandomForestClassifier (n_estimators= 10, criterion="entropy") 
# Train random forest classifier 
classifer = classifier. fit (X_test,y_test) 
# Predict on test data 
y_pred = classifier.predict(X_test) 
# Evaluate the performance of RandomForest on the testing data 
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred))

#SVM 
svm = SVC() 
# Train SVM 
svm.fit(X_train, y_train) 
# Predict on test data 
y_pred = svm.predict (X_test) 
# Evaluate the performance of SVM on the testing data 
print ("SVM Accuracy:", accuracy_score(y_test, y_pred))
print ("SVM Confusion Matrix:") 
print (confusion_matrix(y_test, y_pred))

#GaussianNB
classifier = GaussianNB()

classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
#Evalute
print("GaussianNB accuracy: ", accuracy_score(y_test, y_pred))
print("GaussianNB matrix: ")
print (confusion_matrix(y_test, y_pred))
