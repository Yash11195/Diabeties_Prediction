# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sns import heatmap

df=pd.read_csv("C:\\Users/User/Downloads/pima-data.csv")
print(df.head()) # Display top 5 rows.
print(df.tail()) # Display bottom 5 rows.

plt.figure(figsize=(9,9))
heatmap(df,annot=True) # Display the correlation of between columns.
plt.show()

df.drop('skin',1,inplace=True)
diabetes_map={True:1,False:0}
df['diabetes']=df['diabetes'].map(diabetes_map)
print(df.head())

from sklearn.model_selection import train_test_split

feature_column_names=['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
prdicted_class_name=['diabetes']
X=df[feature_column_names].values
Y=df[prdicted_class_name].values
split_test_size=0.30
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=split_test_size,random_state=42) # Splitting the data

from sklearn import impute

fill_0=impute(missing_values=0, strategy="mean", axis=0)
X_train=fill_0.fit_transform(X_train)
X_test=fill_0.fit_transform(X_test)
from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()   
nb_model.fit(X_train,Y_train)  # Training Gaussain Naive Bayes Model
Y_pred=nb_model.predict(X_test)
print(Y_pred)
print("accuracy of model is", nb_model.score(X_test,Y_test)) 

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train,Y_train)   # Training Random Forest Model
Y_pred1=rf.predict(X_test)
print(Y_pred1)
print(print("accuracy of model is", rf.score(X_test,Y_test)))
print(rf.score(X_test,Y_test))

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(C=0.7 ,random_state=42)
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
print(y_pred)
print("accuracy of model is", nb.score(X_test,Y_test))
print(lr.score(X_train,Y_train))

from sklearn import metrics

print(format(metrics.accuracy_score(Y_test,y_pred)))
print(format(metrics.confusion_matrix(Y_test,y_pred)))
print(metrics.classification_report(Y_test,y_pred))

