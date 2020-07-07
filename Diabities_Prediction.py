import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("C:\\Users/User/Downloads/pima-data.csv")
print(df)
print(df.corr())
df=df.drop('skin',1)
print(df)
diabetes_map={True:1,False:0}
df['diabetes']=df['diabetes'].map(diabetes_map)
print(df.head(5))
from sklearn.model_selection import train_test_split
feature_column_names=['num_preg',''
                                 'glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
prdicted_class_name=['diabetes']
X=df[feature_column_names].values
Y=df[prdicted_class_name].values
split_test_size=0.30
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=split_test_size,random_state=42)
#from sklearn import impute
#fill_0=impute(missing_values=0, strategy="mean", axis=0)
#X_train=fill_0.fit_transform(X_train)
#_test=fill_0.fit_transform(X_test)
#from sklearn.naive_bayes import GaussianNB
#nb_model=GaussianNB()
#module=nb_model.fit(X_train,Y_train)
#Y_pred=nb_model.fit(X_train,Y_train).predict(X_test)
#print(Y_pred)
#print("accuracy of model is", nb_model.score(X_test,Y_test))
from sklearn.ensemble import RandomForestClassifier
nb_model1=RandomForestClassifier(random_state=42)
module1=nb_model1.fit(X_train,Y_train)
Y_pred1=nb_model1.fit(X_train,Y_train).predict(X_test)
print(Y_pred1)
print(print("accuracy of model is", nb_model1.score(X_test,Y_test)))
#print(nb_model1.score(X_train,Y_train))
#from sklearn.linear_model import LogisticRegression
#nb=LogisticRegression(C=0.7 ,random_state=42)
#mod=nb.fit(X_train,Y_train)
#Pred=nb.fit(X_train,Y_train).predict(X_test)
#print(Pred)
#print(print("accuracy of model is", nb.score(X_test,Y_test)))
#print(nb.score(X_train,Y_train))
from sklearn import metrics
#print(format(metrics.accuracy_score(Y_test,Pred)))
#print(format(metrics.confusion_matrix(Y_test,Pred)))
#print(metrics.classification_report(Y_test,Pred))

