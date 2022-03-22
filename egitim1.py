# -*- coding: utf-8 -*-
#1-kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2---------veriönişleme--------------2#


#2.1-veri yükleme
df = pd.read_csv("veriler.csv")


x = df.iloc[:,1:4].values
y= df.iloc[:,4:].values



# ----------------- VERİLERİ ANALİZ EDİP EN İYİ DEĞİŞKENLERİ KULLANMA ---------------------
# Verilerin arasındaki yüzdesel ilişkiyi söyler-------------------

#--------------------------- TRAİN-TEST --------------- ( OPTİNAL)

#2.6- Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#2.7- Eğitim verilerinin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Logistic Regression

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train,y_train)

tahmin = logr.predict(X_test)
print("LOGR")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,tahmin)
print(cm)




#KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("KNN")
cm = confusion_matrix(y_test,y_pred)
print(cm)

#SUPPORT VECTOR MACHİNE
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)

#Naif Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("GNB")
print(cm)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dcs = DecisionTreeClassifier(criterion="entropy")
y_pred = dcs.fit(X_train,y_train).predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("DCS")
print(cm)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10,criterion = "gini")
y_pred = rf.fit(X_train,y_train).predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("RF")
print(cm)







