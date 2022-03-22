# -*- coding: utf-8 -*-

#1-kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2---------veriönişleme--------------2#
#2.1-veri yükleme
df = pd.read_csv("odev_tenis.csv")
#2.3- endcoder :  Kategorik verilerden ---> Numeriğe çevirme işlemi

from sklearn import preprocessing

veriler2 = df.apply(preprocessing.LabelEncoder().fit_transform)


outlook = veriler2.iloc[:,:1]

from sklearn import preprocessing
outlook = df.iloc[:,0:1].values
#------ LabelEndcoder() : Her bir veriyi sayılar ile isimlendirir . 
let = preprocessing.LabelEncoder()
outlook[:,0] = let.fit_transform(df.iloc[:,0])
#------ OneHotEncoder() : Her bir veriyi sayısallaştırır ve 0,1 etiketilerini verir.
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

#2.4- Oluşturulan Numpy dizilerini ---> Data frame dönüşümü ve oluşanların birleşmesi.



havadurumu = pd.DataFrame(data = outlook,index = range(14), columns=["overcast","rainy","sunny"])
sonveriler = pd.concat([havadurumu,df.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([sonveriler,veriler2.iloc[:,-2:]],axis=1)




humidity = sonveriler.iloc[:,4:5]

vegitim = sonveriler.iloc[:,:4]
vegitim1 = sonveriler.iloc[:,-2:]
egitimson = pd.concat([vegitim,vegitim1],axis=1)
#2.6- Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(egitimson,humidity,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_predict = r2.predict(x_test)
print(y_predict)




import statsmodels.api as sm

#1.Değişkenlerden oluşan bir dizi oluşturulup sırasıyla değişkenler elenecek. P_valusi yüksek olanlar elenecek. sıra sıra amaçç  : modeli geliştirmek . 
#FORMÜLDEKİ B0 I 1 YAPTIK
X= np.append(arr = np.ones((14,1)).astype(int),values = egitimson, axis=1)


#VERİLERDEKİ ELEM KOLAYLAŞTIRMAK İÇİN VERİLERİ TOPLU ALARAK ELE ALDIK. 
X_l = egitimson.iloc[:,[0,1,2,3,5]].values
#veriyi dizi haline getirmek
X_l = np.array(X_l,dtype=float)

#asıl modeli oluşturma
#OLS() metodundaki ilk parametre bağlı değişken 2. bağımsız değişken...

model = sm.OLS(humidity,X_l).fit()

print(model.summary())






