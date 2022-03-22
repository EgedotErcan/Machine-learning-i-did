# -*- coding: utf-8 -*-

#1-kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2---------veriönişleme--------------2#


#2.1-veri yükleme
df = pd.read_csv("veriler.csv")

# ----------------- VERİLERİ ANALİZ EDİP EN İYİ DEĞİŞKENLERİ KULLANMA ---------------------
# Verilerin arasındaki yüzdesel ilişkiyi söyler-------------------
print(df.corr()) 




#2.2.2-eksik verileri tamamlama
"""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = df.iloc[:,1:4].values

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
"""




#2.3- endcoder :  Kategorik verilerden ---> Numeriğe çevirme işlemi
#-------Katagorik verileri numeriğe çevirmenin kolay yolu (hiç uğraştırmıyor genellikle bunu kullan sonra ayırırsın....)

veriler2 = df.apply(preprocessing.LabelEncoder().fit_transform)
"""



from sklearn import preprocessing
ulke = df.iloc[:,0:1].values
#------ LabelEndcoder() : Her bir veriyi sayılar ile isimlendirir . 
let = preprocessing.LabelEncoder()
ulke[:,0] = let.fit_transform(df.iloc[:,0])

#------ OneHotEncoder() : Her bir veriyi sayısallaştırır ve 0,1 etiketilerini verir.
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
"""








#2.4- Oluşturulan Numpy dizilerini ---> Data frame dönüşümü ve oluşanların birleşmesi.
"""
yas = df.iloc[:,1:4].values
cinsi = df.iloc[:,-1].values

sonuc = pd.DataFrame(data = ulke, index= range(22), columns=["fr","tr","us","en"])
sonuc1 = pd.DataFrame(data= yas, index= range(22),columns=["boy","kilo","yas"])
sonuc2 = pd.DataFrame(data = cinsi, index= range(22), columns=["cinsiyet"] )

#2.5- Dataframe birleştirmesi
s = pd.concat([sonuc,sonuc1],axis = 1)
result = pd.concat([s,sonuc2],axis = 1)
"""







#--------------------------- TRAİN-TEST --------------- ( OPTİNAL)
"""
#2.6- Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(s,sonuc2,test_size=0.33,random_state=0)


#2.7- Eğitim verilerinin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_test = sc.fit_transform(x_test)
X_train = sc.fit_transform(x_train)
"""



# ----------- VERİLERİ ÖLÇEKLENDİRME SCALLİNG--------
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
a_ölcekli = sc.fit_transform(x_train)
"""







#----------------------Uzun p_value analizi-----------------
import statsmodels.api as sm
"""
#1.Değişkenlerden oluşan bir dizi oluşturulup sırasıyla değişkenler elenecek. P_valusi yüksek olanlar elenecek. sıra sıra amaçç  : modeli geliştirmek . 
#FORMÜLDEKİ B0 I 1 YAPTIK
X= np.append(arr = np.ones((22,1)).astype(int),values = traningverileri, axis=1)


#VERİLERDEKİ ELEM KOLAYLAŞTIRMAK İÇİN VERİLERİ TOPLU ALARAK ELE ALDIK. 
X_l = traningverileri.iloc[:,[0,1,2,3,4,5,6]].values
#veriyi dizi haline getirmek
X_l = np.array(X_l,dtype=float)

#asıl modeli oluşturma
#OLS() metodundaki ilk parametre bağlı değiken 2. bağımsız değişken

model = sm.OLS(boy,X_l).fit()

print(model.summary())
"""
























