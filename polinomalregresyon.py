#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
#görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()




#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
#görselleştirme
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()




#4. dereceden polinomal regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
#görselleştirme
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler





#NOTTTT :      SVR da scaller kullanılması önemli ve gereklidir. 

#2.7- Eğitim verilerinin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_ölcekli = sc.fit_transform(X)
sc2 = StandardScaler()
Y_ölcekli = sc2.fit_transform(Y)




#support vector kullanarak rbf,poly,linear modelleri.

from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(X_ölcekli,Y_ölcekli)

plt.scatter(X_ölcekli,Y_ölcekli, color="blue")
plt.plot(X_ölcekli,svr_reg.predict(X_ölcekli),color="red")
plt.show()

svr_reg = SVR(kernel = "poly")
svr_reg.fit(X_ölcekli,Y_ölcekli)

plt.scatter(X_ölcekli,Y_ölcekli, color="blue")
plt.plot(X_ölcekli,svr_reg.predict(X_ölcekli),color="red")
plt.show()

svr_reg = SVR(kernel = "linear")
svr_reg.fit(X_ölcekli,Y_ölcekli)

plt.scatter(X_ölcekli,Y_ölcekli, color="blue")
plt.plot(X_ölcekli,svr_reg.predict(X_ölcekli),color="red")
plt.show()



#-----------DecisionTREEE Regresyon için---------------
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')


plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))



#--------------Random forest regresyon için.---------- Kollektif öğrenme 


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()




























