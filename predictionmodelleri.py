#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm


# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')
#Verileri değişkenlere göre ayırma ve dizileri oluşturma
x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values
#Veriler arasındaki ilişkiyi korelasyon matirisi ile bulma
print(veriler.corr()) 










#Linear Regression modeli inşası ve R2 değerlerine göre inceleme


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
print("Linear R2 degeri:")
print(r2_score(Y,lin_reg.predict(X)))










#Polinomal Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)) ))
print(model2.fit().summary())








#SVR 

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )







#Decision TREE

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())
print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )






#Random Forest 

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,Y.ravel())
print('Random Forest OLS')
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )















