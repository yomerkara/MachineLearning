# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:54:34 2020

@author: win10
"""
#1. Kütüphanelerin eklenmesi
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#2.1 Veri ekleme
veri=pd.read_csv('veriler.csv')
print(veri)

#2.2 EKsik Veri ekleme

eksikveri=pd.read_csv('eksikveriler.csv')

print(eksikveri)

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy ='mean')
#iloc = integerlocation
Yas=eksikveri.iloc[:,1:4].values
print(Yas)

imputer =imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

#Encoder (Kategorik ->Numeric)

ulke=veri.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

#Label=Verilerin sayısala dönüştürülmesi

le= preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veri.iloc[:,0:1])

print(ulke)

#OneHot= Sayısal verilerin varlığa göre eşlenmesi
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#Verilerin birleştirilmesi ve Dataframe oluşturulması
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veri.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

#axis=same as horizontel 
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)

print(s2)

#Verilerin Train/Test olarak ayırılması
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#Verilerin standrtize edilmesi/ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
