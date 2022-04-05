# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:07:07 2022

@author: julian
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from  sklearn import preprocessing
import statistics as st

accuracy = []


def regresionLogistica(xtrain,ytrain,xtest,ytest):
    
    clf = LogisticRegression(solver='sag')
    
    #~ ##~ clf = LogisticRegression(solver = 'liblinear')
    
    #~ ####Escalado de los datos####
    
    #~ #Robust Scaler
    #xtrain = preprocessing.RobustScaler().fit_transform(xtrain)
    #xtest = preprocessing.RobustScaler().fit_transform(xtest)
    
    xtrain = preprocessing.StandardScaler().fit_transform(xtrain)
    xtest = preprocessing.StandardScaler().fit_transform(xtest)
    
    
    clf.fit(xtrain, ytrain)
    
    y_pred = clf.predict(xtest)
    
    print("El accuracy es:",accuracy_score(ytest, y_pred))
    
    accuracy.append(accuracy_score(ytest, y_pred))
    
    print ('Clase real{}\nClase predicha{}'.format(ytest, y_pred))
    
    print (xtest)

    y_pred_proba = clf.predict_proba(xtest)
    print (y_pred_proba)
    #~ plt.show()
    
    
    
if __name__=='__main__':
    
    
    for i in range (1,4):
        print("Se hizo el numero " + str(i) + " de la regresion Logistica.")
        df1 = pd.read_csv("data_validation_train"+str(i)+".csv", sep=',', engine='python')
        df2 = pd.read_csv("diagnosis_validation_train"+str(i)+".csv", sep=',', engine='python')
        df3 = pd.read_csv("data_validation_test"+str(i)+".csv", sep=',', engine='python')
        df4 = pd.read_csv("diagnosis_validation_test"+str(i)+".csv", sep=',', engine='python')
        regresionLogistica(df1.values,df2, df3.values, df4)
        
    
    print("El promedio de las regresiones logisticas en los plieges es: ", st.mean(accuracy))
    
    print("Ahora probamos todo con el conjunto de entrenamiento de .90.")
    
    df1 = pd.read_csv("data_train"+".csv", sep=',', engine='python')
    df2 = pd.read_csv("diagnosis_train"+".csv", sep=',', engine='python')
    df3 = pd.read_csv("data_test"+".csv", sep=',', engine='python')
    df4 = pd.read_csv("diagnosis_test"+".csv", sep=',', engine='python')
    
    regresionLogistica(df1.values,df2, df3.values, df4)
    
    print("El accuracy total obtenido del conjunto de prueba es: ", accuracy[3])
    