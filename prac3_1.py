# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:02:43 2022

@author: julian
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle

np.set_printoptions(threshold=sys.maxsize, suppress=True)

class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

class train_set:
	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

class data_set:
	def __init__(self, validation_set, test_set, train_set):
		self.validation_set = validation_set
		self.test_set = test_set
		self.train_set = train_set

def generate_train_test(file_name):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['diagnosis'],axis=1).values
	y = df['diagnosis'].values

	#Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
	X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.1, shuffle = True)


	#~ print (X_train.shape)
	#~ print (X_train)
	#~ print (y_train.shape)
	#~ print (y_train)

	#~ #Crea pliegues para la validación cruzada
	validation_sets = []
	kf = KFold(n_splits=3)
	for train_index, test_index in kf.split(X_train):
		#~ print("TRAIN:", train_index, "\n",  "TEST:", test_index)
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))

	#~ #Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)

	my_train_set = train_set(X_train, y_train)


	#~ #Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set, my_train_set)

	return (my_data_set)

if __name__=='__main__':
	my_data_set = generate_train_test('breast-cancer.csv')

	print (my_data_set.test_set.X_test)
	#~ print(type(my_data_set.test_set.X_test))
	#~ print ('\n----------------------------------------------------------------------------------\n')

	#Guarda el dataset en formato csv
	np.savetxt("data_test.csv", my_data_set.test_set.X_test, delimiter=",", fmt="%s",
           header="id,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst")

	np.savetxt("diagnosis_test.csv", my_data_set.test_set.y_test, delimiter=",", fmt="%s",
           header="diagnosis", comments="")

	np.savetxt("data_train.csv", my_data_set.train_set.X_train, delimiter=",", fmt="%s",
			   header="id,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst")

	np.savetxt("diagnosis_train.csv", my_data_set.train_set.y_train, delimiter=",", fmt="%s",
			   header="diagnosis", comments="")

	i = 1
	for val_set in my_data_set.validation_set:
		np.savetxt("data_validation_train" + str(i) + ".csv", val_set.X_train, delimiter=",", fmt="%s",
		   header="id,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst", comments="")
		np.savetxt("data_validation_test" + str(i) + ".csv", val_set.X_test, delimiter=",", fmt="%s",
           header="id,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst", comments="")
		np.savetxt("diagnosis_validation_train" + str(i) + ".csv", val_set.y_train, delimiter=",", fmt="%s",
           header="diagnosis", comments="")
		np.savetxt("diagnosis_validation_test" + str(i) + ".csv", val_set.y_test, delimiter=",", fmt="%s",
           header="diagnosis", comments="")
		i = i + 1

	#Guarda el dataset en pickle
	dataset_file = open ('dataset.pkl','wb')
	pickle.dump(my_data_set, dataset_file)
	dataset_file.close()

	dataset_file = open ('dataset.pkl','rb')
	my_data_set_pickle = pickle.load(dataset_file)
	print ("-----------------------------------------------")
	print (my_data_set_pickle.test_set.X_test)