import numpy as np
from numpy import genfromtxt
import pandas as pd
from matplotlib import pyplot as plt

def genDataSet():
	train = pd.read_csv("train.csv", index_col=0)
	y_train = pd.get_dummies(train[["type"]], prefix="")
	train.drop("type", inplace=True, axis=1)

	test = pd.read_csv("test.csv", index_col=0)

	train_test = pd.concat([train, test], axis=0)

	train_test.drop("color", inplace=True, axis=1)

	X_train = train_test.iloc[:len(y_train)]
	X_test  = train_test.iloc[len(y_train):]

	X = np.array(X_train, dtype=np.float32)
	ytrue = np.array(y_train, dtype=np.float32)

	#print X
	#print X_test
	print y_train
	return X, ytrue
#genDataSet()
