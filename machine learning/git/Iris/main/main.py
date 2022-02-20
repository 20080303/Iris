from datetime import date
from telnetlib import X3PAD
from tkinter import Y
from matplotlib.cbook import print_cycles
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pre = knn.predict(x_test)
print("expectation:\n {}".format(y_test))
print("predictions:\n {}".format(y_pre))
print("score: {:.2f}".format(np.mean(y_pre == y_test)))