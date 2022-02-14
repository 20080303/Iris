from datetime import date
from telnetlib import X3PAD
from tkinter import Y
from matplotlib.cbook import print_cycles
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.2,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
x_pre = knn.predict(x_test)
print(knn.score(x_test,y_test))