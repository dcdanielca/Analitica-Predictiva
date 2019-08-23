#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:01:56 2019

@author: daniel
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
from sklearn.model_selection import train_test_split


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    random_state = 3)

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Acurracy of Decision Tree classifier on training ser:{:.2f}'
      .format(clf.score(X_train, y_train)))


print('Acurracy of Decision Tree classifier on training ser:{:.2f}'
      .format(clf.score(X_test, y_test))) 
