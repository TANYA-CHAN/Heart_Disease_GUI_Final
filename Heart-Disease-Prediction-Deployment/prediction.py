# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

heart = pd.read_csv("heart_cleveland_upload.csv")

# creating a copy of dataset so that will not affect our original dataset.
heart_df = heart.copy()

# Renaming some of the columns 
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head())

# model building 

#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

"""### K-nearest-neighbor classifier """

# creating Knn Model
# Knn_model= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# Knn_model.fit(x_train_scaler, y_train)
# y_pred_knn= Knn_model.predict(x_test_scaler)
# Knn_model.score(x_test_scaler,y_test)

# print('Classification Report\n', classification_report(y_test, y_pred_knn))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_knn)*100),2)))

# cm = confusion_matrix(y_test, y_pred_knn)
# cm

"""### Support Vector Classifier"""

SVC_model= SVC()
SVC_model.fit(x_train_scaler, y_train)
y_pred_SVC= SVC_model.predict(x_test_scaler)
SVC_model.score(x_test_scaler,y_test)

print('Classification Report\n', classification_report(y_test, y_pred_SVC))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_SVC)*100),2)))

cm = confusion_matrix(y_test, y_pred_SVC)
cm

# """### Random Forest Classifier"""

# RF_model= RandomForestClassifier(n_estimators=20)
# RF_model.fit(x_train_scaler, y_train)
# y_pred_RF= RF_model.predict(x_test_scaler)
# p = RF_model.score(x_test_scaler,y_test)
# print(p)

# print('Classification Report\n', classification_report(y_test, y_pred_RF))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_RF)*100),2)))

# cm = confusion_matrix(y_test, y_pred_RF)
# cm

# """### Decison Tree Classifier"""

# DT_model= DecisionTreeClassifier()
# DT_model.fit(x_train_scaler, y_train)
# y_pred_DT= DT_model.predict(x_test_scaler)
# DT_model.score(x_test_scaler,y_test)

# print('Classification Report\n', classification_report(y_test, y_pred_DT))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_DT)*100),2)))

# cm = confusion_matrix(y_test, y_pred_DT)
# cm

# Creating a pickle file for the classifier
# filename = 'heart_disease_prediction_LR.pkl'
# pickle.dump(LR_model, open(filename, 'wb'))

# filename = 'heart_disease_prediction_KNN.pkl'
# pickle.dump(Knn_model, open(filename, 'wb'))

filename = 'heart_disease_prediction_SVC.pkl'
pickle.dump(SVC_model, open(filename, 'wb'))

# filename = 'heart_disease_prediction_RF.pkl'
# pickle.dump(RF_model, open(filename, 'wb'))

# filename = 'heart_disease_prediction_DT.pkl'
# pickle.dump(DT_model, open(filename, 'wb'))
