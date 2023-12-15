# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:10:56 2021

@author: Alex
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
import seaborn as sns

data = pd.read_csv("D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/NewsData_cleaned_112521.csv")

X = data.drop("LABEL",axis=1)
y = data["LABEL"]

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

######### SVM #########
##linear kernal
SVM=LinearSVC(C=5)
SVM.fit(X_train, y_train)
y_preds = SVM.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_preds)
print(classification_report(y_test, y_preds))
print("\nThe confusion matrix is:")
print(cnf_matrix1)

# vis confusion matrix
df_cm = pd.DataFrame(cnf_matrix1, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu") # font size
plt.xlabel('Predicted Class', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Actual Class', fontsize = 15) # y-axis label with fontsize 15
plt.title('Confusion Matrix',fontsize = 20)
plt.show()

# vis svm
clf_report1= classification_report(y_test, y_preds,output_dict=True)
report1 = pd.DataFrame(clf_report1)
sns.heatmap(report1.iloc[:, :].T, annot=True)
plt.title("SVM Classification Report (Linear)")
plt.show()

## Radial Basis Function(RBF) kernal
SVM_Model2=svm.SVC(C=100, kernel='rbf', 
                           verbose=True, gamma="auto")
SVM_Model2.fit(X_train, y_train)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model2.predict(X_test))
print("Actual:")
print(y_test)
y_pred=SVM_Model2.predict(X_test)
SVM_matrix = confusion_matrix(y_test, y_pred)
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

# vis svm
clf_report2= classification_report(y_test, y_pred,output_dict=True)
report2 = pd.DataFrame(clf_report2)
sns.heatmap(report2.iloc[:, :].T, annot=True)
plt.title("SVM Classification Report (RBF)")
plt.show()

# vis confusion matrix
df_cm = pd.DataFrame(SVM_matrix, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu") # font size
plt.xlabel('Predicted Class', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Actual Class', fontsize = 15) # y-axis label with fontsize 15
plt.title('Confusion Matrix',fontsize = 20)
plt.show()

## POLY
SVM_Model3=svm.SVC(C=10000, kernel='poly',degree=1,
                           gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(X_train, y_train)

print("SVM prediction:\n", SVM_Model3.predict(X_test))
print("Actual:")
print(y_test)
y_pred=SVM_Model3.predict(X_test)
SVM_matrix3 = confusion_matrix(y_test, y_pred)
print("\nThe confusion matrix is:")
print(SVM_matrix3)
print("\n\n")

# vis svm
clf_report3= classification_report(y_test, y_pred,output_dict=True)
report3 = pd.DataFrame(clf_report3)
sns.heatmap(report3.iloc[:, :].T, annot=True)
plt.title("SVM Classification Report (Poly)")
plt.show()

# vis confusion matrix
df_cm = pd.DataFrame(SVM_matrix3, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu") # font size
plt.xlabel('Predicted Class', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Actual Class', fontsize = 15) # y-axis label with fontsize 15
plt.title('Confusion Matrix',fontsize = 20)
plt.show()