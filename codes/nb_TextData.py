# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 03:02:08 2021

@author: Alex
"""

## Textmining Naive Bayes Example
import nltk
from sklearn import preprocessing
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D 
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
## conda install pydotplus
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import seaborn as sns

data = pd.read_csv("D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/NewsData_cleaned_112521.csv")

X = data.drop("LABEL",axis=1)
y = data["LABEL"]

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#Naive Bayes 
MyModelNB = MultinomialNB()
NB1 = MyModelNB.fit(X_train, y_train)
y_preds = MyModelNB.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_preds)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
print(classification_report(y_test, y_preds))

clf_report = classification_report(y_test, y_preds,output_dict=True)
report = pd.DataFrame(clf_report)

sns.heatmap(report.iloc[:, :].T, annot=True)
plt.title("Naive Bayes Classification Report")
plt.show()

# vis confusion matrix
df_cm = pd.DataFrame(cnf_matrix1, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu") # font size
plt.xlabel('Predicted Class', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Actual Class', fontsize = 15) # y-axis label with fontsize 15
plt.title('Confusion Matrix',fontsize = 20)
plt.show()