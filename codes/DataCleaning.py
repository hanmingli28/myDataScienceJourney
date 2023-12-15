# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 20:43:56 2021

@author: Alex
"""

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

path = 'D:/GU/School Work/Fall 21/ANLY 501/assignment 1/raw data/'
filename = 'NewsData.csv'

My_FILE=open(path+filename, "r")

for next_row in My_FILE:
    print(next_row)

My_FILE.close()

My_Title_List=[]
My_Content_List=[]
with open(path+filename, "r") as My_FILE:
    next(My_FILE)  ## skip the first row
    
    for next_row in My_FILE:
        row_element = next_row.split(",")
        My_Title_List.append(row_element[1])
        My_Content_List.append(row_element[2])

print(My_Content_List)
print(My_Title_List)

CV = CountVectorizer(input='content', stop_words="english")
MyMat = CV.fit_transform(My_Content_List)
ColNames = CV.get_feature_names()
DF= pd.DataFrame(MyMat.toarray(), columns=ColNames)


DF['LABEL'] = 'Description'

DF.to_csv('D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/NewsData_cleaned_093021.csv', index = False)
