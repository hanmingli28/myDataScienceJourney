# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 03:52:19 2021

@author: Alex
"""

import requests
import io, json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer




# Data gathering from NewsAPI
BaseURL="https://newsapi.org/v2/everything"
# BaseURL="https://newsapi.org/v2/top-headlines"

 
URLPost = {'apiKey': '3976cde0c3a74380ae38ffb2660f0163',
                    'q':'wildfire','california'
                    'source':'CNN',
                    'pageSize': 85,
                    'sortBy' : 'top',
                    'totalRequests': 50}

response1=requests.get(BaseURL, URLPost)
jsontxt = response1.json()
# print(jsontxt)

# # print response directly
# with open("NewsData.txt", "w", encoding='utf-8') as f:
#     f.write(response1.text)
    
    
# with open("NewsData_json.txt", "w", encoding='utf-8') as fj:
#     json.dump(jsontxt, fj, ensure_ascii=False)

## Create a new csv file to save the headlines
MyFILE=open("NewsData.csv","w")
### Place the column names in - write to the first row
WriteThis="Author,Title,Description,URL\n"
MyFILE.write(WriteThis)
MyFILE.close()


MyFILE=open("NewsData.csv", "a")
for items in jsontxt["articles"]:
    # print(items)
    
    if items["author"] != None:
        Author=items["author"]
    
    
    ## CLEAN the Title
    ##----------------------------------------------------------
    ##Replace punctuation with space
    # Accept one or more copies of punctuation         
    # plus zero or more copies of a space
    # and replace it with a single space
    Title=items["title"]
    Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'\ +', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'\"', ' ', Title, flags=re.IGNORECASE)
    
    # and replace it with a single space
    ## NOTE: Using the "^" on the inside of the [] means
    ## we want to look for any chars NOT a-z or A-Z and replace
    ## them with blank. This removes chars that should not be there.
    Title=re.sub(r'[^a-zA-Z]', " ", Title, flags=re.VERBOSE)
    ##----------------------------------------------------------
    
    Headline=items["description"]
    Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\ +', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
    
    URL=items["url"]

    # print("Author: ", Author, "\n")
    # print("Title: ", Title, "\n")
    # print("Headline News Item: ", Headline, "\n\n")
        
    WriteThis=Author+ "," + Title + "," + Headline + "," + URL + "\n"
    
    MyFILE.write(WriteThis)
    
## CLOSE THE FILE
MyFILE.close()