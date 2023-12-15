library(httr)
library(jsonlite)
library(tidyverse)
library(ggplot2)
setwd('D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/')
mydata <- read.csv('D:/GU/School Work/Fall 21/ANLY 501/assignment 1/raw data/California_Fire_Incidents.csv')
head(mydata, n=5)
AllColNames = names(mydata)
NumCol = ncol(mydata)
NumRow = nrow(mydata)

# drop unwanted columns
dropped.data = mydata[-c(2:4,6:9,11:14,16:22,24:25,27:31,33:40)]
(ColNames = names(dropped.data))
head(dropped.data)
str(dropped.data)

# missing values
sapply(dropped.data, function(x) sum(is.na(x))) # check
dropped.data = na.omit(dropped.data) # remove missing values
sapply(dropped.data, function(x) sum(is.na(x))) # check again, fixed!

# incorrect values
sapply(dropped.data, function(c) sum(c==0)) # check if there is any 0
dropped.data[dropped.data == 0] <- NA 
dropped.data = na.omit(dropped.data) # remove rows with 0
sapply(dropped.data, function(c) sum(c==0)) # check again
nrow(dropped.data)

# data type
str(dropped.data)
dropped.data[3] <- lapply(dropped.data[3], factor)

# feature generation
  # normalize AcresBurned
MinMax <- function(x) {
  MyMax=max(x)
  MyMin = min(x)
  Diff = MyMax - MyMin
  normVal = x/(Diff)
  return(normVal)
}

dropped.data$Severity <- MinMax(dropped.data$AcresBurned)
head(dropped.data)

write.csv(dropped.data, "California_Fire_Incidents_cleaned_093021.csv")
