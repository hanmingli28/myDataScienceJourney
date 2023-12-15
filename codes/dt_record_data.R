## LIBRARIES
# install.packages("caret")
library(rpart)   ## FOR Decision Trees
library(rattle)  ## FOR Decision Tree Vis
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(network)
library(ggplot2)
#Sys.setenv(NOAWT=TRUE)
library(wordcloud)
## ONCE: install.packages("tm")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
#library(SnowballC)
library(proxy)
## ONCE: if needed:  install.packages("stringr")
library(stringr)
## ONCE: install.packages("textmineR")
library(textmineR)
library(igraph)
library(caret)
#library(lsa)
library(dplyr)



set.seed(2021)
path <- 'D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/California_Fire_Incidents_cleaned_111321.csv'
wildfire <-read.csv(path)
head(wildfire)

# shuffle
shuffle_index <- sample(1:nrow(wildfire))
head(shuffle_index)

wildfire <- wildfire[shuffle_index, ]
head(wildfire)

str(wildfire)
nrow(wildfire)

# cleaning
# clean_wildfire <- wildfire %>%
#   select(-c(PassengerId, Embarked, Name, Cabin, Ticket, Parch, SibSp)) %>%
#   mutate(Pclass = factor(Pclass, levels = c(1, 2, 3), labels = c('Upper', 'Middle', 'Lower')),
#          Survived = factor(Survived, levels = c(0, 1), labels = c('No', 'Yes')),
#          Sex = factor(Sex, levels = c("male", "female"), labels = c('male', 'female'))) %>%
#   na.omit()
# glimpse(clean_wildfire)
# nrow(clean_wildfire)
# 
# apply(clean_wildfire, 2, table) 

## Define the function on any dataframe input x
clean_wildfire <- wildfire
GoPlot <- function(x) {
  
  G <-ggplot(data=clean_wildfire, aes(.data[[x]], y="") ) +
    geom_bar(stat="identity", aes(fill =.data[[x]])) 
  
  return(G)
}

## Use the function in lappy
lapply(names(clean_wildfire), function(x) GoPlot(x))

## Split into TRAIN and TEST data
(DataSize=nrow(clean_wildfire)) ## how many rows?
(TrainingSet_Size<-floor(DataSize*(3/4))) ## Size for training set
(TestSet_Size <- DataSize - TrainingSet_Size) ## Size for testing set

set.seed(1234)

## This is the sample of row numbers
(MyTrainSample <- sample(nrow(clean_wildfire),
                         TrainingSet_Size,replace=FALSE))

## Use the sample of row numbers to grab those rows only from
## the dataframe
(MyTrainingSET <- clean_wildfire[MyTrainSample,])
table(MyTrainingSET$Label)

## Training and Testing datasets MUST be disjoint. Why?
(MyTestSET <- clean_wildfire[-MyTrainSample,])
table(MyTestSET$Label)

## REMOVE THE LABELS from the test set
(TestKnownLabels <- MyTestSET$Label)
(MyTestSET <- MyTestSET[ , -which(names(MyTestSET) %in% c("Label"))])

MyTrainingSET
str(MyTrainingSET)

DT <- rpart(MyTrainingSET$Label ~ ., data = MyTrainingSET, method="class")
summary(DT)

DT2<-rpart(MyTrainingSET$Label ~ ., data = MyTrainingSET,cp=0.1, method="class")
summary(DT2)

DT3<-rpart(MyTrainingSET$Label ~ ., 
           data = MyTrainingSET,cp=0, method="class",
           parms = list(split="information"),minsplit=2)
summary(DT3)

DT4<-rpart(MyTrainingSET$Label ~ Injuries + StructuresDestroyed + PersonnelInvolved, 
           data = MyTrainingSET,cp=0.05, method="class",
           parms = list(split="information"),minsplit=2)
summary(DT4)

DT5<-rpart(MyTrainingSET$Label ~ Injuries + StructuresDestroyed + PersonnelInvolved + Latitude + Longitude, 
           data = MyTrainingSET,cp=0.032, method="class",
           parms = list(split=c("information", "gini")),minsplit=2, maxdepth = 15)
summary(DT5)

plotcp(DT)
plotcp(DT5)

fancyRpartPlot(DT, cex=.7)
fancyRpartPlot(DT2, cex=.7)
fancyRpartPlot(DT3, cex=.7)
fancyRpartPlot(DT4, cex=0.7)
fancyRpartPlot(DT5, cex=0.7)
head(wildfire)

## DT---------------------------------
(DT_Prediction= predict(DT5, MyTestSET, type="class"))
## Confusion Matrix
(A <- table(DT_Prediction,TestKnownLabels)) ## one way to make a confu mat
## Accuracy
(accuracy <- sum(diag(A))/sum(A))
