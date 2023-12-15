library(tm)
#install.packages("tm")
library(stringr)
# library(wordcloud)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
## ONCE: install.packages("slam")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering
library(naivebayes)
#Loading required packages
#install.packages('tidyverse')
library(tidyverse)
#install.packages('ggplot2')
library(ggplot2)
#install.packages('caret')
library(lattice)
library(caret)
#install.packages('caretEnsemble')
library(caretEnsemble)
#install.packages('psych')
library(psych)
#install.packages('Amelia')
library(Amelia)
#install.packages('mice')
library(mice)
#install.packages('GGally')
library(GGally)
library(e1071)

setwd("D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/")

wfData="California_Fire_Incidents_cleaned_112521.csv"
head(wfDF<-read.csv(wfData))

# correct data types
str(wfDF)
 wfDF$ArchiveYear <- as.factor(wfDF$ArchiveYear)
wfDF$MajorIncident <- as.factor(wfDF$MajorIncident)

# split dataset
(size <- as.integer(nrow(wfDF)*0.3)) # test dataset will be 20% of the data
(sample <- sample(nrow(wfDF), size, replace = FALSE))

(df_test <- wfDF[sample, ])
(df_train <- wfDF[-sample, ])

# remove and keep the label + remove index column
# TEST DATASET
(df_test_label <- df_test$MajorIncident)
df_test_NL <- df_test[ , -which(names(df_test) %in% c("X", "MajorIncident"))]
head(df_test_NL)
(dim(df_test_NL))
# TRAIN DATASET
(df_train_label <- df_train$MajorIncident)
df_train_NL <- df_train[ , -which(names(df_train) %in% c("X", "MajorIncident"))]
head(df_train_NL)
(dim(df_train_NL))

# NB model
# METHOD 1
(NB_e1071 <- naiveBayes(df_train_NL, df_train_label, laplace = 1))
NB_e1071_pred <- predict(NB_e1071, df_test_NL)

# confusion matrix
table(NB_e1071_pred, df_test_label)

# vis
# install.packages("gmodels")
library(gmodels)
CrossTable(x=df_test_label,y=NB_e1071_pred, prop.chisq = F)
barplot(table(df_test_label), main = "Actual Label")
barplot(table(NB_e1071_pred), main = "Predicted Label")

# METHOD 2: cross validation
x <- df_train_NL 
y <- df_train_label
model = train(x,y,'nb',trControl=trainControl(method='cv', number=10))
model$results

Predict <- predict(model,df_test_NL)
table(Predict,df_test_label)
# vis
plot(model)
