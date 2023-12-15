### ARM with Twitter text data

setwd("D:/GU/School Work/Fall 21/ANLY 501/assignment 1")

# load twitter credentials
twitter_Path = "D:/Accounts/API/twitter API.txt"
(tokens<-read.csv(twitter_Path, header=T, sep=","))
(consumerKey=as.character(tokens$API_key))
(consumerSecret=as.character(tokens$API_secret))
(access_Token=as.character(tokens$access_token))
(access_Secret=as.character(tokens$access_secret))

requestURL='https://api.twitter.com/oauth/request_token'
accessURL='https://api.twitter.com/oauth/access_token'
authURL='https://api.twitter.com/oauth/authorize'

# install.packages("arulesViz")
library(rlang)
library(usethis)
library(devtools)
library(base64enc)
library(RCurl)
library(httr)
library(twitteR)
library(ROAuth)
library(networkD3)
library(arules)
library(rtweet)
library(jsonlite)
library(streamR)
library(rjson)
library(tokenizers)
library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
# install.packages("syuzhet")  ## sentiment analysis
library(syuzhet)
library(stringr)
library(arulesViz)
library(igraph)
library(httpuv)
library(openssl)

setup_twitter_oauth(consumerKey,consumerSecret,access_Token, access_Secret)
(Search<-twitteR::searchTwitter("#wildfire", n=1000, lang="en"))

(Search_DF <- twitteR::twListToDF(Search))
TransactionTweetsFile = "TweetResults.csv"
(Search_DF$text[3])

## Start the file
Trans <- file(TransactionTweetsFile)

## Tokenize to words 
Tokens<-tokenizers::tokenize_words(
  Search_DF$text[1],stopwords = stopwords::stopwords("en"), 
  lowercase = T,  strip_punct = T, strip_numeric = T,
  simplify = T)

## Write tokens
cat(unlist(Tokens), "\n", file=Trans, sep=",")
close(Trans)

## Append remaining lists of tokens into file
## Recall - a list of tokens is the set of words from a Tweet
Trans <- file(TransactionTweetsFile, open = "a")
for(i in 2:nrow(Search_DF)){
  Tokens<-tokenize_words(Search_DF$text[i],
                         stopwords = stopwords::stopwords("en"), 
                         lowercase = T,  
                         strip_punct = T, 
                         simplify = T)
  
  cat(unlist(Tokens), "\n", file=Trans, sep=",")
  cat(unlist(Tokens))
}

close(Trans)

## Read the transactions data into a dataframe
TweetDF <- read.csv(TransactionTweetsFile, 
                    header = FALSE, sep = ",")
head(TweetDF)
(str(TweetDF))

## Convert all columns to char 
TweetDF<-TweetDF %>%
  mutate_all(as.character)
(str(TweetDF))
# We can now remove certain words
TweetDF[TweetDF == "t.co"] <- ""
TweetDF[TweetDF == "rt"] <- ""
TweetDF[TweetDF == "http"] <- ""
TweetDF[TweetDF == "https"] <- ""
TweetDF[TweetDF == "tweetojie"] <- ""
TweetDF[TweetDF == "nature_o"] <- ""
TweetDF[TweetDF == "want"] <- ""

## Check it so far....
TweetDF

## Clean with grepl - every row in each column
MyDF<-NULL
MyDF2<-NULL
for (i in 1:ncol(TweetDF)){
  MyList=c() 
  MyList2=c() # each list is a column of logicals ...
  MyList=c(MyList,grepl("[[:digit:]]", TweetDF[[i]]))
  MyDF<-cbind(MyDF,MyList)  ## create a logical DF
  MyList2=c(MyList2,(nchar(TweetDF[[i]])<4 | nchar(TweetDF[[i]])>12))
  MyDF2<-cbind(MyDF2,MyList2) 
  ## TRUE is when a cell has a word that contains digits
}
## For all TRUE, replace with blank
TweetDF[MyDF] <- ""
TweetDF[MyDF2] <- ""
(head(TweetDF,10))

# Now we save the dataframe using the write table command 
write.table(TweetDF, file = "UpdatedTweetFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")
TweetTrans <- read.transactions("UpdatedTweetFile.csv", sep =",", 
                                format("basket"),  rm.duplicates = T)
inspect(TweetTrans)

############ Create the Rules  - Relationships ###########
TweetTrans_rules = arules::apriori(TweetTrans, 
                                   parameter = list(support=0.04, conf=0.9, minlen=2))
#maxlen
inspect(TweetTrans_rules[1:15])
##  Sort by Conf
SortedRules_conf <- sort(TweetTrans_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf[1:15])
## Sort by Sup
SortedRules_sup <- sort(TweetTrans_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup[1:15])
## Sort by Lift
SortedRules_lift <- sort(TweetTrans_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift[1:15])

####################################################
### HERE - you can affect which rules are used
###  - the top for conf, or sup, or lift...
####################################################
TweetTrans_rules<-SortedRules_lift[1:50]
inspect(TweetTrans_rules)

## Convert the RULES to a DATAFRAME
Rules_DF2<-DATAFRAME(TweetTrans_rules, separate = TRUE)
(head(Rules_DF2))
str(Rules_DF2)
## Convert to char
Rules_DF2$LHS<-as.character(Rules_DF2$LHS)
Rules_DF2$RHS<-as.character(Rules_DF2$RHS)

## Remove all {}
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[{]', replacement='')
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[}]', replacement='')

Rules_DF2

###########################################
###### Do for SUp, Conf, and Lift   #######
###########################################
## Remove the sup, conf, and count
## USING LIFT
Rules_L<-Rules_DF2[c(1,2,5)]
names(Rules_L) <- c("SourceName", "TargetName", "Weight")
head(Rules_L,30)

## USING SUP
Rules_S<-Rules_DF2[c(1,2,3)]
names(Rules_S) <- c("SourceName", "TargetName", "Weight")
head(Rules_S,30)

## USING CONF
Rules_C<-Rules_DF2[c(1,2,4)]
names(Rules_C) <- c("SourceName", "TargetName", "Weight")
head(Rules_C,30)

## CHoose and set
# Rules_Sup<-Rules_C
Rules_Sup<-Rules_L
# Rules_Sup<-Rules_S

###########################################################################
#############       Build a NetworkD3 edgeList and nodeList    ############
###########################################################################
############################### BUILD THE NODES & EDGES ###################
(edgeList<-Rules_Sup)
(MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE)))

nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                       # because networkD3 library requires IDs to start at 0
                       nName = igraph::V(MyGraph)$name)
## Node Degree
(nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                       v = igraph::V(MyGraph), mode = "all")))

## Betweenness
BetweenNess <- igraph::betweenness(MyGraph, 
                                   v = igraph::V(MyGraph), 
                                   directed = TRUE) 

(nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))

#############################################################
########## BUILD THE EDGES ##################################
#############################################################
# Recall that ... 
# edgeList<-Rules_Sup
getNodeID <- function(x){
  which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
}
## UPDATE THIS !! depending on # choice
(getNodeID("salary")) 

edgeList <- plyr::ddply(
  Rules_Sup, .variables = c("SourceName", "TargetName" , "Weight"), 
  function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                          TargetID = getNodeID(x$TargetName)))

head(edgeList)
nrow(edgeList)

################################################################
##############  Dice Sim #######################################
################################################################
#Calculate Dice similarities between all pairs of nodes
#The Dice similarity coefficient of two vertices is twice 
#the number of common neighbors divided by the sum of the degrees 
#of the vertices. Method dice calculates the pairwise Dice similarities 
#for some (or all) of the vertices. 
DiceSim <- igraph::similarity.dice(MyGraph, vids = igraph::V(MyGraph), mode = "all")
head(DiceSim)

#Create  data frame that contains the Dice similarity between any two vertices
F1 <- function(x) {data.frame(diceSim = DiceSim[x$SourceID +1, x$TargetID + 1])}
#Place a new column in edgeList with the Dice Sim
head(edgeList)
edgeList <- plyr::ddply(edgeList,
                        .variables=c("SourceName", "TargetName", "Weight", 
                                     "SourceID", "TargetID"), 
                        function(x) data.frame(F1(x)))
head(edgeList)
## NetworkD3 Object
#https://www.rdocumentation.org/packages/networkD3/versions/0.4/topics/forceNetwork

D3_network_Tweets <- networkD3::forceNetwork(
  Links = edgeList, # data frame that contains info about edges
  Nodes = nodeList, # data frame that contains info about nodes
  Source = "SourceID", # ID of source node 
  Target = "TargetID", # ID of target node
  Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
  NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
  Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
  Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
  height = 1080, # Size of the plot (vertical)
  width = 1920,  # Size of the plot (horizontal)
  fontSize = 10, # Font size
  linkDistance = networkD3::JS("function(d) { return d.value*1000; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
  linkWidth = networkD3::JS("function(d) { return d.value*5; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
  opacity = 5, # opacity
  zoom = TRUE, # ability to zoom when click on the node
  opacityNoHover = 5, # opacity of labels when static
  linkColour = "red"   ###"edges_col"red"# edge colors
) 

# Save network as html file
networkD3::saveNetwork(D3_network_Tweets, 
                       "ARM_NetD3.html", selfcontained = TRUE)


library(igraph)
library(visNetwork)
library(networkD3)
#data(MisLinks, MisNodes)
library(igraph)
#https://dplyr.tidyverse.org/reference/mutate.html
library(dplyr)
## for pipes %>%
library(magrittr)

###################################################
#############       igraph    #####################
###################################################
## Make sure it works
My_igraph1 <- graph(edges=c(1,2, 2,3, 3,1, 1,1), n=3, directed=F)
plot(My_igraph1)


# write.table(edgeList[1:3], file = "edgeList.csv", col.names = FALSE, 
#             row.names = FALSE, sep = ",")
# write.table(nodeList[1:2], file = "nodeList.csv", col.names = FALSE, 
#             row.names = FALSE, sep = ",")

edgeList_num = read.csv("edgeList.csv")
names(edgeList_num) <- c("SourceName", "TargetName", "Weight")
nodeList_num = nodeList[1:2]
head(edgeList_num)
head(nodeList_num)

(My_igraph2 <- 
    graph_from_data_frame(d = edgeList_num, vertices = nodeList_num, directed = TRUE))

E(My_igraph2)
E(My_igraph2)$Weight

V(My_igraph2)$size = 20
## or you can set this in plot....

(E_Weight<-edgeList_num$Weight)
(E(My_igraph2)$Weight <- edge.betweenness(My_igraph2))
E(My_igraph2)$color <- "purple"

layout1 <- layout.fruchterman.reingold(My_igraph2)

## plot or tkplot........
plot(My_igraph2, edge.arrow.size = 0.2,
       vertex.size=E_Weight*5, 
       vertex.color="lightblue",
       layout=layout1,
       edge.arrow.size=.5,
       vertex.label.cex=1, 
       vertex.label.dist=2, 
       edge.curved=0.2,
       vertex.label.color="black",
       edge.weight=5, 
       edge.width=E(My_igraph2)$Weight,
       #edge_density(My_igraph2)
       ## Affect edge lengths
       rescale = F, 
       ylim=c(0,14),
       xlim=c(0,20)
)

#######################################################
#############       visNetwork    #####################
#######################################################
nodes<-data.frame(id = nodeList$ID, name = nodeList$nName, nodeDegree = nodeList$nodeDegree)
edges<-data.frame(from = edgeList_num$SourceName, to = edgeList_num$TargetName)
head(edges)
head(nodes)
nodes$label <- nodes$name
nodes$nodeDegree <- as.factor(nodes$nodeDegree)
nodes$color.background <- c("yellow","orange","red")[nodes$nodeDegree]

MyVis <- visNetwork(nodes, edges, layout = "layout_with_fr",
           arrows="middle", height = 1080, width = 1920) %>%
  visOptions(selectedBy = "nodeDegree")

visSave(MyVis, file = "ARM_VisNetwork.html", selfcontained = T, background = "white")
                              