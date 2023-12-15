library(stats)  ## for dist
#install.packages("NbClust")
library(NbClust)
library(cluster)
library(mclust)
library(amap)  ## for Kmeans (notice the cap K)
library(factoextra) ## for cluster vis, silhouette, etc.
library(purrr)
#install.packages("stylo")
library(stylo)  ## for dist.cosine
#install.packages("philentropy")
library(philentropy)  ## for distance() which offers 46 metrics
## https://cran.r-project.org/web/packages/philentropy/vignettes/Distances.html
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm) ## to read in corpus (text data)

setwd("D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data")
(data<-read.csv("California_Fire_Incidents_cleaned_101321.csv"))
head(data)
Record_data = data[-c(7,11)]
head(Record_data)

## change all to numeric (not int)
str(Record_data)
library(dplyr)
Record_data <- Record_data %>%
  mutate_all(as.numeric)
str(Record_data)

## distance Metric Matrices using dist
(M2_Eucl <- dist(Record_data,method="minkowski", p=2))  #Euclidean distance 
(M1_Man <- dist(Record_data,method="manhattan")) #Manhattan distance
# (CosSim<- dist(Record_data,method="cosine")) 
(CosSim <- stylo::dist.cosine(as.matrix(Record_data))) #Cosine similarity

## visualize distance matrices by heatmaps
png(file="eucl_dis_heatmap.png", width=1600, height=1600)
heatmap(as.matrix(M2_Eucl), cexRow=3, cexCol = 3)
dev.off()

png(file="man_dis_heatmap.png", width=1600, height=1600)
heatmap(as.matrix(M1_Man), cexRow=3, cexCol = 3)
dev.off()

png(file="cossim_dis_heatmap.png", width=1600, height=1600)
heatmap(as.matrix(CosSim), cexRow=3, cexCol = 3)
dev.off()

## and to choose k
## https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/hclust

################## hierarchical clustering ##################
Hist1 <- hclust(M2_Eucl, method="ward.D2")
plot(Hist1)
##Ward suggested a general agglomerative 
# hierarchical clustering procedure, where 
# the criterion for choosing the pair of 
# clusters to merge at each step is based 
# on the optimal value of an objective function.

Hist2 <- hclust(M1_Man, method="ward.D2")
plot(Hist2)

Hist3 <- hclust(CosSim, method="ward.D2")
plot(Hist3)

##################  k - means ##################

k <- 4 # number of clusters
(kmeansResult1 <- kmeans(Record_data, k)) ## uses Euclidean
kmeansResult1$centers

## k-means result analysis
df = data.frame(kmeansResult1$cluster)
colnames(df) <- c("kmeans_result")
df$Label = data[c(11)]
head(df)
(kmeans_result <- sort(table(df$kmeans_result)))
(original_label <- sort(table(df$Label)))

plot1 <- barplot(kmeans_result, ylim = c(0,160),
                 main = "k-means clustering result, k = 4", 
                 ylab = "Frequency")
text(x = plot1, y = as.numeric(kmeans_result), label = as.numeric(kmeans_result), pos = 3, cex = 1, col = "red")

barplot(original_label)
plot1 <- barplot(original_label,ylim = c(0,190),
                 main = "Clustering based on known label", 
                 ylab = "Frequency")
text(x = plot1, y = as.numeric(original_label), label = as.numeric(original_label), pos = 3, cex = 1, col = "red")



############# To use a different sim metric----------
## one option is akmeans
## https://cran.r-project.org/web/packages/akmeans/akmeans.pdf

library(akmeans)

akmeans(Record_data, min.k=5, max.k=10, verbose = TRUE)
##d.metric = 1  is for Euclidean else it uses Cos Sim



################## Cluster vis ##################
(fviz_cluster(kmeansResult1, data = Record_data,
              ellipse.type = "convex",
              # ellipse.type = "concave",
              palette = "jco",
              # axes = c(1, 4), # num axes = num docs (num rows)
              ggtheme = theme_minimal(),
              title = "k-means clustering, k=3"))


################## Determine optimal number of k ##################

fviz_nbclust(
  as.matrix(Record_data), 
  kmeans, 
  k.max = 6,
  method = "wss", ##Within-Cluster-Sum of Squared Errors 
  diss = get_dist(as.matrix(Record_data), method = "euclidean")
)


## Silhouette
fviz_nbclust(Record_data, method = "silhouette", 
             FUN = hcut, k.max = 5)
