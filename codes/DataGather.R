# install.packages("selectr")
# install.packages("rvest")
# install.packages("xml2")
# install.packages("httr")
# install.packages("tidyverse")
library("selectr")
library("rvest")
library("xml2")
library("httr")
library("jsonlite")
library("tidyverse")

# ## ambee
# ## BUILD THE URL
# base <- "https://api.ambeedata.com/"
# # base <- "https://api.ambeedata.com/latest/fire"
# endpoint <- "latest/fire"
# 
# API_KEY = "553b14ab3d2b2a7ad36531df25571cd02d66416a4ba4c34b0a9683fccb1d1474"
# lat = "36"
# lng = "119"
# distance = "25"
# 
# URL = paste(base,endpoint,
#             "?","lat", "=", lat, 
#             "&","lng","=",lng,
#             "&","distance","=",distance)
# 
# ambee_Call<-httr::GET(URL, add_headers("x-api-key" = API_KEY,
#                       "Content-type" = "application/json"))
# 
# (ambee_Call)

# AerisWeather
base <- "https://api.aerisapi.com/fires/"
action <- "search"
ID = "zyHrTuaFHNsyawGsEtR7v"
Key = "cjfW0Xmo570EZ3wRVAlVUOdEM7ocWTlFLp2c7Nxr"
FM = "json"
q = "state: ca"

URL = paste(base,action,
            "?","query", "=", q,
            "&","format","=",FM,
            "&","client_id","=",ID,
            "&","client_secret","=",Key)

Aeris_Call <- httr::GET(URL)

## Print to a file
AerisData <- httr::content(Aeris_Call)
DataFile <- file("AerisData.rds")
saveRDS(object = AerisData, file = DataFile)