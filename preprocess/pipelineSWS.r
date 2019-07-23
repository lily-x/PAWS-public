# PAWS preprocessing - pipeline
# Lily Xu
# December 2018


### setup
library(sp)
library(rgdal)

rm(list = ls())

path     <- "/Users/lilyxu/Documents/_files/usc/PAWS/code/"
timezone <- "Asia/Phnom_Penh"   # "UTC/GMT"
crs_in   <- "+proj=longlat +datum=WGS84"
crs_use  <- "+proj=utm +zone=48 +north +datum=WGS84 +units=m +no_defs"
resolution <- 1000

startYear <- 2013
endYear   <- 2019

inputPath <- "inputs/sws/"
outputPath <- paste("outputs/sws/", resolution, "/", sep="")

rawDataFilename <- paste(path, inputPath, "SWS_patrolObservations_20190506.csv", sep="")
####################

sourcePrefix <- "preprocess/"

setwd(path)
Sys.setenv(TZ=timezone)

# load in boundary
boundary <- readOGR(dsn=paste(inputPath, "boundary", sep=""), "CA")
boundary <- spTransform(boundary, CRS(crs_use))


# create output folder if it does not exist
if (!dir.exists(outputPath)) {
  dir.create(outputPath, recursive=TRUE)
}

dateFormat <- NULL


### run pre-processing scripts
source(paste(sourcePrefix, "datasetSanitization.r", sep=""))

source(paste(sourcePrefix, "createGrids.r", sep=""))

### process dynamic features
source(paste(sourcePrefix, "effort.r", sep=""))

findHumanPoachingActivity <- function(data) {
  data$poach[(data$Observation.Category.1 == "Live animals" & data$Threat == "Hunting")] <- "Hunting - Live animals"
  data$poach[(data$Observation.Category.1 == "Animal Parts and Bushmeat")] <- "Hunting - Animal Parts and Bushmeat"
  data$poach[(data$Observation.Category.1 == "Carcass")] <- "Hunting - Carcass"
  data$poach[(data$Observation.Category.1 == "Domestic Animals")] <- "Domestic Animals"
  data$poach[(data$Observation.Category.1 == "People Confronted") & (data$Threat == "Hunting")] <- "People Confronted Hunting"
  data$poach[(data$Observation.Category.1 == "Transportation") & (data$Threat == "Hunting")] <- "Hunting - Transportation"
  data$poach[(data$Observation.Category.1 == "Camp") & (data$Threat == "Hunting")] <- "Hunting - Camp"
  data$poach[(data$Observation.Category.1 == "Land Clearing") & (data$Threat == "Hunting")] <- "Hunting - Land Clearing"
  data$poach[(data$Observation.Category.1 == "Sign - Indirect Evidence") & (data$Threat == "Hunting")] <- "Hunting - Indirect Evidence"

  data$poach[(data$Observation.Category.2 == "Firearms & Ammunition")] <- "Firearms & Ammunition"
  data$poach[(data$Observation.Category.2 == "Trap")] <- "Trap"
  data$poach[(data$Observation.Category.2 == "Traditional Weapons")] <- "Traditional Weapons"

  return(data)
}
source(paste(sourcePrefix, "humanActivity.r", sep=""))


### process static features
findAnimalObservations <- function(data) {
  data$importantAnimal[data$Species == "Asian Elephant" | data$Species == "Elephant"] <- "elephant"
  data$importantAnimal[data$Species == "Banteng"] <- "banteng"
  data$importantAnimal[data$Species == "Wild Pig"] <- "wild_pig"
  data$importantAnimal[data$Species == "Red Muntjac" | data$Species == "Muntjac"] <- "muntjac"

  # select only the rows with an illegal activity in poach
  data <- data[!is.na(data$importantAnimal),]

  return(data)
}
source(paste(sourcePrefix, "animalCount.r", sep=""))


# habitat
elevationMapFilename <- paste(inputPath, "elevationMap.tif", sep="")
forestCover <- readOGR(paste(inputPath, "forestCover", sep=""), "MPF_forestcover06")
source(paste(sourcePrefix, "habitat.r", sep=""))


# distance
getDistanceShapes <- function(inputPath) {
  # (note: boundary is automatically included)
  roads              <- readOGR(dsn=paste(inputPath, "roads", sep=""), "SWS_road")
  route76            <- readOGR(dsn=paste(inputPath, "roads", sep=""), "NR76")
  crossroads         <- readOGR(dsn=paste(inputPath, "roads", sep=""), "crossroads")
  riversPermanent    <- readOGR(dsn=paste(inputPath, "rivers", sep=""), "river_permanent")
  riversIntermittent <- readOGR(dsn=paste(inputPath, "rivers", sep=""), "river_intermittent")
  patrolPosts        <- readOGR(dsn=paste(inputPath, "patrolPosts", sep=""), "MPF_20Outpost")
  villages           <- readOGR(dsn=paste(inputPath, "villages", sep=""), "Village in Mondulkiri-WGS84")
  waterholes         <- readOGR(dsn=paste(inputPath, "waterholes", sep=""), "Waterhold_WGS84")
  #waterholesTracked  <- readOGR(dsn=paste(inputPath, "waterholes", sep=""), "waterholes_tracked")
  vietnam            <- readOGR(dsn=paste(inputPath, "vietnam", sep=""), "VNM_adm0")
  coreZone           <- readOGR(dsn=paste(inputPath, "coreZone", sep=""), "SWS_core_zone")
  conservationZone   <- readOGR(dsn=paste(inputPath, "zones", sep=""), "conservation_zone")
  communityZone      <- readOGR(dsn=paste(inputPath, "zones", sep=""), "community_zone")
  sustainableUseZone <- readOGR(dsn=paste(inputPath, "zones", sep=""), "sustainable_use_zone")


  # change the IDs of the spatial lines and polygons of the files to merge
  # riversPermanent    <- spChFIDs(riversPermanent, paste("river", row.names(riversPermanent), sep="_"))
  # riversIntermittent <- spChFIDs(riversIntermittent, paste("river", row.names(riversIntermittent), sep="_"))

  shapes <- list(roads=roads, route76=route76, crossroads=crossroads, rivers_permanent=riversPermanent, rivers_intermittent=riversIntermittent, patrol_posts=patrolPosts, villages=villages, waterholes=waterholes, vietnam=vietnam, core_zone=coreZone, conservation_zone=conservationZone, community_zone=communityZone, sustainable_use_zone=sustainableUseZone)

  for (i in 1:length(shapes)) {
    shapes[[i]] <- spTransform(shapes[[i]], CRS(crs_use))
  }

  return(shapes)
}
shapes <- getDistanceShapes(inputPath)

shapeNames <- names(shapes)

source(paste(sourcePrefix, "distance.r", sep=""))
