# PAWS preprocessing - dataset sanitization
# Lily Xu
# December 2018

library(raster)

outFilename <- paste(path, outputPath, "patrolObservationClean.csv", sep="")


# note: dateFormat is an optional parameter
sanitizeData <- function(rawDataFilename, outFilename, boundary, crs_in, startYear, endYear, dateFormat=NULL) {
  data <- read.csv(rawDataFilename)
  
  # transform boundary to CRS of input data
  boundary <- spTransform(boundary, CRS(crs_in))
  bounds   <- extent(boundary)
  
  # locate out-of-bounds data
  rowsToRemove <- (data$X == 0 & data$Y == 0)
  rowsToRemove <- (data$X < xmin(bounds) | data$X > xmax(bounds)) + rowsToRemove
  rowsToRemove <- (data$Y < ymin(bounds) | data$Y > ymax(bounds)) + rowsToRemove
  
  if (is.null(dateFormat) | missing(dateFormat)) {
    dateFormat <- c("%b %d, %Y", "%d-%b-%y", "%m/%d/%Y")
  } else {
    dateFormat <- c(dateFormat)
  }
  
  # format dates correctly
  data$Patrol.End.Date   <- format(as.Date(data$Patrol.End.Date, tryFormats=dateFormat, tz=timezone), "%Y-%m-%d")
  data$Patrol.Start.Date <- format(as.Date(data$Patrol.Start.Date, tryFormats=dateFormat, tz=timezone), "%Y-%m-%d")
  data$Waypoint.Date     <- format(as.Date(data$Waypoint.Date, tryFormats=dateFormat, tz=timezone), "%Y-%m-%d")
  
  # add year and month columns
  data$Year  <- format(as.Date(data$Waypoint.Date, tz=timezone), format="%Y")
  data$Month <- format(as.Date(data$Waypoint.Date, tz=timezone), format="%m")
  
  # remove dates beyond start and end years
  rowsToRemove <- (data$Year < startYear | data$Year > endYear) + rowsToRemove
  
  # remove out-of-bounds data
  rowsToRemove <- as.logical(rowsToRemove)
  data         <- subset(data, !rowsToRemove)
  
  # patrol IDs are given as strings, so we assign a unique numerical ID to each station
  patrolIDInt <- as.numeric(factor(data$Patrol.ID, levels=unique(data$Patrol.ID)))
  
  # patrol day is an int, representing the day number for that particular waypoint
  patrolDay <- anydate(data$Waypoint.Date) - anydate(data$Patrol.Start.Date) + 1
  
  # ID_New is the unique IDs used in the trajectories
  data$ID_New <- as.numeric(paste(patrolIDInt, patrolDay, sep="."))
  
  write.csv(data, outFilename, row.names=FALSE)
  
  return(data)
}


# read in data
data <- sanitizeData(rawDataFilename, outFilename, boundary, crs_in, startYear, endYear, dateFormat)
