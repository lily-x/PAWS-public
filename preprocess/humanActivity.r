# PAWS preprocessing - human activity
# Lily Xu
# December 2018

# libraries needed
library(rgdal)
library(sp)
library(raster)
library(adehabitatHR)
library(maptools)
library(rgeos)


# specify input files
dataFilename         <- paste(path, outputPath, "patrolObservationClean.csv", sep="")
trajectoryFilename   <- paste(path, outputPath, "effort/trajectory.Rdata", sep="")

outFilepath   <- paste(path, outputPath, "human_activity/", sep="")

# create 'human_activity' folder if it does not exist
if (!dir.exists(outFilepath)) {
  dir.create(outFilepath)
}

load(trajectoryFilename)


saveCoordinates <- function(rast, boundary, outFilepath) {
  # save XY coordinate values
  rast <- setValues(rast, 1:ncell(rast))
  rast <- mask(rast, boundary)
  XYvalues <- rasterToPoints(rast)
  # round values to prevent, for example, 10 from being written as 9.999999
  XYvalues <- round(XYvalues, digits=0)
  # prevent values from appearing in scientific notation
  XYvalues <- format(XYvalues, scientific=FALSE)
  XYfilename <- paste(outFilepath, "XY.csv", sep="")
  write.csv(XYvalues, file=XYfilename, quote=FALSE)
}


# note: the only difference here is we purge data that don't have at least 3 points in a row
# other ones return dataGridPixels, this returns data
readData <- function(dataFilename, crs_in, crs_use) {
  data <- read.csv(dataFilename)
  
  # grid as used previously:
  coordinates(data) <- data[,c("X", "Y")] # convert to a spatialpoints* object
  proj4string(data) <- CRS(crs_in)
  data <- spTransform(data, CRS(crs_use))
  
  return(data)
}


findObservations <- function(data, trajectory) {
  # select data from data where the IDs in trajectory == ID_New
  dataNew <- list()
  for (i in 1:length(trajectory)) {
    dataNew[[i]] <- data[as.character(data$ID_New) %in% (levels(ld(trajectory[[i]])$id)),]
  }
  data <- do.call("rbind", dataNew)
  
  
  # TODO: check manually for order
  # Either add in alphabetical order or check manually for order when retrieving from stack 
  data <- findHumanPoachingActivity(data)
  
  # select only the rows with an illegal activity in poach
  data <- data[!is.na(data$poach),]
  
  return(data)
}


# set everything outside boundary to NA
# TODO: data should already be in proper CRS, right?
maskIllegalActivities <- function(data, rast, boundary, crs_use) {
  illegalActivityAll <- rasterize(data, rast, field=1, fun="count", na.rm=T)
  values(illegalActivityAll)[is.na(values(illegalActivityAll))] <- 0
  proj4string(illegalActivityAll) <- CRS(crs_use)
  crs(illegalActivityAll) <- crs_use
  illegalActivityAllMasked <- mask(illegalActivityAll, boundary)
  
  return(illegalActivityAllMasked)
}



###############################################
# Extract data per year and month for each illegal activity class
# combine the above and below (i.e. year and poaching classification)

divideClasses <- function(rast, data, illegalActivityAllMasked) {
  class_arr <- array(0, 
                     dim = c(length(rast), 12, NROW(unique(data$Year)), NROW(unique(data$poach))),
                     dimnames = list(NULL, 
                                     c("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"),
                                     as.character(sort(unique(data$Year))), 
                                     as.character(unique(data$poach))))
  
  #sg: #data.expanded <- data[rep(row.names(data), data$poach),]

  # divide up by classes
  system.time(
    for (class in levels(as.factor(data$poach))) {
      for (year in levels(as.factor(data$Year))) {
        for (month in levels(as.factor(data$Month))) {
          if (sum(data$Year == year & data$Month == month & data$poach == class) > 0) {
            illegalActivity <- rasterize(
              data[data$Year == year & data$Month == month & data$poach == class,], 
              rast, 1, fun="count", na.rm=T)
            values(illegalActivity)[is.na(values(illegalActivity))] <- 0
            class_arr[,as.numeric(month), year, class] <- 
              class_arr[,as.numeric(month), year, class] + values(illegalActivity)
          }
        }
      }
    }
  )
  
  # crop to boundary
  illegalActivityDataFrame <- as.data.frame(illegalActivityAllMasked, xy=TRUE)
  illegalActivityDataFrame$layer[!is.na(illegalActivityDataFrame$layer)]
  class_arr <- class_arr[!is.na(illegalActivityDataFrame$layer),,,]
  
  # save data as Rdata and CSV files
  monthFilename <- paste(outFilepath, "human_activity_month.csv", sep="")
  write.csv(class_arr, file=monthFilename)
  
  # # annual counts of all illegal activities 
  # annualSum <- apply(class_arr, c(1, 3, 4), sum, na.rm=TRUE)
  # yearFilename1 <- paste(outFilepath, "human_activity_year.Rdata", sep="")
  # yearFilename2 <- paste(outFilepath, "human_activity_year.csv", sep="")
  # save(annualSum, file=yearFilename1)
  # write.csv(annualSum, file=yearFilename2)
}


# create visualization of all classes of illegal activities
visualizeClasses <- function(outFilepath, data, boundary) {
  illegalActivityClasses <- list()
  numCols <- ceiling(length(unique(data$poach)) / 3)
  
  png(file=paste(outFilepath, "plot_activities.png", sep=""), width=numCols*450, height=1350)
  par(mfrow=c(3, numCols))
  
  # create plot of all illegal activities
  for (class in unique(data$poach)) {
    illegalActivityClasses[[class]] <- rasterize(data[data$poach==class,], rast, field="poach", fun="count", na.rm=T)
    values(illegalActivityClasses[[class]])[is.na(values(illegalActivityClasses[[class]]))] <- 0
    
    # mask to park boundary
    illegalActivityClasses[[class]] <- mask(illegalActivityClasses[[class]], boundary)
    plot(illegalActivityClasses[[class]], main=class, cex.main=2)
    lines(boundary)
  }
  dev.off()
  
  return(illegalActivityClasses)
}


# create combined visualization of all activities
visualizeCombinedActivities <- function(outFilepath, illegalActivityAllMasked, boundary) {
  png(file=paste(outFilepath, "plot_all_activity.png", sep=""),
      width=800, height=600)
  plot(illegalActivityAllMasked, main="All illegal activity", cex.main=2)
  lines(boundary)
  dev.off()
}


rast <- loadGridRast(gridFilename, crs_use)
saveCoordinates(rast, boundary, outFilepath)
data <- readData(dataFilename, crs_in, crs_use)
data <- findObservations(data, trajectory)
illegalActivityAllMasked <- maskIllegalActivities(data, rast, boundary, crs_use)
divideClasses(rast, data, illegalActivityAllMasked)
illegalActivityClasses <- visualizeClasses(outFilepath, data, boundary)
visualizeCombinedActivities(outFilepath, illegalActivityAllMasked, boundary)

