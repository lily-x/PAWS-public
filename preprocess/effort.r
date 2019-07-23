# PAWS preprocessing - patrol effort
# Lily Xu
# December 2018

### NOTE: there are two ways to generate effort.
# This code uses patrol length. Another method is to use utilization distribution.
# This will result in slightly diverent results, but they are very well correlated.


### setup

# libraries needed
library(maptools)
library(rgeos)
library(rgdal)
library(sp)
library(trip)
library(adehabitatLT)
library(spatstat)
library(adehabitatHR)
library(raster)
library(anytime)
library(colorspace)


dataFilename <- paste(path, outputPath, "patrolObservationClean.csv", sep="")

outFilepath <- paste(path, outputPath, "effort/", sep="")

# create folder if it does not exist
if (!dir.exists(outFilepath)) {
  dir.create(outFilepath)
}

# The following columns must be available in the inupt data with right names
# colNames = list('Waypoint Date', 'Waypoint Time', 'Patrol ID', 'Patrol Start Date', 'Patrol End Date', 'X', 'Y')


formatData <- function(data, crs_in, crs_use) {
  # create a DateTime column in the data file (for compatibility)
  data$DateTime <- paste(anydate(data$Waypoint.Date), data$Waypoint.Time)
  data$DateTimePosix <- as.POSIXct(strptime(data$DateTime, format="%Y-%m-%d %H:%M:%S", tz=timezone))
  
  coordinates(data) <- data[,c("X", "Y")] # convert to a spatialpoints* object
  proj4string(data) <- CRS(crs_in)
  data <- spTransform(data, CRS(crs_use))
  
  return(data)
}

# load and process data
cleanData <- function(dataFilename, crs_in, crs_use, timezone) {
  data <- read.csv(dataFilename)
  data <- formatData(data, crs_in, crs_use)

  # TODO: make sure these columns removed don't have illegal activity detected?
  # remove incorrect data (data at some time or multiple same positions)
  # these are due to data entry problems
  removeSameTimePatrol <- function(x, id) {
    if (any(table(x$DateTime[x$Patrol.ID==id]) > 4)) {
      x <- x[!x$Patrol.ID == id, ]
    }
    return(x)
  }

  removeSamePositionPatrol <- function(x, id) {
    if (any(table(x$Y[x$Patrol.ID==id]) > 6)) {
      x <- x[!x$Patrol.ID == id, ]
    }
    return(x)
  }

  for (i in unique(data$Patrol.ID)) {
    data <- removeSameTimePatrol(data, i)
    data <- removeSamePositionPatrol(data, i)
  }

  # elimate duplicates by adding 59 seconds to duplicated
  repeat {
    dups <- duplicated(data$DateTimePosix, data$ID_New)
    table(dups)
    data$DateTimePosix[dups] <- data$DateTimePosix[dups] + 59
    if (all(dups==FALSE)) {
      break
    }
  }
  
  # require that each trajectory has more than 3 sequential data points
  data <- data[data$ID_New %in% names(which(table(data$ID_New) > 3)), ]
  
  return(data)
}
data <- cleanData(dataFilename, crs_in, crs_use, timezone)


# helper function for trajectory
# remove points that are too far apart or have too little time elapsed
evaluateSplit <- function(dist, dt) {
  too_far  <- dist > 45000
  too_fast <- dist / dt > 90000 / (60*60)
  return(too_far | too_fast)
}
# evaluateSplit <- function(dist, dt) {
#   too_far  <- dist > 5000  # over 15km apart
#   too_fast <- dist / dt > 10000 / (60*60)  # 30km/hr
#   return(too_far | too_fast)
# }

# maxDist - maximum distance traveled between points (in km)
# maxSpeed - maximum speed traveled between points (in km/hr)
# evaluateSplit <- function(dist, dt, maxDist, maxSpeed) {
#   # RETURN FALSE IF DON'T WANT TO REMOVE ANY TRAJECTORIES
#   return(FALSE)
#   tooFar  <- dist > maxDist * 1000
#   tooFast <- dist / dt > maxSpeed * 1000 / (60*60)
#   return(tooFar | tooFast)
# }

# create trajectory
createTrajectory <- function(data) {
  # split up data to ease processing
  range <- seq(min(data$ID_New), max(data$ID_New), by=15)
  
  # the seq may omit values (if the by value doesn't evenly divide max-min), so we
  # change the last value of range to the max value
  range <- replace(range, length(range), max(data$ID_New))

  dataSplit <- split(data, cut(data$ID_New, range))
  dataSplit <- dataSplit[lapply(dataSplit, length) > 0]
  nSplits   <- length(dataSplit)

  # convert data to ltraj object, used to analyze animal movement
  trajectory <- vector(mode="list", length=nSplits)
  system.time(
    for (j in 1:nSplits) {
      trajectory[[j]] <- as.ltraj(coordinates(dataSplit[[j]]),
                                  date=dataSplit[[j]]$DateTimePosix,
                                  id=dataSplit[[j]]$ID_New)
      trajectory[[j]] <- cutltraj(trajectory[[j]], "evaluateSplit(dist, dt)", nextr=FALSE)
    }
  )
  
  outFilename <- paste(outFilepath, "trajectory.Rdata", sep="")
  save(trajectory, file=outFilename)

  return(trajectory)
}
trajectory <- createTrajectory(data)


# linesToRaster <- function(rasterGrid, lines, resolution) {
#   gridSP <- as(rasterGrid, "SpatialPolygons")
#   #W <- as(SP.win, "owin")
#   window <- as.owin(gridSP)  # observation window
#   linesRaster <- as.psp(lines)
#   
#   # calculate lengths per cell
#   # TODO: eps probably not doing anything
#   linesRaster <- pixellate.psp(linesRaster, window, eps=resolution)
#   
#   # convert pixel image to raster in km
#   linesRaster <- raster(linesRaster, crs=CRS(crs_use))
#   linesRaster <- linesRaster / 1000
#   
#   return(linesRaster)
# }

convertTrajectoryToLine <- function(trajectory, crs_use) {
  ### trajectories to lines (check counts per cell)
  allLines <- lapply(trajectory, ltraj2sldf) # create list of spatial lines
  patrolLines <- do.call(rbind, allLines)    # unlist to 1 spatial lines data frame
  proj4string(patrolLines) <- CRS(crs_use)

  # write to shapefile (in its own folder!)
  shapeFilepath <- paste(outFilepath, "patrol_lines", sep="")
  shapeFilename <- paste("patrol_lines", sep="")
  writeOGR(patrolLines, shapeFilepath, shapeFilename, driver="ESRI Shapefile", overwrite_layer=TRUE)

  return(patrolLines)
}
patrolLines <- convertTrajectoryToLine(trajectory, crs_use)



visualizePatrolLengths <- function(outFilepath, patrolLines, resolution, crs_use, boundary) {
  # convert to a line segment pattern object with maptools
  patrolPSP <- as.psp(patrolLines)

  # calculate lengths per cell
  patrolLengthAll <- pixellate.psp(patrolPSP, eps=resolution)
  # pixellate.psp(patrolPSP, W=NULL, eps=resolution, what="length", DivideByPixelArea=TRUE)

  # convert pixel image to raster in km
  patrolLengthAll <- raster(patrolLengthAll, crs=CRS(crs_use))
  patrolLengthAll <- patrolLengthAll / 1000
  
  # set out-of-bound cells and cells with zero effort to NA
  patrolLengthAll <- mask(patrolLengthAll, boundary)
  patrolLengthAll[patrolLengthAll == 0] <- NA

  # visualize patrol lengths
  png(file=paste(outFilepath, "patrol_effort.png", sep=""), width=800, height=600)
  plot(patrolLengthAll, col=heat_hcl(30, h=c(90, 0), c=c(30, 80), l=c(90, 30), power=1.5), main="Patrol effort", cex.main=2)
  lines(boundary)
  dev.off()
  
  return(patrolLengthAll)
}
patrolEffortAll <- visualizePatrolLengths(outFilepath, patrolLines, resolution, crs_use, boundary)


getPointsWithinBoundary <- function(gridFilename, crs_use) {
  rast <- loadGridRast(gridFilename, crs_use)
  
  # number points in raster and mask
  rasterGrid <- setValues(rast, 1:ncell(rast))
  rasterGridMask <- mask(rasterGrid, boundary)
  pointsWithinBoundary <- which(!is.na(values(rasterGridMask)))
  
  out <- list("rasterGrid"=rasterGrid, "pointsWithinBoundary"=pointsWithinBoundary)
  return(out)
}
out <- getPointsWithinBoundary(gridFilename, crs_use)
rasterGrid <- out$rasterGrid
pointsWithinBoundary <- out$pointsWithinBoundary


pointsToRaster <- function(data, rasterGrid, crs_use) {
  if (ncell(data) == 0) {
    emptyRaster <- rasterGrid
    values(emptyRaster) <- 0
    return (emptyRaster)
  }
  dataRaster <- rasterize(data, rasterGrid, field=1, fun="count", na.rm=FALSE)
  proj4string(dataRaster) <- CRS(crs_use)
  crs(dataRaster) <- crs_use
  
  dataRaster[is.na(dataRaster)] <- 0
  
  return(dataRaster)
}


# emphasize all discrete waypoints that were plotted
addDiscretePoints <- function(data, rasterGrid, month, year, crs_use) {
  dataInMonth <- data[which(data$Year == year & data$Month == month), ]
  dataInMonthRaster <- pointsToRaster(dataInMonth, rasterGrid, crs_use)
  values(dataInMonthRaster) <- .01 * values(dataInMonthRaster)
  return(dataInMonthRaster)
}


# patrol effort broken down by month
# compute effort as length per cell
savePatrolLengthMonth <- function(rasterGrid, data, trajectory, pointsWithinBoundary, crs_use, outFilepath) {
  # create empty array to store data per year and per month
  patrolLengthMonth <- array(0,  # initialize with 0
                            dim = c(ncell(rasterGrid), 
                                    12,  # months per year
                                    NROW(unique(data$Year))),  # number of years
                            dimnames =
                              list(NULL,
                                   sprintf("%02d", 1:12),
                                   sort(unique(data$Year))))
  
  SP.win <- as(rasterGrid, "SpatialPolygons")
  W <- as(SP.win, "owin")
  
  # create list of spatial lines object
  linesInTrajectory <- lapply(trajectory, ltraj2sldf)
  patrolLines <- do.call(rbind, linesInTrajectory)
  proj4string(patrolLines) <- CRS(crs_use)
  
  nSplits <- length(trajectory)
  for (j in 1:nSplits) {
    print(paste(j, "/", nSplits))
    patrol.dates <- aggregate(cbind(strftime(ld(trajectory[[j]])$date, "%Y"),
                                    strftime(ld(trajectory[[j]])$date, "%m")),
                              by = list(ld(trajectory[[j]])$burst),
                              FUN = function(x) x[1])
    lines <- linesInTrajectory[[j]]

    for (year in levels(as.factor(patrol.dates[,2]))) {
      for (month in levels(as.factor(patrol.dates[,3]))) {
        pointsInMonth <- patrol.dates[,2] == year & patrol.dates[,3] == month
        if (sum(pointsInMonth) > 0) {
          linesMonth <- lines[pointsInMonth,]
          proj4string(linesMonth) <- crs_use
          linesMonth <- as(linesMonth, "SpatialLines")

          # TODO: gives warning message. "1 columns of data frame discarded"
          patrolPSP <- as.psp(linesMonth)
          patrolLength <- pixellate.psp(patrolPSP, W, eps=resolution)
          patrolLength <- raster(patrolLength, crs=CRS(crs_use))
          patrolLength <- patrolLength / 1000  # convert m to km
          patrolLengthMonth[, month, year] <- patrolLengthMonth[, month, year] + values(patrolLength)
        }
      }
    }
  }
  
  # replace all NAs with 0
  patrolLengthMonth[is.na(patrolLengthMonth)] <- 0
  
  # remove points outside boundary
  patrolLengthMonthMask <- patrolLengthMonth[pointsWithinBoundary, , , drop=FALSE]
  
  # # ------------
  # # ADD POINTS - give all data points some small value in case they were removed by trajectory calculation
  # dataOriginal <- getData(dataFilename, crs_in, crs_use)
  # for (year in as.character(sort(unique(dataOriginal$Year)))) {
  #   for (month in sprintf("%02d", 1:12)) {
  #     pointsToAdd <- addDiscretePoints(dataOriginal, rasterGrid, month, year, crs_use)
  #     print(paste("adding points ", sum(values(pointsToAdd))))
  #     patrolLengthMonth[, month, year] <- patrolLengthMonth[, month, year] + values(pointsToAdd)[pointsWithinBoundary]
  #   }
  # }
  
  # write out monthly patrol effort
  outfilename <- paste(outFilepath, "patrol_month.csv", sep="")
  write.csv(patrolLengthMonthMask, outfilename, quote=FALSE)
  #write.csv(patrolLengthMonth, outfilenameCSV, na="0", quote=FALSE)
  
  return(patrolLengthMonth)
}
patrolLengthMonth <- savePatrolLengthMonth(rasterGrid, data, trajectory, pointsWithinBoundary, crs_use, outFilepath)


# patrol effort broken down by year
savePatrolLengthYear <- function(outFilepath, patrolLengthMonth, pointsWithinBoundary) {
  patrolLengthYear <- apply(patrolLengthMonth, c(1, 3), sum, na.rm=TRUE)
  patrolLengthYear <- as.data.frame(patrolLengthYear)
  
  # remove points outside boundary
  patrolLengthYearMask <- patrolLengthYear[pointsWithinBoundary, , drop=FALSE]

  outfilename <- paste(outFilepath, "patrol_year.csv", sep="")
  write.csv(patrolLengthYearMask, outfilename)
}
savePatrolLengthYear(outFilepath, patrolLengthMonth, pointsWithinBoundary)


# create combined visualization
visualizeEffortWithPoints <- function(outFilepath, patrolLengthMonth, boundary) {
  patrolLengthAll <- rowSums(patrolLengthMonth)
  patrolLengthAllRast <- rasterGrid
  values(patrolLengthAllRast) <- 0
  values(patrolLengthAllRast) <- patrolLengthAll
  
  png(file=paste(outFilepath, "plot_effort_as_points.png", sep=""),
      width=800, height=600)
  patrolLengthAllMask <- mask(patrolLengthAllRast, boundary)
  values(patrolLengthAllMask)[values(patrolLengthAllMask) == 0] <- NA # set '0' to NA to plot as white
  plot(patrolLengthAllMask, col=heat_hcl(30, h=c(90, 0), c=c(30, 80), l=c(90, 30), power=1.5))
  lines(boundary)
  title(main="Patrol effort", cex.main=2)
  dev.off()
}
visualizeEffortWithPoints(outFilepath, patrolLengthMonth, boundary)


