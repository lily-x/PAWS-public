# PAWS preprocessing - distance
# Lily Xu
# December 2018

### setup
library(sp)
library(rgeos)
library(rgdal)
library(raster)
library(maptools)

outFilepath  <- paste(path, outputPath, "distance/", sep="")

# create 'distance' folder if it does not exist
if (!dir.exists(outFilepath)) {
  dir.create(outFilepath, recursive=TRUE)
}


# boundary needs its own function because the park boundary shapefile is a filled polygon
# so it would say distance to the polygon is 0 everywhere
computeDistanceToBoundary <- function(dataGridPixels, rast, boundary, outFilepath) {
  distance <- numeric(ncell(rast))
  for (i in seq_len(ncell(rast))) {
    distance[i] <- gDistance(dataGridPixels[i,], as(boundary, "SpatialLines"), byid=TRUE)
  }
  values(rast) <- distance
  
  # set everything outside of the protected area to NA
  rasterMask <- mask(rast, boundary)
  rasterCSV <- as.data.frame(rasterMask, xy=TRUE)
  
  # ensure x, y values are ints, not floats
  rasterCSV["x"] <- round(rasterCSV["x"], 0)
  rasterCSV["y"] <- round(rasterCSV["y"], 0)
  
  rasterMaskCSV <- na.omit(rasterCSV)
  rasterFilename <- paste(outFilepath, "/boundary.csv", sep="")
  write.csv(rasterMaskCSV, file=rasterFilename)
  
  return(rasterMask)
}


computeDistanceToShapeMask <- function(dataGridPixels, rast, shape, boundary, outFilepath, filenamePrefix) {
  # compute distance of each grid cell to the shape
  distance <- numeric(ncell(rast))
  for (i in seq_len(ncell(rast))) {
    distance[i] <- gDistance(dataGridPixels[i,], shape)
  }
  values(rast) <- distance
  
  # maybe just rast[i] <- gDistance? (no need to make distance)
  
  # set everything outside of the protected area to NA
  rasterMask <- mask(rast, boundary)
  shapeCSV <- as.data.frame(rasterMask, xy=TRUE)
  
  # ensure x, y values are ints, not floats
  shapeCSV["x"] <- round(shapeCSV["x"], 0)
  shapeCSV["y"] <- round(shapeCSV["y"], 0)
  
  shapeCroppedCSV <- na.omit(shapeCSV)
  shapeCroppedFilename <- paste(outFilepath, filenamePrefix, ".csv", sep="")
  write.csv(shapeCroppedCSV, file=shapeCroppedFilename)
  
  return(rasterMask)
}



computeAllDistances <- function(boundary, shapes, dataGridPixels, rast, outFilepath) {
  shapeNames <- names(shapes)
  shapeDistances <- list()
  
  # distance to boundary
  shapeDistances["boundary"] <- computeDistanceToBoundary(dataGridPixels, rast, boundary, outFilepath)
  
  # distance to each additional shape
  for (i in 1:length(shapes)) {
    print(paste("processing", i, ", ", shapeNames[i], "..."))
    shape <- shapes[[i]]
    
    shapeDist <- computeDistanceToShapeMask(dataGridPixels, rast, shape, boundary, outFilepath, shapeNames[i])
    
    shapeDistances[shapeNames[i]] <- shapeDist
  }
  
  return(shapeDistances)
}

# TODO: switch to stack of rasters, then use writeRaster with bylayer=TRUE


# save PNG visualizing all distance maps
visualizeDistance <- function(outFilepath, shapeDistances, boundary) {
  shapeNames <- names(shapeDistances)
  numCols <- ceiling(length(shapeDistances) / 2)
  
  png(file=paste(outFilepath, "plot_distances.png", sep=""), width=450*numCols, height=900)
  par(mfrow=c(2, numCols))
  
  for (i in 1:length(shapeDistances)) {
    print(paste("processing", i, ", ", shapeNames[i], "..."))
    
    plot(shapeDistances[[i]], main=shapeNames[i], cex.main=2)
    lines(boundary)
  }
  
  dev.off()
}


rast <- loadGridRast(gridFilename, crs_use)
dataGridPixels <- loadGridSpatialPixels(gridFilename, crs_use)

shapes <- getDistanceShapes(inputPath)
shapeDistances <- computeAllDistances(boundary, shapes, dataGridPixels, rast, outFilepath)
visualizeDistance(outFilepath, shapeDistances, boundary)

