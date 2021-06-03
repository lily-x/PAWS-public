# PAWS preprocessing - habitat
# Lily Xu
# December 2018

# few important functions
#aggregate(),setextent(),resample()


### setup
library(sp)
library(raster)
library(rgdal)
library(gdalUtils)
library(maptools)

outFilepath  <- paste(path, outputPath, "habitat/", sep="")

# create 'habitat' folder if it does not exist
if (!dir.exists(outFilepath)) {
  dir.create(outFilepath)
}

#if you have HDF file and want to convert it to .tif file 
#sds <- get_subdatasets("MCD12Q1.A2013001.h18v08.051.2014308191113.hdf")
#gdal_translate(sds[1],dst_dataset = "anything.tif")


visualizeAndSave <- function(outFilepath, raster, filename, boundary) {
  # mask tif according to boundary
  raster <- mask(raster, boundary)
  
  # visualize
  png(file=paste(outFilepath, "plot_", filename, ".png", sep=""),
      width=800, height=600)
  plot(raster, main=filename)
  lines(boundary)
  dev.off()
  
  rasterCSV <- rasterToPoints(raster)
  rasterCSV <- as.data.frame(rasterCSV)
  
  # ensure x, y values are ints, not floats
  rasterCSV["x"] <- round(rasterCSV["x"], 0)
  rasterCSV["y"] <- round(rasterCSV["y"], 0)
  
  write.csv(rasterCSV, file=paste(outFilepath, filename, ".csv", sep=""))
}


calculateSlope <- function(elevationMapFilename, rast, crs_use, outFilepath) {
  # read tif file and boundary
  elevationMap <- raster(elevationMapFilename)
  elevationMap <- projectRaster(elevationMap, crs=crs_use, method="bilinear")
  
  slopeMap <- terrain(elevationMap, opt="slope")
  
  # resample it to get the same dimensions
  slopeMap <- resample(slopeMap, rast, method="bilinear")
  
  # set any NA values to 0
  slopeMap[is.na(slopeMap)] <- 0
  
  visualizeAndSave(outFilepath, slopeMap, "slope_map", boundary)
}


# discretize provided image, like elevation map or geoTIFF
discretizeImage <- function(imageFilename, name, boundary, rast, crs_use) {
  # read tif file
  image <- raster(imageFilename)
  image <- projectRaster(image, crs=crs_use, method="bilinear")
  
  # resample it to get the same dimensions
  image <- resample(image, rast, method="bilinear")
  
  visualizeAndSave(outFilepath, image, name, boundary)
}


# used to read elevation map, Google Earth Engine GeoTIFF output (like GPP)
discretizeImageNoSave <- function(imageFilename, boundary, rast, crs_use) {
  # read tif file
  image <- raster(imageFilename)
  image <- projectRaster(image, crs=crs_use, method="bilinear")
  
  # resample it to get the same dimensions
  image <- resample(image, rast, method="bilinear")
  
  # mask tif according to boundary
  image <- mask(image, boundary)
  
  imageDataFrame <- as.data.frame(image, xy=TRUE)
  imageDataFrame <- na.omit(imageDataFrame)
  
  # # ensure x, y values are ints, not floats
  # imageDataFrame["x"] <- round(imageDataFrame["x"], 0)
  # imageDataFrame["y"] <- round(imageDataFrame["y"], 0)
  
  return(imageDataFrame)
}


# extract month-by-month data from GeoTIFF images
getMonthlyGeotiff <- function(boundary, rast, start_year, end_year, crs_use) {
  for (year in start_year:end_year) {
    # # last year may not be complete?
    # if (year == end_year) {
    #   months = 1:end_month
    # } else {
    #   months = 1:12
    # }
    for (month in 1:12) {
      imageFilename <- print(sprintf("~/Downloads/large_GPP/GPP_%04d_%d.tif", year, month))
      
      image <- discretizeImageNoSave(imageFilename, boundary, rast, crs_use)
      
      # if first iteration
      if (year == start_year & month == 1) {
        allMonths <- image[,ncol(1:2)]
      }
      
      # last column of data frame
      allMonths[sprintf("%04d_%02d", year, month)] <- image[,ncol(image)]
    }
  }
  
  write.csv(allMonths, paste(outFilepath, "GPP.csv", sep=""))
}


# generally we'd always use slope, not elevation map
discretizeElevationMap <- function(elevationMapFilename, boundary, rast, crs_use) {
  discretizeImage(elevationMapFilename, "elevation_map", boundary, rast, crs_use)
}


discretizeForestCover <- function(forestCover, rast, outFilepath, crs_use) {
  forestCover <- spTransform(forestCover, CRS(crs_use))
  
  # background=0 will set missing values (gaps) to 0
  forestCoverRaster <- rasterize(forestCover, rast, "NEW_FCODE", background=0)
  
  visualizeAndSave(outFilepath, forestCoverRaster, "forest_cover", boundary)
}


rast <- loadGridRast(gridFilename, crs_use)
calculateSlope(elevationMapFilename, rast, crs_use, outFilepath)
discretizeElevationMap(elevationMapFilename, boundary, rast, crs_use)
discretizeForestCover(forestCover, rast, outFilepath, crs_use)
