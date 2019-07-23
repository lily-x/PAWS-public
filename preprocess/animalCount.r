# PAWS preprocessing - human activity
# Lily Xu
# December 2018

# libraries needed
library(rgdal)
library(raster)
library(colorspace)
#library(spatstat)


#animalDataFilename <- paste(path, outputPath, "patrolObservation_NewIDs.csv", sep="")
animalDataFilename <- paste(path, inputPath, "animals/animal_data_SWS.csv", sep="")

outFilepath  <- paste(path, outputPath, "animal_count/", sep="")

# create folder if it does not exist
if (!dir.exists(outFilepath)) {
  dir.create(outFilepath)
}


processAnimalData <- function(animalDataFilename, crs_in, crs_use) {
  data <- read.csv(animalDataFilename)
  
  # grid as used previously:
  coordinates(data) <- data[,c("X", "Y")] # convert to a spatialpoints* object
  proj4string(data) <- CRS(crs_in)
  data <- spTransform(data, CRS(crs_use))
  return(data)
}


# save as CSV
saveRasterAsCSV <- function(raster, outFilepath, filenamePrefix) {
  shapeCSV <- as.data.frame(raster, xy=TRUE)
  
  # ensure x, y values are ints, not floats
  shapeCSV["x"] <- round(shapeCSV["x"], 0)
  shapeCSV["y"] <- round(shapeCSV["y"], 0)
  
  shapeCSV <- na.omit(shapeCSV)
  shapeFilename <- paste(outFilepath, filenamePrefix, ".csv", sep="")
  write.csv(shapeCSV, file=shapeFilename)
}


# create visualization of all classes of illegal activities
visualizeClasses <- function(outFilepath, rast, data, boundary) {
  numCols <- ceiling(length(unique(data$importantAnimal)) / 2)
  
  png(file=paste(outFilepath, "plot_animals.png", sep=""), width=numCols*450, height=900)
  par(mfrow=c(2, numCols))
  
  animalClasses <- list()
  # add plot of all illegal activities
  for (animal in unique(data$importantAnimal)) {
    animalRast <- rasterize(data[data$importantAnimal == animal,], rast, field="importantAnimal", fun="count", na.rm=T)
    
    values(animalRast)[is.na(values(animalRast))] <- 0
    
    # convert raster to im, blur, then convert im to raster
    #animalRast <- raster(blur(as.im(animalRast), sigma=100, bleed=TRUE, normalize=TRUE))
    
    # convolve with a 5x5 Gaussian kernel
    filter <- c(1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1)
    filter = matrix(filter, nrow=5, ncol=5, byrow=TRUE)
    filter <- filter / 256
    
    animalRast <- focal(animalRast, w=filter)
    
    
    # mask to park boundary
    values(animalRast)[is.na(values(animalRast))] <- 0
    animalRast <- mask(animalRast, boundary)
    
    saveRasterAsCSV(animalRast, outFilepath, animal)
    
    #values(animalRast)[values(animalRast) == 0] <- NA # set '0' to NA to plot as white
    plot(animalRast, col=rev(sequential_hcl(10)))
    title(main=animal, cex.main=2)
    lines(boundary)
    
    animalClasses[[animal]] <- animalRast
  }
  dev.off()
  
  return(animalClasses)
}



# create combined visualization of all activities
visualizeAllAnimals <- function(data, rast, boundary, outFilepath) {
  # set everything outside boundary to NA
  # use the dataGridPixels as the grid
  animalsAll <- rasterize(data, rast, field="importantAnimal", fun="count", na.rm=T)
  values(animalsAll)[is.na(values(animalsAll))] <- 0
  
  # convolve with a 5x5 Gaussian kernel
  filter <- c(1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1)
  filter = matrix(filter, nrow=5, ncol=5, byrow=TRUE)
  filter <- filter / 256
  
  animalsAll <- focal(animalsAll, w=filter)
  
  # mask to park boundary
  values(animalsAll)[is.na(values(animalsAll))] <- 0
  animalsAllMasked <- mask(animalsAll, boundary)
  
  saveRasterAsCSV(animalsAllMasked, outFilepath, "all_animals")
  
  png(file=paste(outFilepath, "plot_all_animals.png", sep=""), 
      width=800, height=600)
  #values(animalsAllMasked)[values(animalsAllMasked) == 0] <- NA # set '0' to NA to plot as white
  plot(animalsAllMasked, col=rev(sequential_hcl(10)))
  lines(boundary)
  title(main="All animal species", cex.main=2)
  dev.off()
}


rast <- loadGridRast(gridFilename, crs_use)

# NOTE: in this file for SWS, coordinates come in as crs_use, not crs_in
data <- processAnimalData(animalDataFilename, crs_use, crs_use)

data <- findAnimalObservations(data)
animalClasses <- visualizeClasses(outFilepath, rast, data, boundary)
visualizeAllAnimals(data, rast, boundary, outFilepath)

