# PAWS preprocessing - create grid for discretization
# Lily Xu
# December 2018

# create grids.rds file for the given resolution

library(rgdal)
library(raster)

gridFilename <- paste(path, outputPath, "grid.rds", sep="")


createGrid <- function(boundary, outFilename, resolution) {
  gridExtents <- extent(boundary)
  
  # create a grid to match major UTM coordiantes
  dim1 <- ceiling((
      ceiling(xmax(gridExtents) / resolution) * resolution - 
        floor(xmin(gridExtents) / resolution) * resolution) 
    / resolution)
  dim2 <- ceiling((
      ceiling(ymax(gridExtents) / resolution) * resolution - 
        floor(ymin(gridExtents) / resolution) * resolution)
    / resolution)
  
  grid <- GridTopology(
    cellcentre.offset = c(
      floor(xmin(gridExtents) / resolution) * resolution, 
      floor(ymin(gridExtents) / resolution) * resolution),
    cellsize = c(resolution, resolution), 
    cells.dim = c(dim1, dim2))
  
  # save grid output
  saveRDS(grid, file=outFilename)
  
  # return(grid)
}

createGrid(boundary, gridFilename, resolution)


# rasterize grid
loadGridRast <- function(gridFilename, crs_use) {
  # load the grid to match major UTM coordinates
  grid <- readRDS(gridFilename)
  grid <- SpatialGrid(grid)
  proj4string(grid) <- CRS(crs_use)

  # convert to raster
  rast <- raster(grid)
  
  # # load the grid to match major UTM coordinates
  # grid <- readRDS(gridFilename)
  # 
  # # change to spatial pixels
  # dataGridPixels <- as(SpatialGrid(grid), "SpatialPixels")
  # proj4string(dataGridPixels) <- CRS(crs_use)
  # dataGridPixels <- spTransform(dataGridPixels, CRS(crs_use))
  # 
  # rast <- as(dataGridPixels, "SpatialPixels")
  # fullgrid(rast) <- FALSE
  # rast <- raster(rast)
  # proj4string(rast) <- CRS(crs_use)
  
  return(rast)
}


# create spatial points
loadGridSpatialPixels <- function(gridFilename, crs_use) {
  # load the grid to match major UTM coordinates
  grid <- readRDS(gridFilename)
  
  # change to spatial pixels
  dataGridPixels <- as(SpatialGrid(grid), "SpatialPixels")
  proj4string(dataGridPixels) <- CRS(crs_use)
  dataGridPixels <- spTransform(dataGridPixels, CRS(crs_use))
  
  return(dataGridPixels)
}

