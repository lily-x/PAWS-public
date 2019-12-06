import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon


###########################################################
# utility functions
###########################################################

# np.ndenumerate for ndarray with mask
def maenumerate(marr):
    mask = ~marr.mask.ravel()
    for i, m in zip(np.ndenumerate(marr), mask):
        if m: yield i


# convert a map in an ndarray to a GeoDataFrame object
def map_to_geodataframe(map, x_min, y_min):
    origin = (x_min, y_min)

    # note that (y, x) are flipped due to (row, col) of matrix
    # map has mask
    opened = [(x, y, value) for (y, x), value in maenumerate(map)]

    # TODO: can i change this to 'x' and 'y'? (for consistency)
    map_df = pd.DataFrame({
        'X': [tup[0] for tup in opened],
        'Y': [tup[1] for tup in opened],
        'value': [tup[2] for tup in opened]
    })

    map_df['Coordinates'] = list(zip(map_df['X'], map_df['Y']))
    map_df['Coordinates'] = map_df['Coordinates'].apply(Point)

    map_gdf = gpd.GeoDataFrame(map_df, geometry='Coordinates')

    return map_gdf


# given a masked map, create a GeoDataFrame with 1x1 blocks for each cell
# using the assigned value
def map_to_color_grid(map):
    polygons = []
    values = []
    for (y, x), value in maenumerate(map):
        # map UTM corresponds with center of cell
        polygons.append(Polygon([(x-.5, y-.5), (x+.5, y-.5),
                                 (x+.5, y+.5), (x-.5, y+.5)]))
        values.append(value)

    grid = gpd.GeoDataFrame({'value': values, 'geometry': polygons})
    return grid


# convert points to a GeoDataFrame, drawing a larger block with a specified kernel width
# small_grid: option to draw subgrid
def points_to_grid(points, kernel_width, small_grid=True):
    assert type(kernel_width) is int

    polygons = []
    for (y, x), value in maenumerate(points):
        if kernel_width % 2 == 1:       # odd
            min_val = -(kernel_width // 2)
            max_val = kernel_width // 2 + 1
        else:                           # even
            min_val = -(kernel_width / 2) + 1
            max_val = kernel_width / 2 + 1

        # show 1x1km discretization
        if small_grid:
            small_polygons = []
            for i in range(min_val, max_val):
                for j in range(min_val, max_val):
                    small_polygons += [(x+j, y+i), (x+j+1, y+i), (x+j+1, y+i+1), (x+j, y+i+1), (x+j, y+i)]
                small_polygons += [(x+min_val, y+i)]    # prevent diagonal line in polygon
            polygons.append(Polygon(small_polygons))

        else:
            polygons.append(Polygon([(x+min_val, y+min_val), (x+max_val, y+min_val), (x+max_val, y+max_val), (x+min_val, y+max_val)]))


    grid = gpd.GeoDataFrame({'geometry': polygons})
    return grid
