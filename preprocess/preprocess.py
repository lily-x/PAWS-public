""" calculate patrol effort and num visits
create a shapefile with each patrol as a separate line

Lily Xu
March 2020 """

# TODO:
# - process tiff by discretizing into slope
# - process animal count / survey data
# - process climate

import sys, os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from pyproj import Proj, transform
import rasterio, rasterio.features, rasterio.mask

import richdem as rd

# import fiona
# import shapefile as shp


col_id    = 'ID_New'  # NOTE: we can set this

col_x     = 'X'
col_y     = 'Y'
col_time  = 'Waypoint Time'
col_month = 'Month'
col_year  = 'Year'
col_date  = 'Waypoint Date'

col_patrol_id = 'Patrol ID'
col_patrol_end_date = 'Patrol End Date'
col_patrol_start_date = 'Patrol Start Date'

col_species = 'Species'
col_species_num = 'TOTAL number'


class Grid:
    def __init__(self, out_path, boundary, resolution, crs):
        self.boundary   = boundary
        self.resolution = resolution
        self.out_path   = out_path
        self.crs = crs
        self.nx         = None
        self.ny         = None
        self.centers    = None
        self.grid       = None
        self.transform  = None

        """ create shapefile grid from boundary """
        minx = float(boundary.bounds.minx)
        miny = float(boundary.bounds.miny)
        maxx = float(boundary.bounds.maxx)
        maxy = float(boundary.bounds.maxy)

        # make grid
        nx = int(math.ceil(abs(maxx - minx) / resolution))
        ny = int(math.ceil(abs(maxy - miny) / resolution))

        # track the x, y center of each grid cell
        centers = np.full((nx * ny, 2), np.nan)

        boxes = []
        for i in range(ny):
            for j in range(nx):
                vertices = [[min(minx + resolution*j, maxx), max(maxy - resolution*i, miny)],
                            [min(minx + resolution*(j+1), maxx), max(maxy - resolution*i, miny)],
                            [min(minx + resolution*(j+1), maxx), max(maxy - resolution*(i+1), miny)],
                            [min(minx + resolution*j, maxx), max(maxy - resolution*(i+1), miny)]]

                centers[nx*i+j, 0] = int((min(minx + resolution*j, maxx) + min(minx + resolution*(j+1), maxx)) / 2)
                centers[nx*i+j, 1] = int((max(maxy - resolution*i, miny) + max(maxy - resolution*(i+1), miny)) / 2)

                boxes.append(Polygon(vertices))

        grid_out_path = '{}/grid'.format(out_path)
        if not os.path.exists(grid_out_path):
            os.makedirs(grid_out_path)

        grid = gpd.GeoDataFrame(crs=crs, geometry=boxes)
        grid.to_file('{}/grid.shp'.format(grid_out_path))

        centers_df = pd.DataFrame(centers, columns=['x', 'y'])
        centers_df.to_csv('{}/grid_centers.csv'.format(grid_out_path))

        self.nx = nx
        self.ny = ny
        self.grid = grid
        self.centers = centers

        assert len(grid) > 1, 'Error! Grid has only one cell.'
        print('num cells {}, nx {}, ny {}'.format(len(grid), nx, ny))

        # west, south, east, north, width, height
        transform = rasterio.transform.from_bounds(minx, maxy - ny*resolution, minx + nx*resolution, maxy, nx, ny)
        self.transform = transform
        rast_array = self.shapefile_to_rast_array(boundary)
        self.save_rast_array_to_tif(rast_array, 'boundary')


    def categorical_to_rast_array(self, shapefile, col_name):
        """ given a shapefile, convert to np array that is
        0/1 of whether shapefile covers that grid """

        rast_array = rasterio.features.rasterize(
            [(shapefile.geometry[i], shapefile[col_name][i]) for i in range(len(shapefile))],
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            default_value=0,
            fill=0,
            all_touched=True,
            dtype=rasterio.uint8 #shapefile[col_name].dtype
        )

        return rast_array


    def shapefile_to_rast_array(self, shapefile):
        """ given a shapefile, convert to np array that is
        0/1 of whether shapefile covers that grid """

        rast_array = rasterio.features.rasterize(
            [(shape, 1) for shape in shapefile['geometry']],
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            default_value=1,
            fill=0,
            all_touched=True,
            dtype=rasterio.uint8
        )

        return rast_array


    def save_rast_array_to_tif(self, rast_array, name):
        """ given a np array of a rast, save to tiff image
        TODO: set dtype? """

        print('dtype', rast_array.dtype)

        with rasterio.open(
            '{}/rasterized-{}.tif'.format(self.out_path, name), 'w',
            driver='GTiff',
            dtype=rast_array.dtype,
            count=1,
            width=self.nx,
            height=self.ny,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(rast_array, indexes=1)


    def tif_to_rast_array(self, filename, name):
        """ process tif by discretizing (habitat: elevation, ...) """

        with rasterio.open(filename) as dataset:
            assert dataset.count == 1 # not sure what to do if there is more than 1
            assert dataset.crs == self.crs  # might no longer be necessary

            minx = float(self.boundary.bounds.minx)
            miny = float(self.boundary.bounds.miny)
            maxx = float(self.boundary.bounds.maxx)
            maxy = float(self.boundary.bounds.maxy)

            # crop to bounding rectangle
            shapes = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
            shapes = [Polygon(shapes)]

            mask_image, mask_transform = rasterio.mask.mask(dataset, shapes, all_touched=True, filled=False, crop=True)

            destination = np.zeros((self.ny, self.nx))
            out_rast, out_transform = rasterio.warp.reproject(
                mask_image,
                destination,
                src_transform=mask_transform,
                src_crs=dataset.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=rasterio.warp.Resampling.bilinear)

        return out_rast
        #     # resample data to target shape
        #     data = dataset.read(
        #         # window= ((0, 5000), (0, 5000)),
        #         out_shape=(
        #             dataset.count,
        #             self.ny,
        #             self.nx
        #         ),
        #         resampling=rasterio.enums.Resampling.bilinear
        #     )
        #
        # return data[0]




class Preprocess:
    def __init__(self, out_path, boundary, resolution, crs):
        self.out_path = out_path

        self.boundary = boundary.to_crs(crs)

        self.Grid = Grid(out_path, self.boundary, resolution, crs)
        self.grid = self.Grid.grid

        self.crs = crs

        self.num_cells = len(self.grid)
        self.cells_in_boundary = self.get_cells_in_boundary()

        # will be filled by clean_data
        self.data   = None
        self.months = None
        self.years  = None



    def clean_data(self, raw_data_filename, crs_in, start_year, end_year, only_foot=False):
        """ process raw patrol observations by removing out-of-bounds data """

        print('clean data...')

        out_filename = '{}/patrol_observations_clean.csv'.format(self.out_path)

        data = pd.read_csv(raw_data_filename, low_memory=False)
        print('  data shape start', data.shape)

        # transform boundary of CRS to input data
        boundary_temp = self.boundary.to_crs(crs_in)
        bounds = boundary_temp.bounds        # note this is a pandas Series

        print('  bounds', bounds.values[0])

        if only_foot == True:
            # use only foot patrols
            data = data.loc[data['Patrol Transport Type'] == 'Foot']

        # format dates correctly
        # pick date format
        # defaults: '%b %d, %Y', '%d-%b-%y', '%m/%d/%Y', '%Y-%b-%d'
        # format= .... infer_datetime_format=True
        data[col_patrol_end_date] = pd.to_datetime(data[col_patrol_end_date])
        data[col_patrol_start_date] = pd.to_datetime(data[col_patrol_start_date])
        data[col_date] = pd.to_datetime(data[col_date])

        # add year and month columns
        data[col_month] = data[col_date].dt.month
        data[col_year] = data[col_date].dt.year

        # remove dates beyond start and end years
        data = data[(data[col_year] >= start_year) & (data[col_year] <= end_year)]

        # remove out-of-bounds data
        data = data[~((data[col_x] == 0) & (data[col_y] == 0))]
        data = data[(data[col_x] >= float(bounds.minx)) &
                    (data[col_x] <= float(bounds.maxx)) &
                    (data[col_y] >= float(bounds.miny)) &
                    (data[col_y] <= float(bounds.maxy))]

        # if we have no points left, there's a problem
        assert len(data) > 0, 'Uh oh! Removed all data!'

        # create unique IDs, used in the trajectories
        # patrol IDs are given as strings, so we assign a unique numerical ID to each column
        patrol_id_int, _ = pd.factorize(data[col_patrol_id])
        delta = data[col_date] - data[col_patrol_start_date]
        patrol_day = list(delta.dt.days)
        data[col_id] = ['{}.{}'.format(patrol_id_int[i], patrol_day[i]) for i in range(len(patrol_id_int))]

        # transform X and Y to intended CRS
        in_proj = Proj(crs_in)
        out_proj = Proj(self.crs)
        # transform_x, transform_y = transform(in_proj, out_proj, data[col_x].values, data[col_y].values)

        print('x, y', data[col_y].values, data[col_x].values)

        # note: empirically i have to swap x and y...
        transform_x, transform_y = transform(in_proj, out_proj, data[col_y].values, data[col_x].values)
        data[col_x] = transform_x
        data[col_y] = transform_y

        data.to_csv(out_filename)
        print('  data shape after', data.shape)

        # transform datetime fields to string
        data[col_patrol_end_date] = data[col_patrol_end_date].dt.strftime('%Y-%m-%d')
        data[col_patrol_start_date] = data[col_patrol_start_date].dt.strftime('%Y-%m-%d')
        data[col_date] = data[col_date].dt.strftime('%Y-%m-%d')

        self.data = data
        self.months = sorted(set(data[col_month]))
        self.years  = sorted(set(data[col_year]))
        self.num_months = len(self.months)
        self.num_years  = len(self.years)

        return data


    def get_splits(self, rows, time_format, max_dist=5, max_speed=10):
        """ split lines if waypoints are too far apart """

        all_x = []
        all_y = []

        x_vals = []
        y_vals = []

        x_vals.append(rows[col_x].iloc[0])
        y_vals.append(rows[col_y].iloc[0])

        # split trajectory if too far or too fast
        for i in range(len(rows) - 1):
            x1 = rows[col_x].iloc[i]
            x2 = rows[col_x].iloc[i+1]
            y1 = rows[col_y].iloc[i]
            y2 = rows[col_y].iloc[i+1]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            time1 = datetime.strptime(rows[col_time].iloc[i], time_format)
            time2 = datetime.strptime(rows[col_time].iloc[i+1], time_format)
            dt = (time2 - time1).total_seconds()

            # avoid same-time waypoints
            if dt == 0:
                continue

            too_far = dist > (max_dist*1000)                     # over X km apart
            too_fast = dist / dt > (max_speed*1000) / (60 * 60)  # over X km/hr

            if too_far or too_fast:
                # split into new line series
                all_x.append(x_vals)
                all_y.append(y_vals)
                x_vals = []
                y_vals = []

            x_vals.append(x2)
            y_vals.append(y2)

        all_x.append(x_vals)
        all_y.append(y_vals)

        return all_x, all_y


    def make_lines(self, time_format):
        """ create shapefile vector lines from patrols """

        print('make lines...')

        patrol_IDs = set(self.data[col_id])
        lines_shape = []
        lines_attrib = []

        for id in patrol_IDs:
            rows = self.data.loc[self.data[col_id] == id]

            # must have at least 2 waypoints
            if len(rows) < 2:
                continue

            all_x, all_y = self.get_splits(rows, time_format=time_format)
            for i in range(len(all_x)):
                if len(all_x[i]) < 2:
                    continue
                # print(id, x_vals, y_vals)
                line = LineString([(all_x[i][j], all_y[i][j]) for j in range(len(all_x[i]))])
                lines_shape.append(line)

                lines_attrib.append([rows[col_year].iloc[0], rows[col_month].iloc[0],
                        rows[col_date].iloc[0], rows[col_id].iloc[0]])


        lines_df = pd.DataFrame(lines_attrib, columns=['year', 'month', 'date', 'patrol_id'])

        lines = gpd.GeoDataFrame(lines_df, crs=self.crs, geometry=lines_shape)
        if not os.path.exists('{}/lines'.format(self.out_path)):
            os.makedirs('{}/lines'.format(self.out_path))
        lines.to_file('{}/lines/lines.shp'.format(self.out_path))

        return lines


    def calculate_effort(self, time_format='%H:%M:%S'):
        """ count number of visits and km traveled in each target """

        lines = self.make_lines(time_format)

        print('calculate effort...')
        num_visits = np.zeros((self.num_cells, self.num_months*self.num_years))
        effort = np.zeros((self.num_cells, self.num_months*self.num_years))

        lines_by_month = {}
        for year in self.years:
            lines_by_month[year] = {}
            for month in self.months:
                lines_by_month[year][month] = []

        for i in range(len(lines)):
            year = lines['year'][i]
            month = lines['month'][i]
            lines_by_month[year][month].append(i)

        t = 0
        columns = []

        for year in self.years:
            for month in self.months:
                print('  {}-{}'.format(year, month))
                columns.append('{}-{}'.format(year, month))
                for j in lines_by_month[year][month]:
                    for i in self.cells_in_boundary:
                        box = self.grid.geometry[i]

                        # does it intersect?
                        if box.intersects(lines.geometry[j]):
                            # find intersection with path and compute length
                            intersect = box.intersection(lines.geometry[j])
                            length = intersect.length

                            effort[i, t] += length / 1000  # convert m to km
                            num_visits[i, t] += 1

                t += 1

        num_visits_df = pd.DataFrame(num_visits, columns=columns)
        effort_df = pd.DataFrame(effort, columns=columns)
        self.save_df_to_csv(num_visits_df, 'patrol_effort_visits')
        self.save_df_to_csv(effort_df, 'patrol_effort')

        self.save_map_png(effort.sum(axis=1), 'patrol_effort', cmap='Greens')
        self.save_map_png(num_visits.sum(axis=1), 'num_visits', cmap='Greens')


    def compute_illegal_activity(self, isolate_illegal_activity):
        """ count number of instances of illegal activity in each target """
        print('compute illegal activity...')

        num_instances = np.zeros((self.num_cells, self.num_months*self.num_years))

        # TODO: multiple types of illegal activity?

        data_illegal = isolate_illegal_activity(self.data)

        for i in self.cells_in_boundary:
            minx, miny, maxx, maxy = self.grid.geometry[i].bounds

            relevant_points = data_illegal.loc[
                (minx <= data_illegal[col_x]) & (data_illegal[col_x] < maxx) &
                (miny <= data_illegal[col_y]) & (data_illegal[col_y] < maxy)]

            t = 0
            for year in self.years:
                for month in self.months:
                    month_points = relevant_points.loc[(relevant_points[col_year] == year) &
                                    (relevant_points[col_month] == month)]

                    # if len(relevant_points) > 0:
                    num_instances[i, t] = len(month_points)

                    t += 1

        num_instances_df = pd.DataFrame(num_instances, columns=['{}-{}'.format(y, m) for y in self.years for m in self.months])
        self.save_df_to_csv(num_instances_df, 'illegal_activity')

        self.save_map_png(num_instances.sum(axis=1), 'illegal_activity', cmap='Reds')


    def get_distances(self, shapes):
        """ compute discretized distance to shapefiles """

        print('compute distances...')

        dists = {}

        for k, name in enumerate(shapes):
            shape = shapes[name]

            # ensure shapes are the right CRS
            shape = shape.to_crs(self.crs)

            # for boundary, use only exterior (not closed polygon)
            if name == 'boundary':
                shape = shape.exterior

            dist = np.full(self.num_cells, np.nan)
            for i in self.cells_in_boundary:
                box = self.grid.geometry[i]

                # compute distance to shapefile
                dist[i] = shape.distance(box).min()

            print(name)
            print(dist)

            dists[name] = dist

            # save map to image
            self.save_map_png(dist, name)

        dists_df = pd.DataFrame(dists)
        self.save_df_to_csv(dists_df, 'distances')

        return dists


    def compute_animal_count(important_species):
        # TODO: smooth it out? e.g.. 5x5 or 10x10
        """ compute animal density from patrol observations
        process animal count / survey data
        """
        print('compute animal count...')

        num_instances = np.zeros((self.num_cells, len(important_species)))

        data_wildlife = self.data.loc[self.data['Observation Category  0'] == 'Wildlife']

        col_names = []

        for j, species in enumerate(important_species):
            short_name = important_species[species]
            col_names.append(short_name)

            print('  species {}'.format(species))
            data_species = data_wildlife.loc[(data_wildlife[col_species] == species)]
            print('  > num points', len(data_species))
            for i in self.cells_in_boundary:
                minx, miny, maxx, maxy = self.grid.geometry[i].bounds

                relevant_points = data_wildlife.loc[
                    (minx <= data_wildlife[col_x]) & (data_wildlife[col_x] < maxx) &
                    (miny <= data_wildlife[col_y]) & (data_wildlife[col_y] < maxy) &
                    (data_wildlife[col_species] == species)]

                species_count = np.sum(relevant_points[col_species_num])
                # species_count = len(relevant_points[col_species_num])

                num_instances[i, j] = species_count

            self.save_map_png(num_instances[:, j], short_name)

        num_instances_df = pd.DataFrame(num_instances, columns=col_names)
        self.save_df_to_csv(num_instances_df, 'animal_count')



    def compute_species_over_time(self, species, short_name=None):
        """ compute animal count from patrol observations, over time
        """
        print('compute species count over time - {}...'.format(species))

        if short_name is None:
            short_name = species

        num_instances = np.zeros((self.num_cells, self.num_months*self.num_years))

        data_species = self.data.loc[(self.data['Observation Category  0'] == 'Wildlife') &
                                  (self.data[col_species] == species)]
        print('  > num points', len(data_species))
        for i in self.cells_in_boundary:
            minx, miny, maxx, maxy = self.grid.geometry[i].bounds

            relevant_points = data_species.loc[
                (minx <= data_species[col_x]) & (data_species[col_x] < maxx) &
                (miny <= data_species[col_y]) & (data_species[col_y] < maxy) &
                (data_species[col_species] == species)]

            t = 0
            for year in self.years:
                for month in self.months:
                    month_points = relevant_points.loc[(relevant_points[col_year] == year) &
                                    (relevant_points[col_month] == month)]

                    # num_instances[i, t] = len(month_points)
                    num_instances[i, t] = np.sum(month_points[col_species_num])

                    t += 1

        num_instances_df = pd.DataFrame(num_instances, columns=['{}-{}'.format(y, m) for y in self.years for m in self.months])
        self.save_df_to_csv(num_instances_df, 'animal_count_{}'.format(short_name))
        self.save_map_png(num_instances.sum(axis=1), short_name)


    def feature_categorical(self, filename, name, col_name):
        """ process categorial features (e.g., forest cover, land use)
        given a shapefile with categorical labels

        name: name to save out as
        col_name: column name of categorical label"""

        print('process categorical feature - {}...'.format(name))

        shapefile = gpd.read_file(filename)
        shapefile.to_crs(self.crs)

        # ensure that categorical feature is represented as a number
        shapefile[col_name], factor_map = pd.factorize(shapefile[col_name])

        rast_array = self.Grid.categorical_to_rast_array(shapefile, col_name)
        self.Grid.save_rast_array_to_tif(rast_array, name)

        df = pd.DataFrame({name: rast_array.flatten()})
        self.save_df_to_csv(df, name)

        self.save_map_png(rast_array, name)


    def feature_tif(self, filename, name):
        print('process tif - {}...'.format(name))

        rast_array = self.Grid.tif_to_rast_array(filename, name)

        # might have values way negative
        rast_array[rast_array < 0] = 0
        # TODO: get the resampling to get closest non-crazy number around the boundary?

        self.Grid.save_rast_array_to_tif(rast_array, name)
        df = pd.DataFrame({name: rast_array.flatten()})
        self.save_df_to_csv(df, name)

        self.save_map_png(rast_array, name)


    def get_climate(self):
        """ TODO: get climate data """
        pass

    def get_slope(self):
        """ TODO """

        beau  = rd.rdarray(np.load('imgs/beauford.npz')['beauford'], no_data=-9999)
        slope = rd.TerrainAttribute(beau, attrib='slope_riserun')
        rd.rdShow(slope, axes=False, cmap='jet', figsize=(8,5.5))

        pass


    ####################################################
    # utility functions
    ####################################################

    def get_cells_in_boundary(self):
        cells_in_boundary = []
        for i in range(self.num_cells):
            box = self.grid.geometry[i]
            if self.boundary.intersects(box)[0]:
                cells_in_boundary.append(i)

        return cells_in_boundary

    def save_df_to_csv(self, df, name):
        # add x and y columns to beginning
        df.insert(0, 'x', self.Grid.centers[:, 0])
        df.insert(1, 'y', self.Grid.centers[:, 1])

        # remove out-of-bounds rows
        df = df.loc[self.cells_in_boundary]

        df.to_csv('{}/{}.csv'.format(self.out_path, name))


    def save_map_png(self, map, name, cmap=None, vmin=None, vmax=None):
        """ map is array of values for each cell
        create a 2D array with values
        then save as png image
        """
        if map.ndim == 2:
            map = map.flatten()

        # set cells outside boundary to nan
        masked_map = np.full(self.Grid.ny * self.Grid.nx, np.nan)
        masked_map[self.cells_in_boundary] = map[self.cells_in_boundary]

        masked_map = masked_map.reshape((self.Grid.ny, self.Grid.nx))

        plt.figure()
        plt.imshow(masked_map, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(name)
        plt.colorbar()
        plt.savefig('{}/map_{}.png'.format(self.out_path, name), dpi=150, bbox_inches='tight')
