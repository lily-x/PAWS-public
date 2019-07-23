import numpy as np
from scipy import ndimage
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import shapefile
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utility import *


class Visualize:
    def __init__(self, resolution, input_filepath, output_filepath):
        self.resolution = resolution
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.gridmap = None
        self.x_min = None
        self.y_min = None

    def set_patrol_posts(self, patrol_post_filename):
        patrol_posts = pd.read_csv(patrol_post_filename)
        self.patrol_posts = self.shift_data_coords(patrol_posts)



    ###########################################################
    # utility
    ###########################################################

    def mask_to_grid(self, map):
        return np.ma.masked_where(self.gridmap==0, map)

    # for the gridmap, return list of indices of valid cells
    # order: begin from top left, then go row-by-row
    def get_indices_from_gridmap(self):
        # need this complicated way to compute corresponding indices
        # within the gridmap boundary because numpy indexing
        # starts at lower left corner, and the CSV file assumes
        # ordering starts in top left corner
        idx = [[], []]
        for y in range(self.gridmap.shape[0] - 1, -1, -1):
            add_idx = np.where(self.gridmap[y, :] == 1)
            idx[0] += [y] * add_idx[0].shape[0]
            idx[1] += list(add_idx[0])

        return tuple(idx)

    # get list of np.arrays, where each is a map of predicted risk`
    # at a different threshold of patrol effort
    def get_map_from_csv(self, filename):
        data = pd.read_csv(filename)
        print('  creating map from file {}'.format(filename))

        # discard first column: index of grid cell
        data.drop(data.columns[0], axis=1, inplace=True)
        if data.shape[1] > 1:
            raise Exception('ambiguous input: filename {} has more than one value column'.format(filename))

        idx = self.get_indices_from_gridmap()

        map = np.zeros(self.gridmap.shape)
        map[idx] = data.values[:,0]

        return map

    # maps is a dictionary of {map_name : map}
    def save_maps_to_csv(self, filename_out, maps):
        idx = self.get_indices_from_gridmap()

        map_names = list(maps.keys())

        data = {'x_idx': idx[1], 'y_idx': idx[0]}

        for i in range(len(maps)):
            map_name = map_names[i]
            map = maps[map_name]

            data[map_name] = map[idx]

        data_df = pd.DataFrame(data)
        data_df.to_csv(filename_out)


    # scale and transform to real crs coordinates
    def scale_to_real(self, shape):
        assert type(shape) == gpd.GeoDataFrame

        shape.geometry = shape.geometry.translate(self.x_min, self.y_min)
        shape.geometry = shape.geometry.scale(xfact=self.resolution, yfact=self.resolution, origin=(self.x_min, self.y_min))

        return shape


    ###########################################################
    # visualize
    ###########################################################

    # options:
    # - log_norm: whether rendering is displayed as log.
    #             useful for past patrol effort
    # - min_value and max_value: bounds on the colorbar scale
    # - plot_patrol_post: whether to display patrol posts in images
    def save_map(self, feature_map, feature_name, cmap='Greens', log_norm=False, min_value=None, max_value=None, plot_title=True, plot_patrol_post=True):
        # mask feature map
        feature_map = self.mask_to_grid(feature_map)

        if min_value is None:
            min_value = feature_map.min()
        if max_value is None:
            max_value = feature_map.max()

        fig, ax = plt.subplots()
        if log_norm:
            a = plt.imshow(np.flipud(feature_map), interpolation='none', cmap=cmap, extent=[0, self.gridmap.shape[1], 0, self.gridmap.shape[0]], vmin=min_value, vmax=max_value, norm=LogNorm())
        else:
            a = plt.imshow(np.flipud(feature_map), interpolation='none', cmap=cmap, extent=[0, self.gridmap.shape[1], 0, self.gridmap.shape[0]], vmin=min_value, vmax=max_value)

        plt.colorbar(a)

        # set plot title and labels
        if plot_title:
            plt.title(feature_name)
            #plt.xticks(np.arange(0,mx+1),[self.min_xval+resolution*i for i in range(mx+1)], rotation=60)
            plt.xlabel('x', fontsize=6)
            #plt.yticks(np.arange(0,my+1),[self.min_yval+resolution*i for i in range(my+1)])
            plt.ylabel('y', fontsize=6)

        # plot patrol post locations
        if plot_patrol_post and self.patrol_posts is not None:
            for index, row in self.patrol_posts.iterrows():
                sx = row['x']
                sy = row['y']
                plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color='aqua', markeredgewidth=1, markeredgecolor='blue')

        # set background color
        axes = plt.gca()
        axes.set_facecolor((0,0,0))

        plt.savefig(self.output_filepath + 'plot_{}.png'.format(feature_name))
        plt.close()


    # title - string
    # masked_map - masked np array of map to plot
    # shapefiles - dict of (string, GeoDataFrame) files
    # crs_out - string that specifies crs of the shapefiles
    def save_map_with_features(self, title, masked_map, shapefiles, crs_out, cmap='Reds', vmin=None, vmax=None, log_norm=False):
        map_grid = map_to_color_grid(masked_map)

        # prepare plot
        fig, ax = plt.subplots(figsize=(10,10), dpi=150)
        ax.set_facecolor((.9,.9,.9))    # gray background
        ax.set_aspect('equal')          # displays proportionally

        # hide tick labels
        ax.tick_params(labelbottom=False)
        ax.tick_params(labelleft=False)

        # make shapefiles directory
        if not os.path.exists(self.output_filepath + 'shapefiles/'):
            os.makedirs(self.output_filepath + 'shapefiles/')

        # create output shapefile and save
        map_grid.crs = crs_out  # {'init': crs_out}.
        map_grid = self.scale_to_real(map_grid)
        if log_norm:
            map_grid.plot(ax=ax, column='value', cmap=cmap, legend=True, vmin=vmin, vmax=vmax, norm=LogNorm())
        else:
            map_grid.plot(ax=ax, column='value', cmap=cmap, legend=True, vmin=vmin, vmax=vmax)
        map_grid.to_file('{}shapefiles/map_grid_{}.shp'.format(self.output_filepath, title))

        # plot shapefiles
        shapefiles['boundary'].plot(ax=ax, facecolor='none', edgecolor='black', linewidth=.5) # facecolor='#e4e8c6'
        if 'patrol_posts' in shapefiles:
            shapefiles['patrol_posts'].plot(marker='o', markersize=20, color='blue', ax=ax)
        if 'roads' in shapefiles:
            shapefiles['roads'].plot(ax=ax, facecolor='none', edgecolor='#68200c', linewidth=.5)
        if 'water' in shapefiles:
            shapefiles['water'].plot(ax=ax, facecolor='#40b4d1', edgecolor='black', linewidth=.5)
        if 'rivers' in shapefiles:
            shapefiles['rivers'].plot(ax=ax, facecolor='none', edgecolor='#40b4d1', linewidth=.5)
        if 'patrol_blocks' in shapefiles:
            shapefiles['patrol_blocks'].plot(ax=ax, facecolor='none', edgecolor='black', linewidth=.5)
        if 'core_zone' in shapefiles:
            shapefiles['core_zone'].plot(ax=ax, facecolor='none', edgecolor='green', linewidth=2)
        if 'buffer' in shapefiles:
            shapefiles['buffer'].plot(ax=ax, facecolor='none', edgecolor='#666666', linewidth=2)

        # save out plot
        plt.title('{}'.format(title))
        fig.savefig('{}map_{}.png'.format(self.output_filepath, title))

        plt.close()


    # NOTE: this .npy file must be saved from PatrolProblem.py
    # (or from this script)
    def get_riskmap_from_npy(self, npy_filename):
        riskmap = np.load(npy_filename)
        return riskmap


    # get list of np.arrays, where each is a map of predicted risk`
    # at a different threshold of patrol effort
    def get_maps_from_csv(self, maps_filename):
        num_extra_cols = 4

        map_data = pd.read_csv(maps_filename)
        map_data = self.shift_data_coords(map_data)

        num_maps = len(map_data.columns) - num_extra_cols
        maps = []
        for i in range(num_maps):
            print('  creating map: {}'.format(map_data.columns[num_extra_cols + i]))
            maps.append(np.zeros(self.gridmap.shape))

        for index, row in map_data.iterrows():
            for i in range(num_maps):
                maps[i][int(row['y'])][int(row['x'])] = row[[i+num_extra_cols]]

        for i in range(num_maps):
            maps[i] = self.mask_to_grid(maps[i])


        return map_data.columns[num_extra_cols:], maps


    def shift_data_coords(self, data):
        assert self.x_min is not None
        assert self.y_min is not None

        # compute point by scaling down by resolution
        data['x'] = (data['x'] - self.x_min) / self.resolution
        data['y'] = (data['y'] - self.y_min) / self.resolution

        # convert to int
        data['x'] = data['x'].astype(int)
        data['y'] = data['y'].astype(int)

        return data


    # create gridmap, which is a binary mask of cells within the boundary
    # 0 => point is not inside boundary
    # 1 => point is inside boundary
    def get_gridmap(self, static_features_filename):
        data = pd.read_csv(static_features_filename)

        # compute shifting for each row
        self.x_min = int(np.min(data['x']))
        self.y_min = int(np.min(data['y']))

        data = self.shift_data_coords(data)

        # set max values after scaling down by resolution
        scaled_x_max = int(np.max(data['x']))
        scaled_y_max = int(np.max(data['y']))

        # create gridmap
        gridmap = [[0 for x in range(scaled_x_max+1)] for y in range(scaled_y_max+1)]
        # gridmap = np.zeros((y_max+1, x_max+1))
        for index, row in data.iterrows():
            gridmap[int(row['y'])][int(row['x'])] = 1
        gridmap = np.ma.masked_where(gridmap == 1, gridmap)

        self.gridmap = gridmap

        return gridmap


    # read in and process all features from static features CSV
    def load_static_feature_maps(self, static_features_filename):
        print('load static features from {}...'.format(static_features_filename))

        data = pd.read_csv(static_features_filename)
        data = self.shift_data_coords(data)

        # create feature maps
        static_feature_names = list(data.columns[4:]) + ['Null']

        feature_maps = {}

        for static_feature_name in static_feature_names:
            print('  processing feature: {}'.format(static_feature_name))
            if static_feature_name == 'Null':
                feature_map = np.zeros(self.gridmap.shape)
                for index, row in data.iterrows():
                    feature_map[int(row['y'])][int(row['x'])] = 0

            else:
                feature_map = np.zeros(self.gridmap.shape)
                for index, row in data.iterrows():
                    feature_map[int(row['y'])][int(row['x'])] = row[static_feature_name]

            feature_maps[static_feature_name] = feature_map

        return feature_maps


    # get past patrol effort map
    def get_past_patrol(self, data_filename):
        data = pd.read_csv(data_filename)
        data = self.shift_data_coords(data)

        print('  processing past patrol effort from {} ...'.format(data_filename))

        # create map
        feature_map = np.ones(self.gridmap.shape) / float(10)
        for index, row in data.iterrows():
            feature_map[int(row['y'])][int(row['x'])] += row['current_patrol_effort']

        # mask
        feature_map = self.mask_to_grid(feature_map)

        return feature_map


    # get illegal activity map
    def get_illegal_activity(self, data_filename):
        data = pd.read_csv(data_filename)
        data = self.shift_data_coords(data)

        print('  processing illegal activity from {} ...'.format(data_filename))

        # create map
        feature_map = np.zeros(self.gridmap.shape)
        feature_df = pd.DataFrame(columns=['x', 'y', 'value'])

        for index, row in data.iterrows():
            if row['illegal_activity']:
                feature_map[int(row['y'])][int(row['x'])] += 1
        for x in range(feature_map.shape[0]):
            for y in range(feature_map.shape[1]):
                if feature_map[x][y]:
                    feature_df = feature_df.append(pd.DataFrame([[x, y, feature_map[x][y]]], columns=['x', 'y', 'value']))

        # mask
        feature_map = self.mask_to_grid(feature_map)

        return feature_map
