import os

import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile

from scipy import ndimage
from random import randint
from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt

from utility import *


###########################################################
# process risk map predictions
###########################################################

# average each cell over its adjacent neighbors
def convolve_map(map, width):
    assert type(width) is int

    # 5x5 Gaussian kernel
    kernel = np.asarray([1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1])
    kernel = kernel / 256
    kernel = kernel.reshape((5, 5))
    # kernel = np.ones((width, width)) / (width * width)
    #convolved = ndimage.convolve(map, kernel, mode='constant', cval=0.0)
    convolved = ndimage.convolve(map, kernel)       # default convolution: reflection

    return convolved

# sum a block of cells without normalizing
def convolve_map_sum(map, width):
    assert type(width) is int

    kernel = np.ones((width, width))
    convolved = ndimage.convolve(map, kernel)

    return convolved


# threshold is the bottom percentage we keep
def find_low_effort(gridmap, effort, threshold):
    effort = np.ma.masked_where(gridmap==0, effort)

    percentile = np.percentile(effort, threshold * 100)
    low_effort = np.ma.masked_where(effort > percentile, effort)

    return low_effort


# compute thresholds and divide riskmap into low, medium, and high-risk areas
def find_risk_classes(riskmap, percentile=None):
    # get rid of risk = 0
    riskmap = np.ma.masked_where(riskmap == 0, riskmap)

    valid_cells = riskmap[~riskmap.mask]

    # compute thresholds for different percentiles
    if percentile is None:
        # percentile = [0, 33.3, 33.3, 66.6, 66.6, 100]  # split into thirds
        percentile = [0, 20, 40, 60, 75, 100]  # greater differentiation

    # ensure percentile is a valid array
    assert len(percentile) == 6

    risk_percentile = np.percentile(valid_cells, percentile)
    print('  risk classes percentiles: ', risk_percentile)

    low_risk = np.ma.masked_where(riskmap < risk_percentile[0], riskmap)
    low_risk = np.ma.masked_where(low_risk > risk_percentile[1], low_risk)

    med_risk = np.ma.masked_where(riskmap < risk_percentile[2], riskmap)
    med_risk = np.ma.masked_where(med_risk > risk_percentile[3], med_risk)

    high_risk = np.ma.masked_where(riskmap < risk_percentile[4], riskmap)
    high_risk = np.ma.masked_where(high_risk > risk_percentile[5], high_risk)

    return low_risk, med_risk, high_risk


# create high variance and low variance maps
def find_variance(riskmap, variance_map, percentile=None):
    # mask variance map
    variance_map = np.ma.masked_where(riskmap.mask, variance_map)

    valid_cells = variance_map[~variance_map.mask]

    # compute thresholds for different percentiles
    if percentile is None:
        percentile = [0, 33.3, 33.3, 66.6, 66.6, 100]  # split into thirds
        # percentile = [0, 20, 40, 60, 75, 100]  # greater differentiation

    # ensure percentile is a valid array
    assert len(percentile) == 6

    var_percentile = np.percentile(variance_map, percentile)
    print('  variance classes percentiles: ', var_percentile)

    low_var = np.ma.masked_where(variance_map < var_percentile[0], variance_map)
    low_var = np.ma.masked_where(low_var > var_percentile[1], low_var)

    med_var = np.ma.masked_where(variance_map < var_percentile[2], variance_map)
    med_var = np.ma.masked_where(med_var > var_percentile[3], med_var)

    high_var = np.ma.masked_where(variance_map < var_percentile[4], variance_map)
    high_var = np.ma.masked_where(high_var > var_percentile[5], high_var)

    risk_low_var = np.ma.masked_where(low_var.mask, riskmap)
    risk_med_var = np.ma.masked_where(med_var.mask, riskmap)
    risk_high_var = np.ma.masked_where(high_var.mask, riskmap)

    return risk_low_var, risk_med_var, risk_high_var



# save blocks to shapefile
def save_map_shapefile(vis, map, name, crs_out, cmap='Reds', vmin=0., vmax=1.):
    map_grid = map_to_color_grid(map)

    # prepare plot
    fig, ax = plt.subplots(figsize=(10,10), dpi=150)
    # ax.set_facecolor((0,0,0))   # black background
    ax.set_aspect('equal')      # displays proportionally

    # create output shapefile and save
    map_grid.crs = crs_out  # {'init': crs_out}.
    map_grid = vis.scale_to_real(map_grid)
    map_grid.plot(ax=ax, column='value', legend=True, cmap=cmap, vmin=vmin, vmax=vmax)

    if not os.path.exists('{}/shapefiles/'.format(vis.output_filepath)):
        os.makedirs('{}/shapefiles/'.format(vis.output_filepath))

    map_grid.to_file('{}/shapefiles/{}_grid.shp'.format(vis.output_filepath, name))

    plt.title('Map - {}'.format(name))
    fig.savefig('{}/map_{}.png'.format(vis.output_filepath, name))
    plt.close()



# greedily select n blocks in riskmap using provided selection criteria function
# inputs:
# - riskmap    : masked riskmap of high_risk cells
# - n          : number of blocks to select
# - block_size : size of surrounding block to obscure
# - func_to_use: e.g. np.ma.argmin, np.ma.argmax
def select_risk_blocks(riskmap, n, block_size, func_to_use):
    if block_size % 2 != 1:
        raise Exception('block size {} invalid. must be an odd integer'.format(block_size))

    # create copy of riskmap to avoid making changes to original
    riskmap = riskmap.copy()

    # find the top n blocks without any overlap
    selected_blocks = []
    while len(selected_blocks) < n:
        # select the top/lowest/middle/whatever index according to func_to_use
        idx = func_to_use(riskmap)
        idx = np.unravel_index(idx, riskmap.shape)

        # # select a random index
        # valid_idx = np.where(~riskmap.mask)
        # if np.sum(valid_idx) == 0:
        #     raise Exception('number of blocks {} is too large. out of valid spaces'.format(n))
        # rand_pick = randint(0, valid_idx[0].shape[0]-1)
        # idx = (valid_idx[0][rand_pick], valid_idx[1][rand_pick])

        # center of future blocks must be at least two radiuses away
        invalid_radius = 2 * (block_size // 2)

        new_mask = np.zeros(riskmap.shape)
        for i in range(idx[0] - invalid_radius, idx[0] + invalid_radius + 1):
            for j in range(idx[1] - invalid_radius, idx[1] + invalid_radius + 1):
                # ensure we're within the image boundary
                if 0 <= i < riskmap.shape[0] and 0 <= j < riskmap.shape[1]:
                    new_mask[i, j] = 1

        np.ma.masked_where(new_mask, riskmap, copy=False)

        selected_blocks.append(idx)

    # convert list of blocks to tuple of ([first coords], [second coords])
    first_coord = [i for (i, j) in selected_blocks]
    second_coord = [j for (i, j) in selected_blocks]

    return [first_coord, second_coord]


# mask all the cells adjacent to the boundary
def mask_boundary_cells(gridmap, map, kernel_width):
    assert gridmap.shape == map.shape

    k = kernel_width // 2

    mask = np.zeros(gridmap.shape)
    # set walls to 1
    mask[:k, :] = 1                      # top
    mask[:, :k] = 1                      # left
    mask[gridmap.shape[0]-k:, :] = 1     # bottom
    mask[:, gridmap.shape[1]-k:] = 1     # right

    # set region around every masked cell to 1
    for i in range(1, gridmap.shape[0]):
        for j in range(1, gridmap.shape[1]):
            if gridmap[i,j] == 0:
                mask[i-(k):i+(k+1), j-(k):j+(k+1)] = 1

    # return a copy of the array, with additional mask
    return np.ma.masked_where(mask, map)


def make_field_test_barchart(high, med, low, filename):
    plt.figure(figsize=(4, 2), dpi=150)
    x_pos = np.arange(3)
    barlist = plt.bar(x_pos, [high, med, low])
    barlist[0].set_color('#ef0000')
    barlist[1].set_color('#ef8802')
    barlist[2].set_color('#f9e000')

    fontsize = 10
    plt.xlabel('Predicted poaching risk', fontsize=.8*fontsize)
    plt.xticks(x_pos, ('High', 'Medium', 'Low'), fontsize=fontsize)
    plt.ylabel('Poaching activity / km patrol', fontsize=.8*fontsize)
    plt.yticks(fontsize=fontsize)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# given a mask (numpy boolean array), expand each open cell (where mask is False)
# to the surrounding NxN region
def expand_block_mask(mask, block_size):
    expanded_mask = np.ones(mask.shape, dtype=bool)
    blocks = np.where(mask==False)
    radius = block_size // 2
    for k in range(blocks[0].shape[0]):
        idx = (blocks[0][k], blocks[1][k])
        # go through surrounding NxN region and set all values to False
        for i in range(idx[0] - radius, idx[0] + radius + 1):
            for j in range(idx[1] - radius, idx[1] + radius + 1):
                # ensure we're within the mask boundary
                if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
                    expanded_mask[i, j] = False

    return expanded_mask


class ProcessFieldTest:
    def __init__(self, vis, kernel_width, crs_out, ft_activity, ft_effort, historic_effort):
        self.vis = vis
        self.kernel_width = kernel_width
        self.crs_out = crs_out
        self.ft_activity = ft_activity
        self.ft_effort = ft_effort
        self.historic_effort = historic_effort

    # using actual patrol observations
    # inputs
    # - vis
    # - ft_activity, ft_effort: illegal activity and patrol effort during field tests
    # - riskmap : riskmap to use
    # - historic_effort : map of historic patrol effort
    # - historic_low_threshold : percentile of historic low patrol effort threshold.
    #                            only consider values below this percentile
    # - crs_out : for writing out shapefiles
    # - patrol_effort_threshold : percentile of patrol effort.
    #                             only consider values above this percentile
    def get_field_test_results(self, riskmap,
            historic_low_threshold=50, patrol_effort_threshold=10, num_blocks=50):
        gridmap = self.vis.gridmap
        vis = self.vis
        kernel_width = self.kernel_width
        crs_out = self.crs_out
        historic_effort = self.historic_effort

        ft_activity_orig = self.ft_activity
        ft_effort_orig = self.ft_effort

        # replace all observation values with 1 to make binary
        ft_activity_orig[ft_activity_orig.nonzero()] = 1

        # create 3x3km blocks: aggregate over effort and activity (can overlap unless that's too complicated...)
        #ft_activity_map = convolve_map_sum(ft_activity_orig, kernel_width)
        ft_activity_map = ft_activity_orig
        #ft_effort_map = convolve_map_sum(ft_effort_orig, kernel_width)
        ft_effort_map = ft_effort_orig


        # get max and min value for plots
        activity_max_val = np.max(ft_activity_map)
        activity_min_val = np.min(ft_activity_map)
        effort_max_val = np.max(ft_effort_map)
        effort_min_val = np.min(ft_effort_map)
        risk_max_val = np.max(riskmap)
        risk_min_val = np.min(riskmap)

        print('activity min {} max {}'.format(activity_min_val, activity_max_val))
        print('effort   min {} max {:.3f}'.format(effort_min_val, effort_max_val))
        print('risk     min {} max {:.3f}'.format(risk_min_val, risk_max_val))


        #######################################################
        # find valid riskmap areas
        #######################################################
        # find historically low patrol effort
        historic_low_effort = np.percentile(historic_effort[np.where(gridmap == 1)], historic_low_threshold)
        historic_effort_mask = np.logical_or(historic_effort > historic_low_effort, gridmap==0)

        # find blocks with 'sufficient patrol effort' (percentile threshold)
        sufficient_effort = np.percentile(ft_effort_map[np.where(gridmap == 1)], patrol_effort_threshold)
        print('  sufficient patrol effort threshold =', sufficient_effort)

        # mask away values below a patrol effort threshold or outside gridmap
        sufficient_effort_mask = np.logical_or(ft_effort_map < sufficient_effort, gridmap==0)

        # mask riskmap - must have historically low effort and sufficient current effort
        riskmap_valid = mask_boundary_cells(gridmap, riskmap, kernel_width)
        riskmap_valid = np.ma.masked_where(sufficient_effort_mask, riskmap_valid)
        riskmap_valid = np.ma.masked_where(historic_effort_mask, riskmap_valid)


        #######################################################
        # process riskmap into high-, medium-, and low- risk blocks
        #######################################################
        # visualize human activity and patrol effort maps
        vis.save_map(ft_activity_map, 'field_test-activity', cmap='Reds')
        vis.save_map(ft_effort_map, 'field_test-effort', cmap='Blues')

        # patrol effort and risk map, masked by sufficient patrol effort
        sufficient_effort_map = np.ma.masked_where(sufficient_effort_mask, ft_effort_map)
        vis.save_map(sufficient_effort_map, 'field_test-effort_sufficient', cmap='Blues', log_norm=False, min_value=None, max_value=None, plot_patrol_post=False)
        vis.save_map(riskmap_valid, 'field_test-riskmap_valid', cmap='Reds', log_norm=False, min_value=None, max_value=None, plot_patrol_post=False)

        # current human activity and effort, valid regions
        ft_activity_map_valid = np.ma.masked_where(sufficient_effort_mask, ft_activity_map)
        ft_activity_map_valid = np.ma.masked_where(historic_effort_mask, ft_activity_map_valid)
        ft_effort_map_valid = np.ma.masked_where(sufficient_effort_mask, ft_effort_map)
        ft_effort_map_valid = np.ma.masked_where(historic_effort_mask, ft_effort_map_valid)

        vis.save_map(ft_activity_map_valid, 'field_test-activity_valid', cmap='Reds')
        vis.save_map(ft_effort_map_valid, 'field_test-effort_valid', cmap='Blues')

        # compute percentiles of risk predictions
        risk_percentile = np.arange(0, 101, 25)  # 25 values between 0 and 100
        riskmap_percentiles = np.percentile(riskmap_valid.compressed(), risk_percentile)
        print('  riskmap_percentiles:', np.around(riskmap_percentiles, 3))

        # find high-, medium-, and low-risk blocks
        high_risk = np.ma.masked_less(riskmap_valid, riskmap_percentiles[-2])

        med_risk = np.ma.masked_outside(riskmap_valid,
            riskmap_percentiles[len(riskmap_percentiles) // 2 - 1],
            riskmap_percentiles[len(riskmap_percentiles) // 2])

        low_risk = np.ma.masked_greater(riskmap_valid, riskmap_percentiles[1])

        print('  # cells: high risk {}, med risk {}, low risk {}'.format(
            np.where(~high_risk.mask)[0].shape[0],
            np.where(~med_risk.mask)[0].shape[0],
            np.where(~low_risk.mask)[0].shape[0]))

        vis.save_map(high_risk, 'field_test-riskmap_high', cmap='Reds', log_norm=False, min_value=0, max_value=risk_max_val, plot_patrol_post=False)
        vis.save_map(med_risk, 'field_test-riskmap_med', cmap='Reds', log_norm=False, min_value=0, max_value=risk_max_val, plot_patrol_post=False)
        vis.save_map(low_risk, 'field_test-riskmap_low', cmap='Reds', log_norm=False, min_value=0, max_value=risk_max_val, plot_patrol_post=False)


        #######################################################
        # select risk blocks to avoid overlap
        #######################################################
        # def arg_medium_val(masked_array):
        #     sorted_indices = np.ma.argsort(masked_array, axis=None)
        #     num_masked_entries = np.sum(masked_array.mask)     # True evaluates as 1
        #     center_index = sorted_indices[(sorted_indices.shape[0] - num_masked_entries) // 2]
        #     return center_index
        #
        # print('  with {} blocks:'.format(num_blocks))
        # blocks_high_risk = select_risk_blocks(high_risk, num_blocks, kernel_width, np.ma.argmax)
        # blocks_med_risk  = select_risk_blocks(med_risk, num_blocks, kernel_width, arg_medium_val)
        # blocks_low_risk  = select_risk_blocks(low_risk, num_blocks, kernel_width, np.ma.argmin)
        #
        # # create new masks to display only the selected blocks
        # blocks_high_risk_mask = np.ones(gridmap.shape)
        # blocks_med_risk_mask  = np.ones(gridmap.shape)
        # blocks_low_risk_mask  = np.ones(gridmap.shape)
        #
        # blocks_high_risk_mask[tuple(blocks_high_risk)] = 0
        # blocks_med_risk_mask[tuple(blocks_med_risk)]   = 0
        # blocks_low_risk_mask[tuple(blocks_low_risk)]   = 0
        #
        # # create masked riskmaps with only selected blocks
        # blocks_high_riskmap = np.ma.masked_where(blocks_high_risk_mask, riskmap)
        # blocks_med_riskmap  = np.ma.masked_where(blocks_med_risk_mask, riskmap)
        # blocks_low_riskmap  = np.ma.masked_where(blocks_low_risk_mask, riskmap)
        #
        # vis.save_map(blocks_high_riskmap, 'field_test-riskmap_blocks_high', cmap='Reds', log_norm=False, min_value=0, max_value=risk_max_val, plot_patrol_post=False)
        # vis.save_map(blocks_med_riskmap, 'field_test-riskmap_blocks_med', cmap='Reds', log_norm=False, min_value=0, max_value=risk_max_val, plot_patrol_post=False)
        # vis.save_map(blocks_low_riskmap, 'field_test-riskmap_blocks_low', cmap='Reds', log_norm=False, min_value=0, max_value=risk_max_val, plot_patrol_post=False)
        blocks_high_riskmap = high_risk
        blocks_med_riskmap = med_risk
        blocks_low_riskmap = low_risk


        #######################################################
        # aggregate patrol effort and activity over the blocks
        #######################################################
        activity_high_risk = np.ma.masked_where(blocks_high_riskmap.mask, ft_activity_map)
        activity_med_risk  = np.ma.masked_where(blocks_med_riskmap.mask, ft_activity_map)
        activity_low_risk  = np.ma.masked_where(blocks_low_riskmap.mask, ft_activity_map)

        effort_high_risk = np.ma.masked_where(blocks_high_riskmap.mask, ft_effort_map)
        effort_med_risk  = np.ma.masked_where(blocks_med_riskmap.mask, ft_effort_map)
        effort_low_risk  = np.ma.masked_where(blocks_low_riskmap.mask, ft_effort_map)

        high_avg = activity_high_risk.sum() / effort_high_risk.sum()
        med_avg  = activity_med_risk.sum() / effort_med_risk.sum()
        low_avg  = activity_low_risk.sum() / effort_low_risk.sum()

        # count number of blocks with nonzero patrol effort
        num_high_nonzero = effort_high_risk.nonzero()[0].shape[0]
        num_med_nonzero  = effort_med_risk.nonzero()[0].shape[0]
        num_low_nonzero  = effort_low_risk.nonzero()[0].shape[0]

        # report activity / patrol effort in each risk category (H, M, L)
        print('    high risk: #o {}, effort {:.2f}, #o / effort {:.3f}    #c {}, #o / #c {:.3f}'.format(
            activity_high_risk.sum(), effort_high_risk.sum(), high_avg, num_high_nonzero, activity_high_risk.sum() / num_high_nonzero))
        print('    med risk:  #o {}, effort {:.2f}, #o / effort {:.3f}    #c {}, #o / #c {:.3f}'.format(
            activity_med_risk.sum(), effort_med_risk.sum(), med_avg, num_med_nonzero, activity_med_risk.sum() / num_med_nonzero))
        print('    low risk:  #o {}, effort {:.2f}, #o / effort {:.3f}    #c {}, #o / #c {:.3f}'.format(
            activity_low_risk.sum(), effort_low_risk.sum(), low_avg, num_low_nonzero, activity_low_risk.sum() / num_low_nonzero))

        # display final result as a barchart
        make_field_test_barchart(high_avg, med_avg, low_avg, filename=vis.output_filepath + 'SWS_field_test_barchart.png')

        # # compute number of cells patrolled among the risk blocks
        # expand_high_risk = expand_block_mask(blocks_high_riskmap.mask, kernel_width)
        # expand_med_risk = expand_block_mask(blocks_med_riskmap.mask, kernel_width)
        # expand_low_risk = expand_block_mask(blocks_low_riskmap.mask, kernel_width)
        #
        # # mask patrol effort with expanded risk blocks
        # orig_effort_high = np.ma.masked_where(expand_high_risk, ft_effort_orig)
        # orig_effort_med = np.ma.masked_where(expand_med_risk, ft_effort_orig)
        # orig_effort_low = np.ma.masked_where(expand_low_risk, ft_effort_orig)

        # # count number of cells within the high-, low-, and medium- risk block that have been patrolled
        # print('cells within blocks with nonzero patrol')
        # min_patrol = 0
        # num_high_nonzero_patrol = np.where(orig_effort_high.compressed() > min_patrol)[0].shape[0]
        # num_med_nonzero_patrol = np.where(orig_effort_med.compressed() > min_patrol)[0].shape[0]
        # num_low_nonzero_patrol = np.where(orig_effort_low.compressed() > min_patrol)[0].shape[0]
        #
        # kernel_sq = kernel_width * kernel_width
        #
        # print('  num high: {}, # obs / # cells {:.3f} | normalized: {:.1f}, normalized #o/#c {:.3f}'.format(
        #     num_high_nonzero_patrol, activity_high_risk.sum() / num_high_nonzero_patrol,
        #     num_high_nonzero_patrol / kernel_sq, activity_high_risk.sum() / (num_high_nonzero_patrol / kernel_sq)))
        # print('  num med:  {}, # obs / # cells {:.3f} | normalized: {:.1f}, normalized #o/#c {:.3f}'.format(
        #     num_med_nonzero_patrol, activity_med_risk.sum() / num_med_nonzero_patrol,
        #     num_med_nonzero_patrol / kernel_sq, activity_med_risk.sum() / (num_med_nonzero_patrol / kernel_sq)))
        # print('  num low:  {}, # obs / # cells {:.3f} | normalized: {:.1f}, normalized #o/#c {:.3f}'.format(
        #     num_low_nonzero_patrol, activity_low_risk.sum() / num_low_nonzero_patrol,
        #     num_low_nonzero_patrol / kernel_sq, activity_low_risk.sum() / (num_low_nonzero_patrol / kernel_sq)))

        # if SAVE:
        #     # save activity maps, masked to high/med/low risk block
        #     vis.save_map(activity_high_risk, 'field_test-activity_high_risk', cmap='Reds', log_norm=False, min_value=activity_min_val, max_value=activity_max_val, plot_patrol_post=False)
        #     vis.save_map(activity_med_risk, 'field_test-activity_med_risk', cmap='Reds', log_norm=False, min_value=activity_min_val, max_value=activity_max_val, plot_patrol_post=False)
        #     vis.save_map(activity_low_risk, 'field_test-activity_low_risk', cmap='Reds', log_norm=False, min_value=activity_min_val, max_value=activity_max_val, plot_patrol_post=False)
        #
        #     # save patrol effort maps, masked to high/med/low risk blocks
        #     vis.save_map(effort_high_risk, 'field_test-effort_high_risk', cmap='Blues', log_norm=False, min_value=effort_min_val, max_value=effort_max_val, plot_patrol_post=False)
        #     vis.save_map(effort_med_risk, 'field_test-effort_med_risk', cmap='Blues', log_norm=False, min_value=effort_min_val, max_value=effort_max_val, plot_patrol_post=False)
        #     vis.save_map(effort_low_risk, 'field_test-effort_low_risk', cmap='Blues', log_norm=False, min_value=effort_min_val, max_value=effort_max_val, plot_patrol_post=False)

        # if SAVE:
            # vis.save_map(masked_riskmap, 'field_test-sufficientPE', cmap='Reds', plot_patrol_post=True)
        #save_riskmap_as_shapefile(extents, crs_out, masked_riskmap, 'field_test-sufficientPE')
