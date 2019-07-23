import sys
import os

import numpy as np
from scipy import ndimage
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import shapefile

from utility import *
from visualize import *
from process_risk_maps import *


PARK = 'SWS_May2019'
RESOLUTION = 1000

KERNEL_WIDTH = 5
PATROL_EFFORT_THRESHOLD = 1    # low-effort areas must be below this fraction of squares

SAVE = True # whether or not to save output images and files


input_filepath = '../preprocess_consolidate/{}/{}/output/'.format(PARK, RESOLUTION)

# output_filepath = './{}_DecJan/output/'.format(PARK)
# ft_input_filepath = '../../cambodia/field tests/2019 05 analysis/DecJan/'
# risk_filename     = ft_input_filepath + 'predictions_SWS_2018_method_gp_5.csv'
# variance_filename = ft_input_filepath + 'variances_SWS_2018_method_gp_5.csv'

output_filepath = './{}_FebMar/output/'.format(PARK)
ft_input_filepath = '../../cambodia/field tests/2019 05 analysis/FebMar/'
risk_filename     = ft_input_filepath + 'predictions_SWS_2019_method_gp_5.csv'
variance_filename = ft_input_filepath + 'variances_SWS_2019_method_gp_5.csv'

# output_filepath = './{}_June/output/'.format(PARK)
# ft_input_filepath = '../../cambodia/field tests/2019 06/'
# risk_filename     = ft_input_filepath + 'predictions_SWS_2019_method_gp_10.csv'
# variance_filename = ft_input_filepath + 'variances_SWS_2019_method_gp_10.csv'

if SAVE:
    import matplotlib

if not os.path.exists(output_filepath):
    os.makedirs(output_filepath)



if __name__ == "__main__":
    #######################################################
    # set up input files
    #######################################################
    static_features_filename = input_filepath + 'static_features.csv'
    patrol_post_filename = input_filepath + 'patrol_posts.csv'
    x_filename = input_filepath + 'rainy_2month/rainy_x.csv'
    y_filename = input_filepath + 'rainy_2month/rainy_y.csv'

    # crs_in: "+proj=longlat +datum=WGS84"
    crs_out = '+proj=utm +zone=48 +north +datum=WGS84 +units=m +no_defs'

    vis = Visualize(RESOLUTION, input_filepath, output_filepath)


    #######################################################
    # load data
    #######################################################

    # create gridmap, which maps
    print('get gridmap...')
    gridmap = vis.get_gridmap(static_features_filename)
    vis.set_patrol_posts(patrol_post_filename)

    print('processing features...')
    historical_effort_map = vis.get_past_patrol(x_filename)
    historical_activity_map = vis.get_illegal_activity(y_filename)

    # load features from static features file
    static_features = vis.load_static_feature_maps(static_features_filename)

    if SAVE:
        vis.save_map(historical_effort_map, 'patrol_effort')
        vis.save_map(historical_activity_map, 'illegal_activity', cmap='Reds')

        for feature_name in static_features:
            vis.save_map(static_features[feature_name], feature_name)


    #######################################################
    # read in riskmaps
    #######################################################

    # open riskmap and save images
    print('reading in risk predictions from {}...'.format(risk_filename))
    riskmap_thresholds, riskmaps = vis.get_maps_from_csv(risk_filename)

    # open variance maps
    print('reading in variance predictions from {}...'.format(variance_filename))
    _, variance_maps = vis.get_maps_from_csv(variance_filename)

    # combine all riskmaps together
    combined_riskmap = np.copy(riskmaps[0])
    for i in range(1, len(riskmaps)):
        combined_riskmap += riskmaps[i]
    combined_riskmap = combined_riskmap / len(riskmaps)

    if SAVE:
        # risk_min_val = 0
        risk_min_val = min([np.min(riskmap) for riskmap in riskmaps])
        risk_max_val = max([np.max(riskmap) for riskmap in riskmaps])
        print('  riskmap min value: {:.5f}, max value {:.5f}'.format(risk_min_val, risk_max_val))

        for i in range(len(riskmaps)):
            vis.save_map(riskmaps[i], 'risk_map_{}_{}'.format(i, riskmap_thresholds[i]), cmap='Reds', plot_patrol_post=False)
            #vis.save_map(riskmaps[i], 'risk_map_{}_{}'.format(i, riskmap_thresholds[i]), cmap='Reds', min_value=risk_min_val, max_value=risk_max_val, plot_patrol_post=False)
        vis.save_map(combined_riskmap, 'risk_map_COMBINED', cmap='Reds', min_value=risk_min_val, max_value=risk_max_val, plot_patrol_post=False)

        for i in range(len(riskmaps)):
            vis.save_map(variance_maps[i], 'variance_map_{}_{}'.format(i, riskmap_thresholds[i]), cmap='Greens', plot_patrol_post=False)


    #######################################################
    # process low patrol effort
    #######################################################

    # # set zero patrol effort to no value
    # historical_effort_map = np.ma.masked_where(historical_effort_map==0, historical_effort_map)

    print('  convolving patrol effort then finding low patrol effort...')
    convolved_patrol = convolve_map(historical_effort_map, KERNEL_WIDTH)
    low_effort_map = find_low_effort(vis.gridmap, convolved_patrol, PATROL_EFFORT_THRESHOLD)
    if SAVE:
        vis.save_map(convolved_patrol, 'patrol_effort_convolved', log_norm=True, plot_patrol_post=True)
        vis.save_map(low_effort_map, 'patrol_effort_convolved_low', log_norm=True, plot_patrol_post=True)


    #######################################################
    # analyze field test results
    #######################################################

    print('#######################################')
    print('analyzing field test results')
    print('#######################################')

    ft_filename = ft_input_filepath + 'combined.csv'
    print('load field test data from file {}...'.format(ft_filename))
    ft_data = vis.load_static_feature_maps(ft_filename)
    ft_effort = ft_data['patrol effort']
    ft_activity = ft_data['illegal activity']
    if SAVE:
        vis.save_map(ft_effort, 'ft_effort', cmap='Greens', plot_patrol_post=False)
        vis.save_map(ft_activity, 'ft_activity', cmap='Reds', plot_patrol_post=False)

    historical_effort_map = convolve_map(historical_effort_map, KERNEL_WIDTH)

    process_ft = ProcessFieldTest(vis, KERNEL_WIDTH, crs_out, ft_activity, ft_effort, historical_effort_map)

    SELECTED_MAP = 1       # selected riskmap
    print('-------------------------')
    print('riskmap {}, {}'.format(SELECTED_MAP, riskmap_thresholds[SELECTED_MAP]))
    riskmap_convolved = convolve_map(riskmaps[SELECTED_MAP], KERNEL_WIDTH)
    process_ft.get_field_test_results(riskmap_convolved, historic_low_threshold=80, patrol_effort_threshold=1)

    if SAVE:
        vis.save_map(riskmap_convolved, 'riskmap_convolved_{}'.format(KERNEL_WIDTH), cmap='Reds', min_value=risk_min_val, max_value=risk_max_val)


    for i in range(len(riskmaps)):
        print('-------------------------')
        print('riskmap {}, {}'.format(i, riskmap_thresholds[i]))
        riskmap_convolved = convolve_map(riskmaps[i], KERNEL_WIDTH)
        process_ft.get_field_test_results(riskmap_convolved, historic_low_threshold=80, patrol_effort_threshold=1)



    print('-------------------------')
    print('using combined riskmap')
    riskmap_convolved = convolve_map(combined_riskmap, KERNEL_WIDTH)
    process_ft.get_field_test_results(riskmap_convolved, historic_low_threshold=80, patrol_effort_threshold=1)

    sys.exit(0)



    #######################################################
    # select low-, medium-, and high-risk areas
    #######################################################

    print('#######################################')
    print('selecting field test areas')
    print('#######################################')

    # select which riskmap to use
    SELECTED_MAP = 7
    print('using riskmap {} with threshold {}'.format(SELECTED_MAP, riskmap_thresholds[SELECTED_MAP]))

    riskmap_selected = riskmaps[SELECTED_MAP]

    # normalize map
    print('  normalizing map')
    riskmap_selected -= riskmap_selected.min()
    riskmap_selected /= riskmap_selected.max()

    riskmap_convolved = convolve_map(riskmap_selected, KERNEL_WIDTH)

    if SAVE:
        vis.save_map(riskmap_selected, 'riskmap_selected', cmap='Reds')#, min_value=risk_min_val, max_value=risk_max_val)
        vis.save_map(riskmap_convolved, 'riskmap_convolved', cmap='Reds')#, min_value=risk_min_val, max_value=risk_max_val)

    # avoid picking cells along boundary
    riskmap_convolved = mask_boundary_cells(gridmap, riskmap_convolved, KERNEL_WIDTH)

    # mask riskmap to only show cells with low patrol effort
    print('masking riskmap to show only cells with low patrol effort...')
    masked_riskmap = np.ma.masked_where(low_effort_map.mask, riskmap_convolved)
    if SAVE:
        vis.save_map(masked_riskmap, 'riskmap_low_effort', cmap='Reds', min_value=0., max_value=1.)


    # produce maps visualizing low, medium, and high risk areas
    print('finding risk classes: high, medium, low...')
    risk_percentile = [0, 25, 25, 50, 50, 100]
    low_risk, med_risk, high_risk = find_risk_classes(masked_riskmap, percentile=risk_percentile)
    if SAVE:
        vis.save_map(low_risk, 'riskmap_low_risk_{}-{}'.format(risk_percentile[0], risk_percentile[1]),
                        cmap='Reds', min_value=0., max_value=1.)
        vis.save_map(med_risk, 'riskmap_med_risk_{}-{}'.format(risk_percentile[2], risk_percentile[3]),
                        cmap='Reds', min_value=0., max_value=1.)
        vis.save_map(high_risk, 'riskmap_high_risk_{}-{}'.format(risk_percentile[4], risk_percentile[5]),
                        cmap='Reds', min_value=0., max_value=1.)


    # produce maps visualizing high risk areas with high and low variance
    print('find high risk, low/med/high variance...')
    variance_map = variance_maps[SELECTED_MAP]
    variance_percentile = [0, 50, 50, 90, 90, 100]

    risk_low_var, risk_med_var, risk_high_var = find_variance(high_risk, variance_map, percentile=variance_percentile)
    if SAVE:
        vis.save_map(risk_low_var, 'riskmap_high_risk_low_var', cmap='Reds', min_value=0., max_value=1.)
        vis.save_map(risk_med_var, 'riskmap_high_risk_med_var', cmap='Reds', min_value=0., max_value=1.)
        vis.save_map(risk_high_var, 'riskmap_high_risk_high_var', cmap='Reds', min_value=0., max_value=1.)

    # save out map and CSV
    csv_out = output_filepath + 'risk_areas.csv'
    maps_out = {'selected riskmap': riskmap_selected, 'variance map': variance_map,
                'convolved riskmap': riskmap_convolved,
                'low risk': low_risk, 'medium risk': med_risk, 'high risk': high_risk,
                'high risk low var': risk_low_var, 'high risk med var': risk_med_var, 'high risk high var': risk_high_var}
    vis.save_maps_to_csv(csv_out, maps_out)


    if SAVE:
        # visualize risk maps
        print('visualizing high risk maps with low/med/high variance...')
        save_riskmap_as_shapefile(vis, KERNEL_WIDTH, crs_out, risk_low_var, 'high risk low var', min_value=0., max_value=1.)
        save_riskmap_as_shapefile(vis, KERNEL_WIDTH, crs_out, risk_med_var, 'high risk med var', min_value=0., max_value=1.)
        save_riskmap_as_shapefile(vis, KERNEL_WIDTH, crs_out, risk_high_var, 'high risk high var', min_value=0., max_value=1.)
