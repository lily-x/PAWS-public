import sys
import os
sys.path.append('../field_tests')

from visualize import *
from utility import *


# normalize - normalize all maps to scale between 0 and 1
def read_map_thresholds(vis, shapes, filename, name, color, crs_out, normalize=True):
    # open riskmap and save image
    print('reading in risk predictions from file {}...'.format(filename))
    map_thresholds, maps = vis.get_maps_from_csv(filename)

    #map_min_value = 0
    map_min_value = min([map.min() for map in maps])
    map_max_value = max([map.max() for map in maps])

    if normalize:
        vmin, vmax = 0., 1.
    else:
        vmin, vmax = None, None

    # save pretty plot with features
    for i in range(len(maps)):
        if normalize:
            print('  normalizing. old min is {:.5f} and max {:.5f}'.format(maps[i].min(), maps[i].max()))
            maps[i] = (maps[i] - map_min_value) / (map_max_value - map_min_value)

        vis.save_map_with_features('{}_{}_{}'.format(name, i, map_thresholds[i]),
                                    maps[i], shapes, crs_out, cmap=color, vmin=vmin, vmax=vmax)

    # save plain plot with no features
    for i in range(len(maps)):
        vis.save_map(maps[i], '{}_{}_{}'.format(name, i, map_thresholds[i]), cmap=color, log_norm=False, min_value=vmin, max_value=vmax, plot_patrol_post=False)

    # combine all riskmaps together
    combined_map = maps[0]
    for i in range(1, len(maps)):
        combined_map += maps[i]
    combined_map = combined_map / len(maps)

    vis.save_map_with_features('{}_combined'.format(name),
                                combined_map, shapes, crs_out, cmap=color, vmin=vmin, vmax=vmax)

    vis.save_map(combined_map, '{}_combined'.format(name), cmap=color, log_norm=False, min_value=vmin, max_value=vmax, plot_patrol_post=False)
