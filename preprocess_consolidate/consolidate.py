# PAWS preprocessing code
# Lily Xu
# April 2019

import os
import numpy as np
import pandas as pd


def consolidate_pipeline(filepath, static_feature_names, start_year, end_year, num_months, out_prefix, use_months=None):
    if use_months is None:
        sections_per_year = 12 // num_months
    else:
        sections_per_year = len(use_months) // num_months
    static_features = combine_static_features(filepath, static_feature_names)
    activity = process_human_activity(filepath, start_year, end_year, num_months, out_prefix, use_months)
    effort   = process_patrol_effort(filepath, start_year, end_year, num_months, out_prefix, use_months)
    climate  = process_climate(filepath, start_year, end_year, num_months, out_prefix, use_months)
    x, y = combine_data(static_features, activity, effort, climate, filepath, num_months, sections_per_year, start_year, end_year, out_prefix)

    return x, y


###########################################################
# combine static features
###########################################################

# combine all static features into one table
def combine_static_features(filepath, static_feature_names):
    # make 'output' folder if it doesn't already exist
    if not os.path.exists('{}/output/'.format(filepath)):
        os.makedirs('{}/output/'.format(filepath))

    print('combining {} static features...'.format(len(static_feature_names)))

    features = pd.DataFrame()

    for feat in static_feature_names:
        print('  reading feature {}'.format(feat))
        data = pd.read_csv('{}/input/{}.csv'.format(filepath, feat))

        # add additional info in the first iteration
        if feat == static_feature_names[0]:
            features['spatial_id'] = data.iloc[:, 0]
            features['x'] = data.iloc[:, 1]
            features['y'] = data.iloc[:, 2]

        # add feature data
        features[feat] = data.iloc[:, -1]  # last column

    # write out CSV
    file_out = '{}/output/static_features.csv'.format(filepath)
    print('  writing out file: {}'.format(file_out))
    features.to_csv(file_out)

    return features



###########################################################
# helper functions
# for processing patrol effort and human activity
###########################################################

# assertions
def validate_months(data, num_months, months_use):
    assert len(data.columns) % 12 == 0    # ensure we have 12 months for each year
    assert num_months <= 12               # no more than one year
    assert 12 % num_months == 0           # cleanly divides 12

    # set months_use to all 12 months if not set
    if months_use is None:
        months_use = list(range(1, 13))

    assert len(months_use) <= 12
    assert num_months <= len(months_use)
    assert len(months_use) % num_months == 0

    return months_use


# update column headers and write CSV file
def save_combined_months(data_sum, start_year, end_year, num_months, months_use, filepath, type_name, out_prefix):
    # make 'output' folder if it doesn't already exist
    if not os.path.exists('{}/output/{}_{}month'.format(filepath, out_prefix, num_months)):
        os.makedirs('{}/output/{}_{}month'.format(filepath, out_prefix, num_months))

    # change column headers to be readable
    years = list(range(start_year, end_year + 1))
    if num_months == 12:
        data_sum.columns = years
    else:
        sections = list(range(0, len(months_use) // num_months))
        data_sum.columns = ['{}-{}'.format(y, s) for y in years for s in sections]

    # write out to CSV
    out_filename = '{}/output/{}_{}month/{}.csv'.format(filepath, out_prefix, num_months, type_name)
    print('  writing out file: {}'.format(out_filename))
    data_sum.to_csv(out_filename)

    return data_sum


# combine columns of data frame according to num_months and months_use
def sum_selected_months(data, num_years, num_months, months_use):
    # get selected months
    use_idx  = [m + 12*y - 1 for y in range(num_years) for m in months_use]
    data_use = data[data.columns[use_idx]]

    # sum across every num_months columns
    data_use.columns = list(range(data_use.shape[1]))
    data_sum = data_use.groupby(data_use.columns // num_months, axis=1).sum()

    return data_sum


###########################################################
# process human activity
###########################################################

# months_use - list of which months to process. allows for seasons.
#              optional parameter. if None, default to [1, ..., 12]
# binary - optional parameter that, if true, restricts illegal activity to 0 or 1
def process_human_activity(filepath, start_year, end_year, num_months, out_prefix='all', months_use=None, binary=True):
    print('processing human activity...')

    data = pd.read_csv('{}/input/human_activity_month.csv'.format(filepath))

    data.drop(columns=data.columns[0], inplace=True)  # first column is only IDs

    months_use = validate_months(data, num_months, months_use)

    num_years = end_year - start_year + 1
    num_activities = len(data.columns) // num_years // 12

    print('  {} years, {} activities'.format(num_years, num_activities))

    # combine illegal activities
    data.columns = list(range(data.shape[1]))
    data_combine = data.groupby(data.columns % (num_years * 12), axis=1).sum()

    # sum data by selected months
    data_sum = sum_selected_months(data_combine, num_years, num_months, months_use)

    # restrict labels to a binary 0 or 1
    data_sum[data_sum > 0] = 1

    return save_combined_months(data_sum, start_year, end_year, num_months, months_use, filepath, 'human_activity', out_prefix)


###########################################################
# process patrol effort
###########################################################

# months_use - list of which months to process. allows for seasons.
#              optional parameter. if None, default to [1, ..., 12]
def process_patrol_effort(filepath, start_year, end_year, num_months, out_prefix='all', months_use=None):
    print('processing patrol effort...')

    data = pd.read_csv('{}/input/patrol_month.csv'.format(filepath))
    data.drop(columns=data.columns[0], inplace=True)  # first column is only IDs

    months_use = validate_months(data, num_months, months_use)

    # sum data by selected months
    num_years = end_year - start_year + 1

    assert data.shape[1] / 12 == num_years  # ensure input start/end year match data

    data_sum = sum_selected_months(data, num_years, num_months, months_use)

    return save_combined_months(data_sum, start_year, end_year, num_months, months_use, filepath, 'patrol_effort', out_prefix)


###########################################################
# process GPP
###########################################################
def process_gpp(filepath, start_year, end_year, num_months, out_prefix='all', months_use=None):
    print('processing GPP...')

    data = pd.read_csv('{}/input/GPP.csv'.format(filepath))
    data.drop(columns=data.columns[0], inplace=True)  # first column is only IDs

    months_use = validate_months(data, num_months, months_use)

    # sum data by selected months
    num_years = end_year - start_year + 1
    data_sum = sum_selected_months(data, num_years, num_months, months_use)

    # average
    data_sum /= num_months

    return save_combined_months(data_sum, start_year, end_year, num_months, months_use, filepath, 'GPP', out_prefix)


###########################################################
# process climate
###########################################################

# use the month before
def process_climate(filepath, start_year, end_year, num_months, out_prefix='all', months_use=None):
    print('processing climate...')

    data = pd.read_csv('{}/input/climate.csv'.format(filepath))
    data = data.transpose()

    # keep only relevant rows and columns
    years    = data.loc['year'].values
    idx_keep = np.where(np.logical_and(start_year <= years, years <= end_year))[0]
    print('idx_keep before', idx_keep)
    print('idx_keep after', idx_keep[0] - num_months, idx_keep[-1] - num_months + 1)

    valid_data = data.iloc[:, (idx_keep[0] - num_months) : (idx_keep[-1] - num_months + 1)]
    valid_data = valid_data.loc[['temp', 'precip']]

    months_use = validate_months(valid_data, num_months, months_use)

    # sum data by selected months
    num_years = end_year - start_year + 1
    data_sum = sum_selected_months(valid_data, num_years, num_months, months_use)

    # get average
    data_avg = data_sum / num_months

    return save_combined_months(data_avg, start_year, end_year, num_months, months_use, filepath, 'climate', out_prefix)



###########################################################
# combine data
###########################################################

#def combine_data(static_features, illegal_activity, patrol_effort, climate, gpp, filepath, num_months, sections_per_year, start_year, end_year, out_prefix='all'):
#def combine_data(static_features, illegal_activity, patrol_effort, climate, filepath, num_months, sections_per_year, start_year, end_year, out_prefix='all'):
def combine_data(static_features, illegal_activity, patrol_effort, filepath, num_months, sections_per_year, start_year, end_year, out_prefix='all'):
    print('combining data...')

    num_cells = patrol_effort.shape[0]  # number of spatial points (grid cells)
    num_time  = patrol_effort.shape[1]  # number of temporal points

    num_years = num_time // sections_per_year

    print('  start year {}, end year {}'.format(start_year, end_year))

    assert static_features.shape[0] == illegal_activity.shape[0] == patrol_effort.shape[0]
    assert end_year - start_year == num_years - 1

    print('  {} time steps, {} cells, {} years ({} - {}), {} static features'.format(num_time, num_cells, num_years, start_year, end_year, static_features.shape[1] - 3))

    # generate table columns
    global_id = np.arange(num_cells)
    global_id = np.tile(global_id, num_time)

    year      = np.arange(start_year, end_year + 1)
    year      = np.repeat(year, num_cells * sections_per_year)

    section   = np.arange(sections_per_year)
    section   = np.tile(section, num_years)
    section   = np.repeat(section, num_cells)

    # or use .iloc[:, -1] to get column #
    spatial_id = static_features['spatial_id']
    spatial_id = spatial_id.squeeze()
    spatial_id = np.tile(spatial_id, num_time)

    x_coord    = static_features['x'].squeeze()
    x_coord    = np.tile(x_coord, num_time)

    y_coord    = static_features['y'].squeeze()
    y_coord    = np.tile(y_coord, num_time)

    # temp = np.repeat(climate.loc['temp'].values, num_cells)
    # precip = np.repeat(climate.loc['precip'].values, num_cells)

    # gpp_flat = gpp.values.T.flatten()


    # combine all X features (static features, patrol effort, climate)
    current_effort = patrol_effort.values.T.flatten()

    # past patrol effort: push everything back a year and set first year to 0
    past_effort = np.roll(current_effort, num_cells)
    past_effort[:num_cells] = 0

    rep_static_features = static_features.drop(columns=['spatial_id', 'x', 'y'])
    rep_static_features = pd.DataFrame(pd.np.tile(rep_static_features, (num_time, 1)))
    rep_static_features.columns = static_features.columns[3:]

    x_col = {'global_id': global_id, 'year': year, 'section': section,
             'spatial_id': spatial_id, 'x': x_coord, 'y': y_coord,
             #'temp': temp, 'precip': precip, #'gpp': gpp_flat,
             'current_patrol_effort': current_effort,
             'past_patrol_effort': past_effort}
    x_complete = pd.DataFrame(data=x_col)
    x_complete = pd.concat([x_complete, rep_static_features], axis=1)



    # combine all Y features
    y_val = illegal_activity.values.T.flatten()
    y_col = {'global_id': global_id, 'year': year, 'section': section,
             'spatial_id': spatial_id, 'x': x_coord, 'y': y_coord,
             'illegal_activity': y_val}
    y_complete = pd.DataFrame(data=y_col)


    # remove zero patrol
    rows_to_remove = np.where(current_effort == 0)[0]

    print('  {} / {} data points with zero patrol effort ({:.2f})'.format(
        len(rows_to_remove), x_complete.shape[0],
        (len(rows_to_remove) / x_complete.shape[0]) * 100))
    print('  {} / {} positive labels all ({:.2f}%)'.format(
        y_val.sum(), y_val.size, (y_val.sum() / y_val.size) * 100))

    x_crop = x_complete.drop(rows_to_remove)
    y_crop = y_complete.drop(rows_to_remove)

    y_crop_val = y_crop['illegal_activity']
    print('  {} / {} positive labels ({:.2f}%) - after crop'.format(
        y_crop_val.sum(), y_crop_val.size,
        (y_crop_val.sum() / y_crop_val.size) * 100))

    x_filename = '{}/output/{}_{}month/{}_x.csv'.format(filepath, out_prefix, num_months, out_prefix)
    y_filename = '{}/output/{}_{}month/{}_y.csv'.format(filepath, out_prefix, num_months, out_prefix)
    print('  writing out x file: {}'.format(x_filename))
    print('  writing out y file: {}'.format(y_filename))
    x_crop.to_csv(x_filename)
    y_crop.to_csv(y_filename)

    plot_patrol_effort(filepath, num_months, out_prefix, x_crop['current_patrol_effort'].values)

    return x_crop, y_crop



# plot number of points at different thresholds of patrol effort
def plot_patrol_effort(filepath, num_months, out_prefix, effort):
    import matplotlib.pyplot as plt

    num_points = 100
    count = np.zeros(num_points)
    thresholds = np.linspace(0, np.max(effort), num_points)

    for i in range(num_points):
        count[i] = len(np.where(effort > thresholds[i])[0])

    plt.plot(thresholds, count)
    plt.xlabel('Patrol effort thresold')
    plt.ylabel('Number of points')
    plt.title('Threshold plot - {} season'.format(out_prefix))

    plt.savefig('{}/output/{}_{}month/threshold_plot.png'.format(filepath, out_prefix, num_months))
    plt.close()
