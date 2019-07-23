from visualize_maps import *


if __name__ == "__main__":

    KERNEL_WIDTH = 5
    PATROL_EFFORT_THRESHOLD = .6
    SAVE = True # whether or not to save output images and files

    NORMALIZE_MAPS = False

    resolution = 1000


    park = 'belum'
    method = 'gp'
    num_classifiers = 5
    # input_filepath = './input/{}/'.format(park)
    # output_filepath = './output/{}/'.format(park)

    # data_input_path = '../inputs/ICDE_input'
    # data_input_path = './output/belum_june2019'
    data_input_path = '../preprocess_consolidate/belum_june2019/1000/output'

    # path = '../preprocess_consolidate/belum_traponly_combined/1000/output/'
    output_path = './output/{}_vis/'.format(park)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    print('  reading in shapefile .shp data...')

    #######################################################
    # SWS
    #######################################################
    if park == 'SWS':
        year = 2018

        # crs_in: "+proj=longlat +datum=WGS84"
        crs_out = '+proj=utm +zone=48 +north +datum=WGS84 +units=m +no_defs'


        input_filepath = '../inputs/sws/'
        boundary_shape = gpd.read_file(input_filepath + 'boundary/CA.shp')
        # patrol_blocks  = gpd.read_file(input_filepath + 'patrolBlocks/PATRL.shp')
        # core_zone      = gpd.read_file(input_filepath + 'coreZone/SWS_core_zone.shp')

        # patrol_posts   = pd.read_csv(input_filepath + 'patrol_posts.csv')
        # patrol_posts['geometry'] = patrol_posts.apply(lambda z: Point(z['x'], z['y']), axis=1)
        # patrol_posts = gpd.GeoDataFrame(patrol_posts)

        rivers = gpd.read_file(input_filepath + 'rivers/Stream_20in_20MPF.shp')
        roads = gpd.read_file(input_filepath + 'roads/roads old/Road_20in_20MPF.shp')

        # set CRS for shapefiles
        boundary_shape = boundary_shape.to_crs(crs_out)
        # patrol_blocks = patrol_blocks.to_crs(crs_out)
        # core_zone = core_zone.to_crs(crs_out)

        shapes = {'boundary': boundary_shape, 'roads': roads,
                  #'patrol_posts': patrol_posts,
                  #'rivers': rivers,
                  #'core_zone': core_zone, 'patrol_blocks': patrol_blocks
                  }


        # pred_filename = '../../cambodia/field tests/2019 06/predictions_SWS_2019_method_gp_10.csv'
        # var_filename = '../../cambodia/field tests/2019 06/variances_SWS_2019_method_gp_10.csv'



    #######################################################
    # QENP
    #######################################################
    elif park == 'QENP':
        year = 2016

        # crs_in: "+proj=longlat +datum=WGS84"
        crs_out = '+proj=utm +zone=36 +south +datum=WGS84 +units=m +no_defs'

        input_filepath = '../inputs/QENP/'
        boundary_shape = gpd.read_file(input_filepath + 'boundary/CA.shp')

        # patrol_posts   = pd.read_csv(input_filepath + 'patrol_posts.csv')
        # patrol_posts['geometry'] = patrol_posts.apply(lambda z: Point(z['x'], z['y']), axis=1)
        # patrol_posts = gpd.GeoDataFrame(patrol_posts)

        # rivers = gpd.read_file(input_filepath + 'water/qenp_riversbigWGS84.shp')
        water = gpd.read_file(input_filepath + 'water/WaterClip.shp')

        shapes = {'boundary': boundary_shape, 'water': water
                  # 'patrol_posts': patrol_posts, 'rivers': rivers,
                  }

    #######################################################
    # MFNP
    #######################################################
    elif park == 'MFNP':
        year = 2016

        # crs_in: "+proj=longlat +datum=WGS84"
        crs_out = '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs'

        input_filepath = '../inputs/MFNP/'
        boundary_shape = gpd.read_file(input_filepath + 'boundary/CA.shp')

        # patrol_posts   = pd.read_csv(input_filepath + 'patrol_posts.csv')
        # patrol_posts['geometry'] = patrol_posts.apply(lambda z: Point(z['x'], z['y']), axis=1)
        # patrol_posts = gpd.GeoDataFrame(patrol_posts)

        water = gpd.read_file(input_filepath + 'water/mfca_water.shp')

        shapes = {'boundary': boundary_shape, #'patrol_posts': patrol_posts,
                  'water': water}

    elif park == 'belum':
        year = 2018
        ext = '_all'
        # ext = '_group1'

        crs_out = '+proj=utm +zone=48 +datum=WGS84 +units=m +no_defs'

        input_filepath = '../inputs/belum/'
        boundary_shape = gpd.read_file(input_filepath + 'clean_crs/boundary.shp')
        water = gpd.read_file(input_filepath + 'clean_crs/lake_clipped.shp')
        rivers = gpd.read_file(input_filepath + 'clean_crs/rivers_clipped.shp')
        thailand_buffer = gpd.read_file(input_filepath + 'clean_crs/thailand_buffer.shp')


        shapes = {'boundary': boundary_shape, 'water': water, 'rivers': rivers, 'buffer': thailand_buffer}

        x_filename = data_input_path + '/all_3month/all_x.csv'
        y_filename = data_input_path + '/all_3month/all_y' + ext + '.csv'
        static_features_filename = data_input_path + '/static_features.csv'

        pred_filename = './output/belum_june2019/predictions_belum' + ext + '_2018_method_{}_{}.csv'.format(method, num_classifiers)
        var_filename = './output/belum_june2019/variances_belum' + ext + '_2018_method_{}_{}.csv'.format(method, num_classifiers)



    #######################################################
    # load data
    #######################################################

    # pred_filename = './output/predictions_sws_2018_method_{}.csv'.format(method)
    # var_filename = './output/predictions_sws_2018_method_{}.csv'.format(method)

    # x_filename = '{}/{}_{}/All_X.csv'.format(data_input_path, park, year)
    # y_filename = '{}/{}_{}/All_Y.csv'.format(data_input_path, park, year)
    # static_features_filename = '{}/{}_allStaticFeat.csv'.format(data_input_path, park)

    # pred_filename = './output/ref_{}_{}/predictions_{}_{}_method_{}_{}.csv'.format(park, year, park, year, method, num_classifiers)
    # var_filename = './output/ref_{}_{}/variances_{}_{}_method_{}_{}.csv'.format(park, year, park, year, method, num_classifiers)


    vis = Visualize(resolution, data_input_path, output_path)
    vis.get_gridmap(static_features_filename)

    historic_patrol_effort_map = vis.get_past_patrol(x_filename)
    historic_illegal_activity_map = vis.get_illegal_activity(y_filename)

    vis.save_map_with_features('{}_{}_effort'.format(park, year),
                                historic_patrol_effort_map, shapes, crs_out, cmap='Blues', log_norm=True)

    vis.save_map_with_features('{}_{}_activity'.format(park, year),
                                historic_illegal_activity_map, shapes, crs_out, cmap='Reds')

    # vis.save_map(historic_patrol_effort_map, 'patrol_effort', cmap='Greens', log_norm=False, plot_patrol_post=False)
    # vis.save_map(historic_illegal_activity_map, 'illegal_activity', cmap='Reds', log_norm=False, plot_patrol_post=False)

    # print('load static features from file {}...'.format(static_features_filename))
    # static_feats = vis.load_static_feature_maps(static_features_filename)
    #
    # for feat in static_feats:
    #     vis.save_map(static_feats[feat], feat, cmap='Greens', log_norm=False, plot_patrol_post=False)


    #######################################################
    # read in riskmaps
    #######################################################

    # read_map_thresholds(vis, shapes, pred_filename, 'predictions_{}'.format(method), 'Reds', crs_out, normalize=NORMALIZE_MAPS)
    read_map_thresholds(vis, shapes, var_filename, 'variances_{}'.format(method), 'Greens', crs_out, normalize=NORMALIZE_MAPS)

    #######################################################
    # visualize difference between plots
    #######################################################
    # diff_filename = 'output/waterhole_differences.csv'
    #
    # # open riskmap and save image
    # print('reading in difference predictions from file {}...'.format(risk_filename))
    # col_heads, riskmaps = vis.get_riskmaps_from_csv(diff_filename)
    #
    # # riskmap_min_value = min([np.min(riskmap) for riskmap in riskmaps])
    # # riskmap_max_value = max([np.max(riskmap) for riskmap in riskmaps])
    #
    # riskmap_min_value = -0.5
    # riskmap_max_value = 0.5
    #
    # if SAVE:
    #     for i in range(len(riskmaps)):
    #         vis.save_map(riskmaps[i], 'waterhole_differences_{}'.format(col_heads[i]), cmap='RdBu_r', log_norm=False, min_value=riskmap_min_value, max_value=riskmap_max_value, plot_patrol_post=False)
