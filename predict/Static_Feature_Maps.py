import numpy as np
import os
import util
import argparse
import pandas as pd

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == "__main__":
    post = True
    plot_title = True
    plot_patrol_effort = True
    plot_illegal_activity = True
    plot_patrol_post = False

    parser = argparse.ArgumentParser(description='Bagging Cross Validation Blackbox function')
    parser.add_argument('-r', '--resolution', default=1000, help='Input the resolution scale')
    parser.add_argument('-p', '--park', help='Input the park name', required=True)
    parser.add_argument('-c', '--category', default='All', help='Input the category')

    args = parser.parse_args()

    resolution = int(args.resolution)
    park = args.park
    category = args.category

    directory = './{0}_datasets/resolution/{1}m/input'.format(park, str(resolution))
    output_directory = './{0}_datasets/resolution/{1}m/output'.format(park, str(resolution))
    patrol_post_path = './{0}_datasets/PatrolPosts.csv'.format(park)

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(output_directory + "/Maps/{0}".format(category)):
        os.makedirs(output_directory + "/Maps/{0}".format(category))

    # test_year, test_quarter = util.test_year_quarter_by_park(park)

    data = pd.read_csv(directory + '/' + 'allStaticFeat.csv')
    if plot_patrol_effort:
        patrol_data = pd.read_csv(directory + '/' + '{0}_X.csv'.format(category))
    if plot_illegal_activity:
        illegal_data = pd.read_csv(directory + '/' + '{0}_Y.csv'.format(category))
    #data = pd.read_csv(directory + '/' + 'boundary_cropped500.csv')

    # --------------------- shifting --------------------------
    x_min=int(np.min(data['x']))
    y_min=int(np.min(data['y']))

    data['x'] = (data['x'] - x_min) / resolution
    data['y'] = (data['y'] - y_min) / resolution

    data['x'] = data['x'].astype(int)
    data['y'] = data['y'].astype(int)

    if plot_patrol_effort:
        patrol_data['x'] = (patrol_data['x'] - x_min) / resolution
        patrol_data['y'] = (patrol_data['y'] - y_min) / resolution

        patrol_data['x'] = patrol_data['x'].astype(int)
        patrol_data['y'] = patrol_data['y'].astype(int)

    if plot_illegal_activity:
        illegal_data['x'] = (illegal_data['x'] - x_min) / resolution
        illegal_data['y'] = (illegal_data['y'] - y_min) / resolution

        illegal_data['x'] = illegal_data['x'].astype(int)
        illegal_data['y'] = illegal_data['y'].astype(int)

    # --------------------- feature map -----------------------
    static_feature_options = list(data.columns[3:]) + ["Null"]

    if plot_patrol_effort:
        static_feature_options = static_feature_options + ["Past Patrol"]
    if plot_illegal_activity:
        static_feature_options = static_feature_options + ["Illegal Activity"]

    for static_feature_option in static_feature_options:
        print("Processing feature: {0} ...".format(static_feature_option))

        x_max=int(np.max(data['x']))
        y_max=int(np.max(data['y']))

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)
        bounds=[-1,0]
        norm = colors.BoundaryNorm(bounds, cmapm.N)

        gridmap = [[0 for x in range(x_max+1)] for y in range(y_max+1)]

        if static_feature_option == "Past Patrol":
            feature_map = np.ones((y_max+1, x_max+1)) / float(10)
            for index, row in data.iterrows():
                gridmap[int(row['y'])][int(row['x'])] = 1
            for index, row in patrol_data.iterrows():
                feature_map[int(row['y'])][int(row['x'])] += row['currentPatrolEffort']

        elif static_feature_option == "Illegal Activity":
            feature_map = np.zeros((y_max+1, x_max+1))
            feature_df = pd.DataFrame(columns=['x', 'y', 'value'])

            for index, row in data.iterrows():
                gridmap[int(row['y'])][int(row['x'])] = 1
            for index, row in illegal_data.iterrows():
                if row['Illegal_Activity']:
                    feature_map[int(row['y'])][int(row['x'])] += 1
            for x in range(feature_map.shape[0]):
                for y in range(feature_map.shape[1]):
                    if feature_map[x][y]:
                        feature_df = feature_df.append(pd.DataFrame([[x, y, feature_map[x][y]]], columns=['x', 'y', 'value']))
        elif static_feature_option == "Null":
            feature_map = np.zeros((y_max+1, x_max+1))
            for index, row in data.iterrows():
                gridmap[int(row['y'])][int(row['x'])] = 1
                feature_map[int(row['y'])][int(row['x'])] = 0

        else:
            feature_map = np.zeros((y_max+1, x_max+1))
            for index, row in data.iterrows():
                gridmap[int(row['y'])][int(row['x'])] = 1
                feature_map[int(row['y'])][int(row['x'])] = row[static_feature_option]

        gridmap = np.ma.masked_where(gridmap == 1, gridmap)


        # --------------- feature map ----------------
        # print(gridmap)
        # print(feature_map[gridmap])

        max_value = np.max(np.ma.masked_where(gridmap==0, feature_map))
        # max_value = 0.25
        min_value = np.min(np.ma.masked_where(gridmap==0, feature_map))
        # min_value = 0.05
        if static_feature_option == "Past Patrol":
            a = plt.imshow(np.flipud(feature_map), interpolation='none', cmap="Greens", extent=[0,x_max+1,0,y_max+1], vmin=min_value, vmax=max_value, norm=LogNorm())
        elif static_feature_option == "Illegal Activity":
            my_cmap = matplotlib.cm.get_cmap('gist_heat_r')
            my_cmap.set_under('w')
            bounds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            # a = plt.imshow(np.flipud(feature_map), interpolation='none', cmap=my_cmap, extent=[0,x_max+1,0,y_max+1], vmin=0.5, vmax=max_value, norm=norm)
            a = plt.scatter(feature_df['y'], feature_df['x'], c=feature_df['value'], s=feature_df['value']*20, alpha=0.8, cmap=my_cmap, vmin=0, vmax=10, norm=norm)
        elif static_feature_option == "Null":
            pass
        else:
            a = plt.imshow(np.flipud(feature_map), interpolation='none', cmap="Greens", extent=[0,x_max+1,0,y_max+1], vmin=min_value, vmax=max_value)

        if np.min(gridmap) == 0:
            h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[0,x_max+1,0,y_max+1])
        #plt.set(h, 'AlphaData', gridmap)

        plt.colorbar(a)
        #plt.xticks(np.arange(0,x_max+1))
        #plt.yticks(np.arange(0,y_max+1))
        #if self.resolution < 500 or len(data) > 1000:
        #plt.grid(ls='solid', lw=0)
        #else:
        #    plt.grid(ls='solid')
        if not post == None:
            if plot_title:

                plt.title(static_feature_option)
                #plt.xticks(np.arange(0,mx+1),[self.min_xval+resolution*i for i in range(mx+1)], rotation=60)
                plt.xlabel("x", fontsize=6)
                #plt.yticks(np.arange(0,my+1),[self.min_yval+resolution*i for i in range(my+1)])
                plt.ylabel("y", fontsize=6)
            if plot_patrol_post:
                patrol_post_data = pd.read_csv(patrol_post_path)
                patrol_post_data['X'] = (patrol_post_data['X'] - x_min) / resolution
                patrol_post_data['Y'] = (patrol_post_data['Y'] - y_min) / resolution

                patrol_post_data['X'] = patrol_post_data['X'].astype(int)
                patrol_post_data['Y'] = patrol_post_data['Y'].astype(int)

                for index, row in patrol_post_data.iterrows():
                    sx = row['X']
                    sy = row['Y']
                    plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

            plt.savefig(output_directory + "/Maps/{0}/{1}.png".format(category, static_feature_option))
            plt.close()
            #plt.show()
        else:
            plt.show()
