# PAWS preprocessing code
# Lily Xu
# April 2019

from consolidate import *


resolution = 1000
park = 'SWS_2013-Apr2019'
filepath = './{}/{}'.format(park, resolution)

start_year = 2013
end_year = 2019

static_feature_names = ['roads', 'route76', 'crossroads',
    'boundary', 'vietnam', 'patrol_posts', 'villages',
    'forest_cover', 'slope_map',
    'waterholes', 'rivers_permanent', 'rivers_intermittent',
    'community_zone', 'conservation_zone', 'core_zone', 'sustainable_use_zone',
    'all_animals', 'banteng', 'elephant', 'muntjac', 'wild_pig']

dry_months   = [1, 2, 3, 4, 11, 12]  # November through April
rainy_months = [5, 6, 7, 8,  9, 10]  # May through October

num_months = 2


# TODO: bump climate back one section



print()
print('-----------------------------------------------')
print('dry season, num_months = {}'.format(num_months))
print('-----------------------------------------------')
consolidate_pipeline(filepath, static_feature_names, start_year, end_year, num_months, 'dry', dry_months)



# static_features = combine_static_features(filepath, static_feature_names)
#
# sections_per_year = len(dry_months) // num_months
# dry_activity = process_human_activity(filepath, start_year, end_year, num_months, 'dry', dry_months)
# dry_effort   = process_patrol_effort(filepath, start_year, end_year, num_months, 'dry', dry_months)
# dry_climate  = process_climate(filepath, start_year, end_year, num_months, 'dry', dry_months)
# dry_x, dry_y = combine_data(static_features, dry_activity, dry_effort, dry_climate, filepath, num_months, sections_per_year, start_year, end_year, 'dry')


print()
print('-----------------------------------------------')
print('rainy season, num_months = {}'.format(num_months))
print('-----------------------------------------------')
consolidate_pipeline(filepath, static_feature_names, start_year, end_year, num_months, 'rainy', rainy_months)



# sections_per_year = len(rainy_months) // num_months
# rainy_activity = process_human_activity(filepath, start_year, end_year, num_months, 'rainy', rainy_months)
# rainy_effort   = process_patrol_effort(filepath, start_year, end_year, num_months, 'rainy', rainy_months)
# rainy_climate  = process_climate(filepath, start_year, end_year, num_months, 'rainy', rainy_months)
# rainy_x, rainy_y = combine_data(static_features, rainy_activity, rainy_effort, rainy_climate, filepath, num_months, sections_per_year, start_year, end_year, 'rainy')


print()
print('-----------------------------------------------')
print('all seasons, num_months = {}'.format(num_months))
print('-----------------------------------------------')
consolidate_pipeline(filepath, static_feature_names, start_year, end_year, num_months, 'all')


# sections_per_year = 12 // num_months
# all_activity = process_human_activity(filepath, start_year, end_year, num_months)
# all_effort   = process_patrol_effort(filepath, start_year, end_year, num_months)
# all_climate  = process_climate(filepath, start_year, end_year, num_months)
# all_x, all_y = combine_data(static_features, all_activity, all_effort, all_climate, filepath, num_months, sections_per_year, start_year, end_year)
