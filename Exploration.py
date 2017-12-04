import numpy as np
import pandas as pd
import joblib


# Data reading
train = pd.read_csv('CAX_TrainingData_McK.csv')
target = train[['offer_gk','driver_response']]
# joblib.dump(target, 'target')
test = pd.read_csv('CAX_TestData_McK.csv')
sample_submission = pd.read_csv('McK_SubmissionFormat.csv')

# Concatenating train and test data in order to produce features in one way
df_all = pd.concat([train, test], axis = 0).reset_index(drop = True)

df_all = df_all.rename(columns = {
        'distance_km': 'distance',
        'duration_min': 'time',
        'hour_key': 'hour',
        'weekday_key': 'weekday',
    }
)
df_all.drop(['driver_response'], axis = 1, inplace = True, errors='ignore')

# Replacing -1 in distances column to median value
mask_distance_unknown = df_all['distance'] == -1
distances = df_all.loc[~mask_distance_unknown, 'distance']
distance_to_replace = np.percentile(distances, 50)
df_all.loc[mask_distance_unknown, 'distance'] = distance_to_replace

# Replacing -1 in time column to median value
mask_time_unknown = df_all['time'] == -1
times = df_all.loc[~mask_time_unknown, 'time']
time_to_replace = np.percentile(times, 50)
df_all.loc[mask_time_unknown, 'time'] = time_to_replace

# Generating features that equals some combination of time and distance
time_powers_list = 1.1 ** np.linspace(-5, 7, 6)
for time_power in time_powers_list:
    df_all['distance_div_time_power_%.2f' % time_power] = df_all['distance'] / (df_all['time'] ** time_power)


# Generating weekday features
holiday_groups = [
    [5, 6],
    [6, 0],
    [5, 6, 0],
]

for holiday_group in holiday_groups:
    holiday_group_str = [str(x) for x in holiday_group]
    colname = 'is_holiday_' + '_'.join(holiday_group_str)
    df_all[colname] = df_all['weekday'].apply(lambda x: 1 if x in holiday_group else 0)

# There are some strange places in data
df_all['is_strange_place'] = 1 * (df_all['origin_order_latitude'] < 15)


# Calculateing distance between driver and passanger
from geopy.distance import vincenty
distance_colnames = ['distance_driver_order']

df_all['distance_driver_order'] = df_all[['origin_order_latitude', 'origin_order_longitude', 'driver_latitude', 'driver_longitude']].apply(
    lambda rec: vincenty(
        (rec['origin_order_latitude'], rec['origin_order_longitude']),
        (rec['driver_latitude'], rec['driver_longitude'])
    ).km, axis = 1
)

# Listing the most intresting points in Moscow
points_list = [
    (55.753315, 37.622500), # center
    (55.783199, 37.620724), # north
    (55.774879, 37.653683), # 3 vokzals
    (55.735964, 37.674282),
    (55.719495, 37.618149),
    (55.744292, 37.556351),
    (55.771202, 37.570427),
    (55.415471, 37.893784), # domodedovo
    (55.605822, 37.289434), # vnukovo
    (55.962587, 37.417308), # sheremetievo
    (55.559388, 38.114864), # jukovky
]

# Now let's calculate distances from driver and passanger to these points. It can take ~30 minutes
for num_point, point in enumerate(points_list):
    colname = f'distance_driver_point_{num_point}'
    distance_colnames.append(colname)
    df_all[colname] = df_all[['driver_latitude', 'driver_longitude']].apply(
        lambda rec: vincenty(
            (rec['driver_latitude'], rec['driver_longitude']),
            point
        ).km, axis = 1
    )
    colname = f'distance_order_point_{num_point}'
    distance_colnames.append(colname)
    df_all[colname] = df_all[['origin_order_latitude', 'origin_order_longitude']].apply(
        lambda rec: vincenty(
            (rec['origin_order_latitude'], rec['origin_order_longitude']),
            point
        ).km, axis = 1
    )
    print(f'point {num_point} done')

# joblib.dump(df_all[['offer_gk'] + distance_colnames], 'df_geo')

# df_geo = joblib.load('df_geo')
# df_all = df_all.merge(df_geo, on = 'offer_gk', how = 'inner')

def cycle_it(df, min, max, n, name):
    """
    This function increase low values of cyclic features because lightgbm can find new pattern that way
    :param df: pd.Series, needed column
    :param min: min value in that column
    :param max: max value in that column
    :param n: n cycles to run
    :param name: prefix name for columns in returned dataframe
    :return: pd.DataFrame, dataframe with cycled columns of initial pd.Series
    """
    if n == 1: return df
    data = np.array([df]*n).T
    names = np.zeros(n).astype('object'); names[0] = name
    for k in range(1, n):
        bound = float(min)+ (float(max)-float(min))*k/n
        data[df <= bound,k]+=max
        names[k] = name + '_' + str(round(bound,2))
    return pd.DataFrame(data, columns=names)

# Additional_features
parts = dict()
parts['hour_cycled'] = cycle_it(df_all['hour'], 0, 24, 3, 'hour_cycled')
parts['weekday_cycled'] = cycle_it(df_all['weekday'], 0, 7, 3, 'weekday_cycled')
# One hot encoded weekday
parts['weekday_ohe'] = pd.get_dummies(df_all['weekday'], prefix='weekday')

# Info about driver's orders number
driver_n_orders = train['driver_gk'].value_counts().reset_index()
driver_n_orders = driver_n_orders.rename(columns = {'driver_gk': 'n_orders', 'index': 'driver_gk'})

# Mapping offer class to another scale
dict_class_to_num = {
    'Economy': 0,
    'Standard': 1,
    'Premium': 2,
    'Delivery': 1,
    'Kids': 1,
    'VIP': 3,
    'XL': 1,
    'VIP+': 3,
    'Test': 1,
}
df_all['offer_class_num'] = df_all['offer_class_group'].apply(lambda x: dict_class_to_num[x])

agg_result = df_all[['order_gk', 'offer_gk']].groupby(['order_gk'], as_index = False).agg(len)
agg_result = agg_result.rename(columns = {'offer_gk': 'offer_count'})

# Merging features
df_all = df_all.merge(agg_result, on = ['order_gk'], how = 'inner')

for key, value_to_concat in parts.items():
    df_all = pd.concat([df_all, value_to_concat], axis = 1)

df_all.drop(['order_gk'], axis = 1, inplace = True, errors = 'ignore')

# Another driver statistics
train_driver = df_all.loc[df_all['offer_gk'].isin(train['offer_gk']), :]
train_driver = train_driver.merge(target, on = 'offer_gk', how = 'inner')

cat_columns = [
    'weekday',
    'offer_class_group',
    'ride_type_desc',
]

float_columns = [
    'driver_latitude',
    'driver_latitude',
    'driver_longitude',
    'origin_order_latitude',
    'origin_order_longitude',
    'distance',
    'time',
    'distance_div_time_power_0.62',
    'distance_div_time_power_0.78',
    'distance_div_time_power_0.98',
    'distance_div_time_power_1.23',
    'distance_div_time_power_1.55',
    'distance_div_time_power_1.95',
] + distance_colnames

bin_cols = list(set(train_driver.columns) - set(cat_columns) - set(float_columns) - set(['driver_gk', 'offer_gk']))

# Calculating different aggregating functions depending on type of feature
# columns_driver_agg = list(set(train_driver.columns) - set(cat_columns))
driver_agg_bin = train_driver[bin_cols + ['driver_gk']].groupby(['driver_gk'], as_index = False).agg(np.mean)
driver_agg_float = train_driver[float_columns + ['driver_gk']].groupby(['driver_gk'], as_index = False).agg(np.median)

driver_agg = driver_agg_bin.merge(driver_agg_float, on = 'driver_gk', how = 'inner')
colnames_new = ['driver_stats_' + colname if colname not in ['driver_gk'] else colname for colname in driver_agg.columns ]
driver_agg.columns = colnames_new
# driver_agg['driver_stats_driver_response']
df_all = df_all.merge(driver_agg, on = 'driver_gk', how = 'inner')

# Encoding categorial columns
columns_to_encode = [
    'offer_class_group',
    'ride_type_desc',
]

for colname in columns_to_encode:
    df_all[colname] = pd.factorize(df_all[colname])[0]

# Difference for distance features
for point_num in range(11):
    colname = f'distance_order_driver_diff_{point_num}'
    colname_driver = f'distance_driver_point_{point_num}'
    colname_order = f'distance_order_point_{point_num}'

    df_all[colname] = df_all[colname_order] - df_all[colname_driver]

target = joblib.load('target')

# Splitting data into train and validation
data_train = df_all.merge(target, on = 'offer_gk', how = 'inner')
train_new = data_train.drop(['driver_response'], axis = 1)
train_new.drop(['offer_gk'], axis = 1, inplace = True, errors = 'ignore')
target_new = data_train['driver_response']

mask_is_train = df_all['offer_gk'].isin(target['offer_gk'])
test_new = df_all.loc[~mask_is_train, :]
test_new = test_new.set_index(['offer_gk'])

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_new, target_new, test_size=0.2, random_state=1)

# Training a lightgbm model
import lightgbm as lgb

lgb_train = lgb.Dataset(x_train, y_train)
lgb_val = lgb.Dataset(x_val, y_val)

params = {
    'objective':'binary',
    'verbosity':1,
    'metric':'auc',
    'num_threads':-1,
    'min_data_in_leaf': 10,
}

params['num_trees'] = 2000
params['num_leaves'] = 511
params['max_depth'] = 14
params['feature_fraction'] = 0.9
params['bagging_fraction'] = 0.9
params['learning_rate'] = 0.01

bst = lgb.train(
    params,
    lgb_train,
    valid_sets = [lgb_val],
    early_stopping_rounds = 10,
    verbose_eval = 10,
)

predict_test = bst.predict(test_new)
predictions = pd.DataFrame(predict_test, index=test_new.index)

sample_submission = sample_submission.set_index(['offer_gk'])
sample_submission.loc[test_new.index, 'driver_response'] = predictions.values.ravel()
sample_submission = sample_submission.reset_index()
# sample_submission.to_csv('complicated_submission_1.csv', index = False)




# # Some feature importance code, not relevant
# n_top_features_to_plot = 30
# lgb_imps, lgb_names = zip(*sorted(zip(bst.feature_importance(),bst.feature_name()), reverse=False))
#
# lgb_imps = lgb_imps[-n_top_features_to_plot:]
# lgb_names = lgb_names[-n_top_features_to_plot:]
#
# ind = np.arange(len(lgb_imps))
# plt.figure(figsize=(15,9))
# plt.barh(ind,lgb_imps)
# plt.yticks(ind,lgb_names);
# plt.savefig('lightgbm_imp_1.pdf')






