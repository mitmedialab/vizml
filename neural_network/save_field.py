# Imports
import ml.evaluate as evaluate
import ml.util as util
import ml.train as train
from helpers.processing import *
from helpers.analysis import *
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import scipy as sc
import numpy as np
import time
import gc
import os
import sys
sys.path.insert(0, '..')


RANDOM_STATE = 42

features_directory = '../features/processed'
saves_directory = './saves'
num_datapoints = None  # None if you want all
max_fields = 5  # number of max fields we will consider using in features
# there's only enough memory to do max_fields=5

# this script must be run to save matrices, before running train_field.py
# takes in all field_level files and combines examples such that all examples have the same feature size
# and each specific feature is in the same location
# used for transfer learning


def load_features():
    features_array = []
    outcomes_array = []
    features_sizes = [0]

    # as we load each by_field/features_{}
    # we record the shape of each Panda Dataframe returned
    # and combine the df into one giant matrix

    # each df has a common set of agg. features, and differ in the number of field features
    # so we merge the df on the agg. features, and impute the missing features
    # with the average value from other examples

    # after imputing is done, we re-separate our giant df by number of columns
    # such that all examples with the same number of columns in the original
    # data are saved to the same file

    for num_fields in range(1, max_fields + 1):
        features_df_file_name = 'by_field/features_{}.csv'.format(num_fields)
        outcomes_df_file_name = 'by_field/outcomes_{}.csv'.format(num_fields)
        features_df = pd.read_csv(
            os.path.join(
                features_directory,
                features_df_file_name),
            nrows=num_datapoints)
        outcomes_df = pd.read_csv(
            os.path.join(
                features_directory,
                outcomes_df_file_name))
        features_sizes.append(features_df.shape[0])
        features_array.append(features_df)
        outcomes_array.append(outcomes_df)

    # here we combine features_array and outcomes_array, and delete the original arrays to save memory
    # any missing features are first filled in with N/A
    features_df = pd.concat(features_array, axis=0, ignore_index=True)
    outcomes_df = pd.concat(outcomes_array, axis=0, ignore_index=True)
    del features_array, outcomes_array

    # drop the fid, impute any N/A entries, and readd fid
    features_id_column = features_df[['fid']]
    features_df = features_df.drop(
        ['fid'], axis=1, inplace=False, errors='ignore')
    features_df = process_features_df(features_df)
    features_df = pd.concat([features_df, features_id_column], axis=1)

    # add feature representing num_fields for each training example
    # and concat it with our features_df
    num_fields_array = []
    for num_fields in range(1, max_fields + 1):
        np_array = np.zeros((features_sizes[num_fields],), dtype=np.int64)
        np_array.fill(num_fields)
        num_fields_array.append(np_array)
    num_fields_array = np.concatenate(num_fields_array)

    assert num_fields_array.shape[0] == features_df.shape[0]
    num_fields_array = pd.DataFrame(
        {"special_original_num_fields": num_fields_array})
    features_df = pd.concat([features_df, num_fields_array], axis=1)

    # process outcomes
    outcome_variable_name = 'all_one_trace_type'
    outcomes = ['line', 'scatter', 'bar']
    outcomes_df_subset = outcomes_df[outcomes_df[outcome_variable_name].isin(
        outcomes)][['fid', outcome_variable_name]]

    # Join features and outcomes
    final_df = pd.merge(features_df, outcomes_df_subset, on='fid', how='inner')
    final_df = final_df.drop(['fid'], axis=1, inplace=False, errors='ignore')
    del features_df, outcomes_df_subset

    # filter out examples with same num fields
    # drop our special_original_num_fields column
    # and save the matrices to disk
    for num_fields in range(1, max_fields + 1):
        X_with_field = final_df[final_df['special_original_num_fields'] == num_fields]
        X_with_field = X_with_field.drop(
            ['special_original_num_fields'],
            axis=1,
            inplace=False,
            errors='ignore')
        X = X_with_field.iloc[:, :-1]
        y = X_with_field.iloc[:, -1]
        y = pd.get_dummies(y).values.argmax(1)

        res = RandomOverSampler(random_state=RANDOM_STATE)
        X, y = res.fit_sample(X, y)
        X, y = util.unison_shuffle(X, y)
        util.save_matrices_to_disk(
            X, y, [0.1, 0.1], saves_directory, 'field_' + str(num_fields), num_datapoints)


def main():
    load_features()


if __name__ == '__main__':
    main()
