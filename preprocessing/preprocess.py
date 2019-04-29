# !/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import pickle
from os.path import join
from shutil import copy2
from sklearn.utils import shuffle as sk_shuffle
import pandas as pd

from preprocessing.deduplication import kd_threshold_deduplication, one_per_user_deduplication
from preprocessing.impute import DataFrameImputer
from experiment_data.ground_truth_fids_99 import ground_truth_fids_99
from helpers.analysis import *
from helpers.processing import *
from helpers.util import *

options = {
    'by_field': False,
    'winsor': 99,
    'scaler': 'standard',
    'save_transforms': True,
    'secondary_deduplication': 'one_per_user',
    'percentage_threshold': {
        'dataset': 0.001,
        'field': 0.001
    }
}

options['subset_cols'] = {
    'dataset': [
        'exists-agg-num',
        'length-agg-mean',
        'general_type_is_c-agg-percentage',
        'general_type_is_q-agg-percentage',
        'general_type_is_t-agg-percentage',
        'unique_percent-agg-mean',
        'mean_value_length-agg-mean',
        'mean-agg-mean',
        'var-agg-mean',
        'normalized_edit_distance-agg-mean'
    ],
    'field': [
        'length',
        'general_type_is_q',
        'general_type_is_t',
        'general_type_is_c',
        'mean',
        'median',
        'var',
        'unique_percent',
        'mean_value_length',
        'field_name_length',
    ]
}


parser = argparse.ArgumentParser(description='Features to process')
parser.add_argument(
    'features_dir',
    type=str,
    help='dir containing unprocessed features')
args = parser.parse_args()

features_dir = args.features_dir
output_dir = os.path.join(features_dir, 'processed')
by_field_dir = 'by_field'
fids_to_separate = [x['fid'] for x in ground_truth_fids_99]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if options['by_field'] and not os.path.exists(join(output_dir, by_field_dir)):
    os.mkdir(join(output_dir, by_field_dir))


def get_scaler(scaler='standard'):
    if scaler == 'standard':
        scaler_instance = StandardScaler()
    if scaler == 'robust':
        scaler_instance = RobustScaler()
    if scaler == 'minmax':
        scaler_instance = MinMaxScaler()
    return scaler_instance


def process_features_df(df,
                        id_fields=['fid'],
                        scaler='standard',
                        shuffle=True,
                        winsor=99,
                        drop_null=True,
                        fit_imputer=None,
                        fit_scaler=None,
                        drop_duplicates=True,
                        secondary_deduplication='threshold',
                        percentage_threshold={
        'dataset': 0.0001,
        'field': 0.0001
                            },
                        subset_cols=[],  # To use for deduplication
                        included_cols=[],
                        excluded_cols=[],
                        numeric_cols=[],
                        return_transforms=False,
                        included_fids=[]
                        ):

    if shuffle:
        df = sk_shuffle(df, random_state=42)

    print('\tInitial shape:', df.shape)
    initial_shape = df.shape
    if len(included_cols):
        print('\tIncluding {} columns'.format(len(included_cols)))
        df = df[included_cols]
    if len(excluded_cols):
        print('\tExcluding {} columns'.format(len(excluded_cols)))
        df.drop(excluded_cols, inplace=True, axis=1, errors='ignore')

    print('\tReplacing inf values')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_null:
        print('\tDropping null features')
        df.dropna(axis=1, how='all', inplace=True)

    print(id_fields)
    if not subset_cols:
        print('\tUsing all columns')
        subset_cols = [c for c in df.columns if c not in id_fields]

        print('\tNum subset cols', subset_cols)
    print('\tDropping exact duplicates')
    if drop_duplicates:
        df.drop_duplicates(subset=subset_cols, inplace=True)
    print('\tShape after exact duplicates: {} (yield {:.5g})'.format(
        df.shape, df.shape[0] / initial_shape[0]))

    if not len(numeric_cols):
        float_feature_names = [
            feature_name for (
                feature_name,
                dtype) in df.dtypes.iteritems() if dtype in [
                np.float32,
                np.float64]]
        integer_feature_names = [
            feature_name for (
                feature_name,
                dtype) in df.dtypes.iteritems() if dtype in [
                np.int32,
                np.int64]]
        numeric_cols = float_feature_names + integer_feature_names
        non_numeric_cols = [
            feature_name for (
                feature_name,
                dtype) in df.dtypes.iteritems() if feature_name not in numeric_cols]

    if winsor:
        print('\tWinsorizing at level', winsor)
        for numeric_col in numeric_cols:
            lower_bracket, upper_bracket = np.nanpercentile(
                df[numeric_col], [(100 - winsor), winsor])
            df.loc[df[numeric_col] < lower_bracket,
                   numeric_col] = lower_bracket
            df.loc[df[numeric_col] > upper_bracket,
                   numeric_col] = upper_bracket

    print('\tImputing missing values')
    if not fit_imputer:
        fit_imputer = DataFrameImputer().fit(df)
    df = fit_imputer.transform(df)

    if secondary_deduplication == 'one_per_user':
        if len(included_fids):
            df = df[df.fid.isin(included_fids)]
        else:
            df = one_per_user_deduplication(df)

    print('\tScaling features')
    if not fit_scaler:
        fit_scaler = get_scaler(scaler).fit(df[numeric_cols])
    df[numeric_cols] = fit_scaler.transform(df[numeric_cols])

    # Run after initial scaling
    if secondary_deduplication == 'threshold':
        num_features = len(subset_cols)

        df = kd_threshold_deduplication(
            df, subset_cols, percentage_threshold=percentage_threshold)
        # Re-scale features
        if not fit_scaler:
            fit_scaler = get_scaler(scaler).fit(df[numeric_cols])
        df[numeric_cols] = fit_scaler.transform(df[numeric_cols])

    cols_with_nulls = list(df.columns[df.isnull().any()])
    if any(cols_with_nulls):
        print('\tFields with null values:', cols_with_nulls)

    included_cols = df.columns.values

    print('\tEnd shape: {} (yield {:.5g})'.format(
        df.shape, df.shape[0] / initial_shape[0]))
    if return_transforms:
        return df, fit_imputer, fit_scaler, included_cols, numeric_cols
    else:
        return df


def main():
    print('Starting processing at', get_time())
    dirs = [features_dir]
    if options['by_field']:
        dirs.append(join(features_dir, by_field_dir))

    feature_files = [
        {
            'full_file_name': 'features_aggregate_single_pairwise.csv',
            'type': 'dataset',
            'preserve_fids': True,
            'id_fields': ['fid']
        },
        {
            'full_file_name': 'field_level_features.csv',
            'type': 'field',
            'preserve_fids': False,
            'id_fields': ['field_id', 'fid']

        }
    ]

    for d in dirs:
        for feature_file_index in [1]:
            feature_file = feature_files[feature_file_index]
            full_file_name = feature_file['full_file_name']
            feature_type = feature_file['type']
            preserve_fids = feature_file['preserve_fids']
            id_fields = feature_file['id_fields']
            file_name = full_file_name.rsplit('.', 1)[0]

            # if full_file_name.startswith('field_level_features'):
            # if 'features' in full_file_name:
            print(full_file_name)
            df = pd.read_csv(
                join(d, full_file_name),
                error_bad_lines=False,
                encoding='utf-8'
            )

            if preserve_fids:
                preserved_fids_mask = df.fid.isin(fids_to_separate)
                df_preserved_fids = df.loc[preserved_fids_mask]
                df = df[~preserved_fids_mask]
                number_of_preserved_fids = sum(
                    1 for x in preserved_fids_mask if x)
                print('Number of ground truth FIDs', number_of_preserved_fids)

            included_fids = []
            if options['secondary_deduplication'] == 'one_per_user' and feature_file['type'] == 'field':
                included_fids = pd.read_csv(join(output_dir, 'included_fids.csv'))[
                    'fid'].tolist()

            processed_df, fit_imputer, fit_scaler, included_cols, numeric_cols = process_features_df(
                df,
                id_fields=id_fields,
                winsor=options['winsor'],
                scaler=options['scaler'],
                return_transforms=True,
                subset_cols=options['subset_cols'][feature_type],
                secondary_deduplication=options['secondary_deduplication'],
                included_fids=included_fids,
                percentage_threshold=options['percentage_threshold'][feature_type]
            )

            processed_df.to_csv(
                join(
                    output_dir,
                    '{}.csv'.format(file_name)),
                index=False)

            if options['secondary_deduplication'] == 'one_per_user' and feature_file['type'] == 'dataset':
                processed_df[['fid']].to_csv(
                    join(output_dir, 'included_fids.csv'), index=False)

            if preserve_fids and number_of_preserved_fids:
                processed_df_preserved_fids = process_features_df(
                    df_preserved_fids,
                    id_fields=id_fields,
                    winsor=None,
                    drop_null=False,
                    fit_imputer=fit_imputer,
                    fit_scaler=fit_scaler,
                    included_cols=included_cols,
                    numeric_cols=numeric_cols,
                    return_transforms=False,
                    secondary_deduplication=None
                )
                processed_df_preserved_fids.to_csv(
                    join(
                        output_dir,
                        '{}_ground_truth.csv'.format(file_name)),
                    index=False)

            pickle.dump({
                'imputer': fit_imputer,
                'scaler': fit_scaler,
                'included_cols': included_cols,
                'numeric_cols': numeric_cols
            }, open(join(output_dir, '{}_transforms.pkl'.format(file_name)), 'wb'))
        for full_file_name in [x for x in os.listdir(
                d) if os.path.isfile(join(d, x)) and not ('features' in x)]:
            copy2(join(d, full_file_name), join(output_dir, full_file_name))


if __name__ == '__main__':
    main()
