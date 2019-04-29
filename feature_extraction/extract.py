#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Feature and Outcome Extraction
'''

import traceback
import pandas as pd

import os
import json
from time import time, strftime

import multiprocessing
from multiprocessing import Pool
from collections import OrderedDict, Counter
from pprint import pprint

from features.aggregate_pairwise_field_features import extract_aggregate_pairwise_field_features
from features.aggregate_single_field_features import extract_aggregate_single_field_features
from features.pairwise_field_features import extract_pairwise_field_features
from features.single_field_features import extract_single_field_features
from outcomes.chart_outcomes import extract_chart_outcomes
from outcomes.field_encoding_outcomes import extract_field_outcomes
from features.transform import supplement_features
from general_helpers import load_raw_data, clean_chunk, persist_features


MAX_FIELDS = 25
total_charts = 0
charts_without_data = 0
chart_loading_errors = 0
feature_extraction_errors = 0
charts_exceeding_max_fields = 0
CHUNK_SIZE = 1000


compute_features_config = {
    'single_field': True,
    'aggregate_single_field': True,

    'pairwise_field': True,
    'aggregate_pairwise_field': True,

    'field_level_features': True,
    'chart_outcomes': True,
    'field_outcomes': True,
    'supplement': False,
}


def extract_chunk_features(args):
    chunk = args['chunk']
    batch_num = args['batch_num']
    chunk_num = args['chunk_num']  # chunk=None, batch_num=0, chunk_num=0):

    df = clean_chunk(chunk)

    num_all_one_type = 0
    global feature_extraction_errors
    global charts_exceeding_max_fields

    chunk_features_by_num_fields = [[] for i in range(MAX_FIELDS)]
    chunk_outcomes_by_num_fields = [[] for i in range(MAX_FIELDS)]

    chunk_features = []
    chunk_outcomes = []
    chunk_field_level_features = []
    chunk_field_level_outcomes = []

    feature_names_by_type = {}

    only_extract_outcomes = False
    if only_extract_outcomes:
        for k in compute_features_config.keys():
            compute_features_config[k] = False
        compute_features_config['outcomes'] = True

    start_time = time()

    # Per dataframe
    for chart_num, chart_obj in df.iterrows():
        fid = chart_obj.fid
        table_data = chart_obj.table_data

        absolute_chart_num = ((chunk_num - 1) * CHUNK_SIZE) + chart_num
        if absolute_chart_num % 100 == 0:
            print('[Batch %s / Chunk %s][%s] %.1f: %s %s' %
                  (batch_num, chunk_num, absolute_chart_num, time() -
                   start_time, fid, 'https://plot.ly/~{0}/{1}'.format(*fid.split(':'))))

        fields = table_data[list(table_data.keys())[0]]['cols']
        sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
        num_fields = len(sorted_fields)
        if num_fields > MAX_FIELDS:
            charts_exceeding_max_fields += 1
            continue

        try:
            extraction_results = extract_features_from_fields(
                sorted_fields, compute_features_config, chart_obj=chart_obj, fid=fid)

            chunk_features_by_num_fields[num_fields - 1].append(
                extraction_results['df_feature_tuples_if_exists'])
            chunk_outcomes_by_num_fields[num_fields -
                                         1].append(extraction_results['df_outcomes_tuples'])
            chunk_features.append(extraction_results['df_feature_tuples'])
            chunk_outcomes.append(extraction_results['df_outcomes_tuples'])
            chunk_field_level_features.extend(
                extraction_results['df_field_level_features'])
            chunk_field_level_outcomes.extend(
                extraction_results['df_field_level_outcomes'])
            feature_names_by_type = extraction_results['feature_names_by_type']
        except Exception as e:
            print('Uncaught exception: {}'.format(e))
            traceback.print_tb(e.__traceback__)
            continue

    r = {
        'dataset_features_by_num_fields': [pd.DataFrame(x) for x in chunk_features_by_num_fields],
        'dataset_features': pd.DataFrame(chunk_features),
        'chart_outcomes_by_num_fields': [pd.DataFrame(x) for x in chunk_outcomes_by_num_fields],
        'chart_outcomes': pd.DataFrame(chunk_outcomes),
        'field_level_features': pd.DataFrame(chunk_field_level_features),
        'field_level_outcomes': pd.DataFrame(chunk_field_level_outcomes),
        'feature_names_by_type': feature_names_by_type,
    }

    return r


def extract_features_from_fields(
        fields, compute_features_config, chart_obj={}, fid=None):
    results = {}

    feature_names_by_type = {
        'basic': ['fid'],
        'single_field': [],
        'aggregate_single_field': [],
        'pairwise_field': [],
        'aggregate_pairwise_field': [],
        'chart_outcomes': [],
        'field_outcomes': []
    }

    df_feature_tuples_if_exists = OrderedDict({'fid': fid})
    df_feature_tuples = OrderedDict({'fid': fid})
    df_outcomes_tuples = OrderedDict()

    if compute_features_config['single_field'] or compute_features_config['field_level_features']:
        single_field_features, parsed_fields = extract_single_field_features(
            fields, fid, MAX_FIELDS=MAX_FIELDS)

        for i, field_features in enumerate(single_field_features):
            field_num = i + 1

            for field_feature_name, field_feature_value in field_features.items():
                if field_feature_name not in ['fid', 'field_id']:
                    field_feature_name_with_num = '{}_{}'.format(
                        field_feature_name, field_num)
                    if field_features['exists']:
                        df_feature_tuples_if_exists[field_feature_name_with_num] = field_feature_value

        if compute_features_config['field_level_features']:
            df_field_level_features = []
            for i, f in enumerate(single_field_features):
                if f['exists']:
                    if compute_features_config['supplement']:
                        f = supplement_features(f)
                    df_field_level_features.append(f)
                    feature_names_by_type['single_field'] = list(f.keys())
            results['df_field_level_features'] = df_field_level_features
        results['single_field_features'] = single_field_features

    if compute_features_config['aggregate_single_field']:
        aggregate_single_field_features = extract_aggregate_single_field_features(
            single_field_features
        )

        if compute_features_config['supplement']:
            aggregate_single_field_features = supplement_features(
                aggregate_single_field_features)

        for k, v in aggregate_single_field_features.items():
            df_feature_tuples[k] = v
            df_feature_tuples_if_exists[k] = v
            feature_names_by_type['aggregate_single_field'].append(k)

        results['aggregate_single_field_features'] = aggregate_single_field_features

    if compute_features_config['pairwise_field'] or compute_features_config['aggregate_pairwise_field']:
        pairwise_field_features = extract_pairwise_field_features(
            parsed_fields,
            single_field_features,
            fid,
            MAX_FIELDS=MAX_FIELDS
        )

        results['pairwise_field_features'] = pairwise_field_features

    if compute_features_config['aggregate_pairwise_field']:
        aggregate_pairwise_field_features = extract_aggregate_pairwise_field_features(
            pairwise_field_features)

        if compute_features_config['supplement']:
            aggregate_pairwise_field_features = supplement_features(
                aggregate_pairwise_field_features)

        for k, v in aggregate_pairwise_field_features.items():
            df_feature_tuples[k] = v
            df_feature_tuples_if_exists[k] = v
            feature_names_by_type['aggregate_pairwise_field'].append(k)

        results['aggregate_pairwise_field_features'] = aggregate_pairwise_field_features

    if compute_features_config['chart_outcomes']:
        outcomes = extract_chart_outcomes(chart_obj)
        for k, v in outcomes.items():
            df_outcomes_tuples[k] = v
            feature_names_by_type['chart_outcomes'].append(k)

    if compute_features_config['field_outcomes']:
        field_level_outcomes = extract_field_outcomes(chart_obj)
        feature_names_by_type['field_outcomes'] = list(
            list(field_level_outcomes)[0].keys())
        results['df_field_level_outcomes'] = field_level_outcomes

    results['df_feature_tuples'] = df_feature_tuples
    results['df_feature_tuples_if_exists'] = df_feature_tuples_if_exists
    results['df_outcomes_tuples'] = df_outcomes_tuples
    results['feature_names_by_type'] = feature_names_by_type

    return results


def process_batch(chunk_batch, batch_num, batch_size, pool, chunk_size=1000):
    batch_start_time = time()
    print('\n==========================')
    print('Running batch: %s' % batch_num)
    global feature_extraction_errors
    global charts_exceeding_max_fields

    feature_extraction_errors = 0
    charts_exceeding_max_fields = 0
    batch_results = pool.map_async(
        extract_chunk_features,
        chunk_batch
    ).get(9999999)

    total_num_rows = batch_size * chunk_size
    batch_time = time() - batch_start_time
    time_per_row = batch_time / (total_num_rows)

    print('Batch time ({}): {:.2f}s'.format(batch_num, batch_time))
    print('Time per 100K rows: {:.0f}s'.format(time_per_row * 100000))
    return batch_results


def write_batch_results(batch_results, features_dir_name, write_header=False):
    batch_dataset_features_dfs_by_num_fields = [[] for i in range(MAX_FIELDS)]
    batch_dataset_features_dfs = []
    batch_chart_outcomes_dfs_by_num_fields = [[] for i in range(MAX_FIELDS)]
    batch_chart_outcomes_dfs = []
    batch_field_level_features_dfs = []
    batch_field_level_outcomes_dfs = []
    feature_names_by_type = {}

    for r in batch_results:
        if not r['dataset_features'].empty:
            batch_dataset_features_dfs.append(r['dataset_features'])
        if not r['chart_outcomes'].empty:
            batch_chart_outcomes_dfs.append(r['chart_outcomes'])
        if not r['field_level_features'].empty:
            batch_field_level_features_dfs.append(r['field_level_features'])
        if not r['field_level_outcomes'].empty:
            batch_field_level_outcomes_dfs.append(r['field_level_outcomes'])

        feature_names_by_type = r['feature_names_by_type']

        for i, dataset_features in enumerate(
                r['dataset_features_by_num_fields']):
            if not dataset_features.empty:
                batch_dataset_features_dfs_by_num_fields[i].append(
                    dataset_features)
        for i, chart_outcomes in enumerate(r['chart_outcomes_by_num_fields']):
            if not chart_outcomes.empty:
                batch_chart_outcomes_dfs_by_num_fields[i].append(
                    chart_outcomes)

    concatenated_results = {
        'features_df_by_num_fields': [pd.concat(features_dfs, ignore_index=True) for features_dfs in batch_dataset_features_dfs_by_num_fields if features_dfs],
        'outcomes_df_by_num_fields': [pd.concat(chart_outcomes_dfs, ignore_index=True) for chart_outcomes_dfs in batch_chart_outcomes_dfs_by_num_fields if chart_outcomes_dfs],
        'features_df': pd.concat(batch_dataset_features_dfs, ignore_index=True) if batch_dataset_features_dfs else pd.DataFrame(),
        'outcomes_df': pd.concat(batch_chart_outcomes_dfs, ignore_index=True) if batch_chart_outcomes_dfs else pd.DataFrame(),
        'field_level_features_df': pd.concat(batch_field_level_features_dfs, ignore_index=True) if batch_field_level_features_dfs else pd.DataFrame(),
        'field_level_outcomes_df': pd.concat(batch_field_level_outcomes_dfs, ignore_index=True) if batch_field_level_outcomes_dfs else pd.DataFrame()
    }

    persist_features(
        concatenated_results,
        MAX_FIELDS,
        features_dir_name,
        feature_names_by_type,
        write_header=write_header)


if __name__ == '__main__':
    parallelize = True
    raw_df_chunks = load_raw_data(chunk_size=CHUNK_SIZE)

    # Create features directory
    base_dir = '..'
    features_dir_name = os.path.join(base_dir, 'features', 'raw')
    os.mkdir(features_dir_name)
    os.mkdir(os.path.join(features_dir_name, 'by_field'))

    first_batch = True
    start_time = time()
    if parallelize:
        batch_num = 1
        chunk_batch = []
        n_jobs = multiprocessing.cpu_count()
        batch_size = n_jobs
        print('Number of jobs: %s' % n_jobs)

        for i, chunk in enumerate(raw_df_chunks):
            chunk_num = i + 1
            chunk_batch.append({
                'chunk': chunk,
                'batch_num': batch_num,
                'chunk_num': chunk_num
            })

            pool = Pool(n_jobs)
            if (chunk_num == (batch_size * batch_num)):
                batch_results = process_batch(
                    chunk_batch, batch_num, batch_size, pool, chunk_size=CHUNK_SIZE)
                write_batch_results(
                    batch_results,
                    features_dir_name,
                    write_header=first_batch)
                batch_num += 1
                chunk_batch = []
                first_batch = False
            pool.close()

        # Process left overs
        pool = Pool(n_jobs)
        remaining_batch_results = process_batch(
            chunk_batch, batch_num, batch_size, pool, chunk_size=CHUNK_SIZE)
        write_batch_results(
            remaining_batch_results,
            features_dir_name,
            write_header=first_batch)
        pool.close()

    else:
        batch_results = []
        for i, chunk in enumerate(raw_df_chunks):
            chunk_num = i + 1
            r = extract_chunk_features({
                'chunk': chunk,
                'chunk_num': chunk_num,
                'batch_num': 'NA'
            })
            batch_results.append(r)
            if (i % 10 == 0):
                write_batch_results(
                    batch_results,
                    features_dir_name,
                    write_header=first_batch)
                batch_results = []
                first_batch = False

    print('Total time: {:.2f}s'.format(time() - start_time))
