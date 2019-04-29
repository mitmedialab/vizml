# !/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Remove charts from a user that are likely created from the same dataset. Uses simplified version of feature extraction

Two-stage process:
1) Generate list of fids to keep based on statistical criteria
2) Output deduplicated chart data

Input: raw plot data AND list of FIDs to preserve (e.g. because of user evaluation)
Output: de-duplicated raw plot data
'''

from os.path import join
import sys
sys.path.insert(0, '..')
from time import time
import pandas as pd
import numpy as np
import csv
import pickle
import json
from experiment_data.ground_truth_fids_99 import ground_truth_fids_99
from feature_extraction.general_helpers import load_raw_data, clean_chunk
from feature_extraction.features.helpers import *
from feature_extraction.features.single_field_features import extract_single_field_features
from feature_extraction.features.type_detection import detect_field_type, data_type_to_general_type, data_types, general_types

fids_to_preserve = [x['fid'] for x in ground_truth_fids_99]
input_file_name = '../data/plot_data_with_all_fields_and_header.tsv'
output_file_name = '../data/{}_deduplicated_one_per_user.tsv'.format(
    input_file_name.rsplit('.', 1)[0])
headers = ['fid', 'chart_data', 'layout', 'table_data']


global unique_fids
global unique_fids_file
unique_fids_file = open('../data/unique_fids.pkl', 'wb')
output_file = csv.writer(open(output_file_name, 'w'), delimiter='\t')
output_file.writerow(headers)

skipped = 0
num_duplicates = 0
CHUNK_SIZE = 1000
ROW_LIMIT = 100

# List of { features: fid }
existing_features = {}
preserved_fids = []
unique_fids = []


def extract_features(chunk, chunk_num):
    global skipped
    global num_duplicates
    df = clean_chunk(chunk)
    start_time = time()

    for chart_num, chart_obj in df.iterrows():
        fid = chart_obj.fid
        table_data = chart_obj.table_data

        absolute_chart_num = ((chunk_num - 1) * CHUNK_SIZE) + chart_num
        if absolute_chart_num % 100 == 0:
            print('[Chunk %s][%s] %.1f: %s %s' % (chunk_num, absolute_chart_num, time(
            ) - start_time, fid, 'https://plot.ly/~{0}/{1}'.format(*fid.split(':'))))

        fields = table_data[list(table_data.keys())[0]]['cols']
        fields = sorted(fields.items(), key=lambda x: x[1]['order'])
        num_fields = len(fields)

        # if num_fields > 25:
        #     skipped += 1
        #     continue
        dataset_features = [num_fields]

        for i, (field_name, d) in enumerate(fields):
            # try:
            field_id = d['uid']
            field_order = d['order']
            field_values = d['data']

            field_length = len(field_values)
            field_type, field_scores = detect_field_type(field_values)
            field_general_type = data_type_to_general_type[field_type]

            try:
                v = parse(field_values, field_type, field_general_type)
                # v = np.ma.array(v).compressed()[:ROW_LIMIT]
                v = np.ma.array(v).compressed()

                characteristic = None

                if len(v):
                    if field_general_type in ['c']:
                        characteristic = pd.Series(v).value_counts().idxmax()

                    if field_general_type in ['t']:
                        characteristic = np.max(v)

                    if field_general_type in ['q']:
                        characteristic = np.mean(v)
            except Exception as e:
                print('Error parsing {}: {}'.format(field_name, e))
                continue
            # = np.append(dataset_features, [field_length, field_general_type, characteristic])
            dataset_features.extend(
                [field_length, field_general_type, characteristic])

        stringified_dataset_features = ''.join(
            [str(s) for s in dataset_features])
        if stringified_dataset_features in existing_features.keys():
            num_duplicates += 1
            old_fid = existing_features[stringified_dataset_features]
            new_fid = fid

            # If we have to preserve the new FID but an identical dataset
            # exists, then replace it
            if new_fid in fids_to_preserve:
                preserved_fids.append(new_fid)
                existing_features[stringified_dataset_features] = new_fid
                continue
        else:
            if fid in fids_to_preserve:
                preserved_fids.append(new_fid)
            existing_features[stringified_dataset_features] = fid

    print('Num skipped:', skipped)
    print('Num preserved FIDs:', len(preserved_fids))
    print('Unique FIDs:', len(existing_features.values()))
    print(
        'Duplicates: {} ({:.3f})'.format(
            num_duplicates,
         (num_duplicates / absolute_chart_num)))


def write_uniques(chunk, chunk_num):
    chunk_rows = []
    start_time = time()

    absolute_chart_num = ((chunk_num - 1) * CHUNK_SIZE)
    if absolute_chart_num % 100 == 0:
        print(
            '[Chunk %s][%s] %.1f' %
            (chunk_num,
             absolute_chart_num,
             time() -
                start_time))

    for i, x in chunk.iterrows():
        if x.fid in unique_fids:
            chunk_rows.append([
                x.fid,
                x.chart_data,
                x.layout,
                x.table_data
            ])
    output_file.writerows(chunk_rows)


if __name__ == '__main__':
    input_file_name = '../data/plot_data.tsv'
    raw_df_chunks = pd.read_table(
        input_file_name,
        error_bad_lines=False,
        chunksize=1000,
        encoding='utf-8'
    )

    for i, chunk in enumerate(raw_df_chunks):
        r = extract_features(chunk, i + 1)
    unique_fids = set(existing_features.values())
    pickle.dump(unique_fids, unique_fids_file)

    raw_df_chunks = pd.read_table(
        input_file_name,
        error_bad_lines=False,
        chunksize=1000,
        encoding='utf-8'
    )
    for i, chunk in enumerate(raw_df_chunks):
        r = write_uniques(chunk, i + 1)
