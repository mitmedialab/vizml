#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
1. Add header to full data
2. Remove final charts without data and layout data
'''


import traceback
import pandas as pd
import csv
import json
from time import time
from ast import literal_eval

charts_without_data = 0
chart_loading_errors = 0

df_final_rows = []
errors = 0
empty_fields = 0

headers = ['fid', 'chart_data', 'layout', 'table_data']
data_file_name = '../data/plot_data.tsv'
chunks = pd.read_csv(
    data_file_name,
    sep='\t',
    names=headers,
    error_bad_lines=False,
    chunksize=1000,
)

all_fields_output_file_name = data_file_name[:-4] + '_with_all_fields_and_header.tsv'
incomplete_output_file_name = data_file_name[:-4] + '_without_all_fields_and_header.tsv'
output_delimiter = '\t'

all_fields_output_file = csv.writer(
    open(
        all_fields_output_file_name,
        'w'),
    delimiter=output_delimiter)
all_fields_output_file.writerow(headers)

incomplete_output_file = csv.writer(
    open(
        incomplete_output_file_name,
        'w'),
    delimiter=output_delimiter)
incomplete_output_file.writerow(headers)

start_time = time()
total_num = 0
final_num_charts = 0
for chunk_num, chunk in enumerate(chunks):
    chunk_start_time = time()

    all_fields_chunk_rows = []
    incomplete_chunk_rows = []

    chunk.fillna('{}', inplace=True)
    for i, x in chunk.iterrows():
        total_num + 1
        try:
            chart_data = literal_eval(x.chart_data)
            layout = literal_eval(x.layout)
            table_data = literal_eval(x.table_data)

            # Filter empty fields
            if not (bool(chart_data) and bool(table_data)):
                empty_fields += 1
                charts_without_data += 1
                chart_loading_errors += 1

                incomplete_chunk_rows.append([
                    x['fid'],
                    json.dumps(chart_data),
                    json.dumps(layout),
                    json.dumps(table_data)
                ])
                continue

            all_fields_chunk_rows.append([
                x['fid'],
                json.dumps(chart_data),
                json.dumps(layout),
                json.dumps(table_data)
            ])
            final_num_charts += 1

        except Exception as e:
            errors += 1
            continue

    all_fields_output_file.writerows(all_fields_chunk_rows)
    incomplete_output_file.writerows(incomplete_chunk_rows)
    total_num += chunk.shape[0]
    print(
        'Chunk {}: {:.2f}s ({:.2f}s)'.format(
            chunk_num + 1,
            time() - start_time,
            time() - chunk_start_time))
    print('\tFinal number of charts: %s (%.2f)' %
          (final_num_charts, float(final_num_charts) / total_num))
    print('\tEmpty fields: %s' % empty_fields)
    print('\tErrors: %s' % errors)
