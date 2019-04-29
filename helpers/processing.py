#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, RobustScaler


def get_feature_set_names_by_type(
        feature_names_by_type, task_type='dataset', feature_set='names'):
    feature_sets = {}
    if task_type == 'dataset':
        feature_names = feature_names_by_type['aggregate_single_field'] + \
            feature_names_by_type['aggregate_pairwise_field']

        dimensions_feature_names = ['exists-agg', 'length-agg']
        type_feature_names = ['data_type', 'general_type']
        value_feature_names = [
            'min-agg',
            'max-agg',
            'mean-agg',
            'median-agg',
            'range-agg',
            'var-agg',
            'std-agg',
            'range_overlap',
            'is_normal',
            'q25-agg',
            'q75-agg',
            'kurtosis-agg',
            '_none',
            'unique',
            'mode',
            'anova',
            'nested',
            'chi_sq',
            'ks_',
            'correlation',
            'shared_elements',
            'identical',
            'sequence',
            '_space-agg',
            'sorted',
            'entropy-agg-',
            'gini-agg-',
            'abs_dev-agg-',
            'normality',
            'monotonic',
            'outliers',
            'moment',
            'quant_coeff_disp',
            'skewness',
            'value_length']
        name_feature_names = [
            'in_name',
            'edit_distance-',
            'uppercase',
            'shared_words',
            'name_length']

        unique_feature_sets = {
            'basic': [],  # feature_names_by_type['basic'],
            'dimensions': [x for x in feature_names if any(x.startswith(e) for e in dimensions_feature_names)],
            'types': [x for x in feature_names if any(e in x for e in type_feature_names)],
            'values': [x for x in feature_names if any(e in x for e in value_feature_names)],
            'names': [x for x in feature_names if any(e in x for e in name_feature_names)]
        }

        feature_sets = {
            'dimensions': unique_feature_sets['basic'] + unique_feature_sets['dimensions'],
            'types': unique_feature_sets['basic'] + unique_feature_sets['dimensions'] + unique_feature_sets['types'],
            'values': unique_feature_sets['basic'] + unique_feature_sets['dimensions'] + unique_feature_sets['types'] + unique_feature_sets['values'],
            'names': unique_feature_sets['basic'] + unique_feature_sets['dimensions'] + unique_feature_sets['types'] + unique_feature_sets['values'] + unique_feature_sets['names']
        }
    if task_type == 'field':
        feature_names = feature_names_by_type['single_field']
        dimensions_feature_names = ['length']
        type_feature_names = ['data_type', 'general_type']
        value_feature_names = [
            'has_none',
            'percentage_none',
            'num_none',
            'num_unique_elements',
            'unique_percent',
            'is_unique',
            'list_entropy',
            'mean_value_length',
            'median_value_length',
            'min_value_length',
            'max_value_length',
            'std_value_length',
            'percentage_of_mode',
            'mean',
            'normalized_mean',
            'median',
            'normalized_median',
            'var',
            'std',
            'coeff_var',
            'min',
            'max',
            'range',
            'normalized_range',
            'entropy',
            'gini',
            'q25',
            'q75',
            'med_abs_dev',
            'avg_abs_dev',
            'quant_coeff_disp',
            'skewness',
            'kurtosis',
            'moment_5',
            'moment_6',
            'moment_7',
            'moment_8',
            'moment_9',
            'moment_10',
            'percent_outliers_15iqr',
            'percent_outliers_3iqr',
            'percent_outliers_1_99',
            'percent_outliers_3std',
            'has_outliers_15iqr',
            'has_outliers_3iqr',
            'has_outliers_1_99',
            'has_outliers_3std',
            'normality_statistic',
            'normality_p',
            'is_normal_5',
            'is_normal_1',
            'is_sorted',
            'is_monotonic',
            'sortedness',
            'lin_space_sequence_coeff',
            'log_space_sequence_coeff',
            'is_lin_space',
            'is_log_space']
        name_feature_names = [
            'in_name',
            'edit_distance-',
            'uppercase',
            'shared_words',
            'name_length']

        unique_feature_sets = {
            'basic': [],  # feature_names_by_type['basic'],
            'dimensions': dimensions_feature_names,
            'types': [x for x in feature_names if any(e in x for e in type_feature_names)],
            'values': value_feature_names,
            'names': [x for x in feature_names if any(e in x for e in name_feature_names)]
        }

        feature_sets = {
            'dimensions': unique_feature_sets['basic'] + unique_feature_sets['dimensions'],
            'types': unique_feature_sets['basic'] + unique_feature_sets['dimensions'] + unique_feature_sets['types'],
            'values': unique_feature_sets['basic'] + unique_feature_sets['dimensions'] + unique_feature_sets['types'] + unique_feature_sets['values'],
            'names': unique_feature_sets['basic'] + unique_feature_sets['dimensions'] + unique_feature_sets['types'] + unique_feature_sets['values'] + unique_feature_sets['names']
        }
    return feature_sets[feature_set]
