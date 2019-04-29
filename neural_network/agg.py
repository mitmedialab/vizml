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


# this script tests aggregate features over line, scatter, and bar outcomes
def load_features():
    features_df_file_name = 'features_aggregate_single.csv'
    outcomes_df_file_name = 'chart_outcomes.csv'

    features_df = pd.read_csv(
        os.path.join(
            features_directory,
            features_df_file_name))
    outcomes_df = pd.read_csv(
        os.path.join(
            features_directory,
            outcomes_df_file_name))

    features_df = features_df[:num_datapoints]
    outcome_variable_name = 'all_one_trace_type'
    outcomes = ['line', 'scatter', 'bar']
    outcomes_df_subset = outcomes_df[outcomes_df[outcome_variable_name].isin(
        outcomes)][['fid', outcome_variable_name]]

    final_df = pd.merge(features_df, outcomes_df_subset, on='fid', how='inner')
    final_df = final_df.drop(['fid'], axis=1, inplace=False, errors='ignore')
    final_df.sample(frac=1.0)

    last_index = final_df.columns.get_loc(outcome_variable_name)
    X = final_df.iloc[:, :last_index]
    y = final_df.iloc[:, last_index]
    y = pd.get_dummies(y).values.argmax(1)

    res = RandomOverSampler(random_state=RANDOM_STATE)
    X, y = res.fit_sample(X, y)
    # shuffle X and y in unison, and then return
    return util.unison_shuffle(X, y)


def main():
    # all the parameters for the script can be adjusted here

    # num_epochs: the max number of epochs we will train the NN for
    # hidden_sizes: the number of neurons in each hidden layer, enter it as a list
    # output_dim: the dimension of the output. Since outputs are 'line', 'scatter', 'bar', it's 3
    # weight_decay: how much to decay LR in the NN. This can be set to 0 since we decrease LR already through
    #   the ReduceLROnPlateau() function
    # dropout: the dropout in each layer
    # patience: how many epochs we go through (with a near constant learning rate, this threshold is adjusted using
    #   threshold) before dropping learning rate by a factor of 10
    # model_prefix: all models will be loaded/saved with the prefix of the file in the beginning
    # save_model: save each epoch's model onto models/ folder.
    # print_test: print test accuracies into test.txt
    # test_best: test the test accuracy of the best model we've found (best
    # model determined using val accuracy)

    # note: training is automatically stopped when learning rate < 0.01 *
    # starting learning rate
    parameters = {
        'batch_size': 200,
        'num_epochs': 100,
        'hidden_sizes': [800, 800, 800],
        'learning_rate': 5e-4,
        'output_dim': 3,
        'weight_decay': 0,
        'dropout': 0.00,
        'patience': 20,
        'threshold': 1e-3,
        'model_prefix': 'agg',
        'save_model': False,
        'print_test': True,
        'test_best': False
    }

    # LOAD loads the unfiltered features from the .csv files and converts them into filtered .npy files into ~/saves
    # TRAIN trains using the given parameters and .npy files
    # EVAL evaluates prefix_.model_number (giving you test accuracy)

    assert len(sys.argv) >= 2, 'You must specify a command LOAD, TRAIN, or EVAL'
    assert(parameters['model_prefix']
           ), 'You must specify a prefix for the model name'
    if parameters['test_best']:
        assert parameters['save_model'], 'You must save a model to test the best version!'

    command = sys.argv[1].lower()
    if command == 'load':
        X, y = load_features()
        # here, we split 10% of examples into val, and 10% into test
        util.save_matrices_to_disk(
            X, y, [0.1, 0.1], saves_directory, parameters['model_prefix'], num_datapoints)
        return

    X_train, y_train, X_val, y_val, X_test, y_test = util.load_matrices_from_disk(
        saves_directory, parameters['model_prefix'], num_datapoints)

    if command == 'train':
        train_dataloader, val_dataloader, test_dataloader = train.load_datasets(
            X_train, y_train, X_val, y_val, parameters, X_test=X_test, y_test=y_test)
        train.train(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            parameters)

    elif command == 'eval':
        assert len(sys.argv) >= 3
        model_suffix = sys.argv[2]
        evaluate.evaluate(model_suffix, X_test, y_test, parameters)
    else:
        assert False, 'The command must either be LOAD, TRAIN, or EVAL'


if __name__ == '__main__':
    main()
