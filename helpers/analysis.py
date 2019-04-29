#!/usr/bin/python3
# -*- coding: utf-8 -*-

from string import capwords
import re
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.utils import resample
from IPython.display import display, HTML

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def format_outcomes_df(outcomes_df, outcome_variable_name,
                       outcomes, id_field='fid'):
    print('Subsetting outcomes')

    outcomes_df[outcome_variable_name].fillna(value=False, inplace=True)
    if outcomes:
        outcomes_df = outcomes_df[outcomes_df[outcome_variable_name].isin(
            outcomes)]
    outcomes_df_subset = outcomes_df[[id_field, outcome_variable_name]]
    return outcomes_df_subset


def plot_learning_curve(train_sizes, train_scores,
                        test_scores, title='Learning Curve', ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("learning_curve.svg", format="svg")
    plt.show()
    return plt


def join_features_and_outcomes(features_df, outcomes_df, on='fid'):
    print('Joining feature and outcome DFs')
    final_df = pd.merge(features_df, outcomes_df, on=on, how='inner')
    final_df = final_df.drop(['fid'], axis=1, inplace=False, errors='ignore')
    if on != 'fid':
        final_df = final_df.drop([on], axis=1, inplace=False, errors='ignore')
    return final_df


def resample_X_and_y(X, y, sampling_mode='over', random_state=42):
    if sampling_mode == 'over':
        res = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = res.fit_sample(X, y)
        value_counts = pd.Series(y).value_counts()
        per_class = value_counts.max()
    elif sampling_mode == 'under':
        res = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = res.fit_sample(X, y)
        value_counts = pd.Series(y).value_counts()
        per_class = value_counts.min()
    elif isinstance(sampling_mode, int):
        X_resampled_arrays, y_resampled_arrays = [], []
        for outcome in np.unique(y):
            outcome_mask = (y == outcome)
            X_resampled_outcome, y_resampled_outcome = resample(
                X[outcome_mask],
                y[outcome_mask],
                n_samples=sampling_mode,
                random_state=random_state
            )
            X_resampled_arrays.append(X_resampled_outcome)
            y_resampled_arrays.append(y_resampled_outcome)
        X_resampled, y_resampled = pd.concat(
            X_resampled_arrays), pd.concat(y_resampled_arrays)
        value_counts = pd.Series(y).value_counts()
        per_class = sampling_mode

    return X_resampled, y_resampled, per_class


def test_leakage(X_train, X_test, y_train, y_test):
    if not set(X_train.index).isdisjoint(X_test.index) or not set(
            y_train.index).isdisjoint(y_test.index):
        print('Training and test sets overlapping')
        return False
    return True


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_most_frequent_k(v, k=2):
    unique, counts = np.unique(v, return_counts=True)
    zipped_counts = np.asarray((unique, counts)).T
    sorted_counts = sorted(zipped_counts, key=lambda k: k[1])[(-1 * k):]
    top = [x[0] for x in sorted_counts]
    return top


def get_mode_confidence(x, num_samples=100):
    observed_counts = Counter()

    for iteration in range(num_samples):
        mode_result = mode(np.random.choice(x, len(x), replace=True)).mode[0]
        observed_counts[mode_result] += 1

    mode_percentages = dict((k, 0) for k in observed_counts)
    for k, v in observed_counts.items():
        mode_percentages[k] = v / num_samples

    return mode_percentages


def get_conf_int(x, alpha=0.95):
    mean = np.mean(x)
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(x, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(x, p))

    return {
        'mean': mean,
        'lower': lower,
        'upper': upper,
        'error': upper - mean
    }


def format_feature_name(n):
    name = n.replace('_', ' ')
    agg = ''
    if '-agg-' in name:
        name, agg = name.split('-agg-')
        return capwords(name), agg
    else:
        return name.title()


def format_feature_importance_outcomes(statistic, columns, p=[], returned_fields=[
                                       'statistic', 'statistic_norm', 'feature_name_formatted', 'aggregations'], aggregations=True, limit=25):
    formatted_feature_names, feature_aggregations = [], []
    for x in pd.Series(columns).apply(format_feature_name):
        if aggregations:
            formatted_feature_names.append(x[0])
            feature_aggregations.append(x[1])
        else:
            formatted_feature_names.append(x)

    results_df = pd.DataFrame({
        'feature_name': columns,
        'feature_name_formatted': formatted_feature_names,

        'statistic': statistic,
        'statistic_norm': statistic / np.nanmax(statistic),
    })

    if aggregations:
        results_df['aggregations'] = feature_aggregations

    if p:
        results_df['p'] = p

    results_df.sort_values(['statistic'], ascending=False, inplace=True)
    results_df.reset_index(inplace=True)

    display(
        HTML(results_df[returned_fields].iloc[:limit, :].to_html(index=False)))


def get_bootstrapped_votes(vote_results_without_na,
                           num_votes=None, num_bootstraps=1000):
    print('Bootstrapping {} vote sets'.format(num_bootstraps))
    bootstrapped_votes = []
    for i in range(0, num_bootstraps):
        if (i % 1000 == 0):
            print(i)
        resampled_votes = {}
        for c, votes in vote_results_without_na.items():
            if num_votes is None:
                num_votes = len(votes)
            resampled_votes[c] = np.random.choice(
                votes, num_votes, replace=True)
        bootstrapped_votes.append(resampled_votes)
    return bootstrapped_votes


def format_feature_importances(model, columns, limit=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_importance_table = [
        [columns[i], importances[i]] for i in indices][:limit]
    feature_importance_df = pd.DataFrame(
        feature_importance_table, columns=[
            'feature', 'aggregation', 'importance'])
    display(feature_importance_df)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_rf_feature_importance(model, columns, limit=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_importance_table = [
        [columns[i], importances[i]] for i in indices][:limit]
    feature_importance_df = pd.DataFrame(
        feature_importance_table, columns=[
            'feature', 'importance'])

    plt.figure(figsize=(8, 24))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt