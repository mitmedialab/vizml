'''
Two deduplication strategies:
1) threshold_deduplication based on radius in KD-tree formed by selected features
2) one_per_user_deduplication that only allows one dataset per user
'''

from time import time
import numpy as np
#from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree as KDTree
# https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/


def kd_threshold_deduplication(df, features, percentage_threshold=0.05):
    print('\tDeduplicating with threshold {}'.format(percentage_threshold))

    start_time = time()
    print('Constructing KDTree')
    # kd_tree = KDTree(df[features])
    kd_tree = KDTree(df[features], balanced_tree=False)
    print('Finished constructing KDTree', time() - start_time)

    dropped_points = []
    distances = []

    points_to_drop = set()
    num_points = 0
    df.reset_index(inplace=True)

    distance_threshold = np.sqrt(len(features) * percentage_threshold**2)
    start_time = time()
    print('Querying KDTree')
    for index, x in df[features].iterrows():
        num_points += 1
        if num_points % 100000 == 0:
            print('\t\tNum points: {}'.format(num_points))
        if index in points_to_drop:
            continue

        # points_within_radius = kd_tree.query_radius([x], r=distance_threshold)[0]
        points_within_radius = kd_tree.query_ball_point(
            [x], r=distance_threshold)[0]

        points_within_radius = [
            p for p in points_within_radius if (
                p != index and p not in dropped_points)]
        points_to_drop.update(points_within_radius)

    print('Finished querying KDTree', time() - start_time)

    df.drop(points_to_drop, inplace=True, errors='ignore')
    df.drop(['index'], axis=1, inplace=True, errors='ignore')
    print('percentage_threshold', percentage_threshold)
    print('Dropped {} points'.format(len(points_to_drop)))
    return df


def one_per_user_deduplication(df):
    print('\tDeduplicating by user')
    df['user'], df['id'] = df['fid'].str.split(':').str
    df.drop_duplicates(subset=['user'], inplace=True)
    df.drop(['user', 'id'], axis=1, inplace=True)
    return df
