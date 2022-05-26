"""
****************************************
 * @author: Xin Zhang
 * Date: 5/20/21
****************************************
"""
import pandas as pd
import nltk
import numpy as np
from tqdm import tqdm


def euclidean_df_array_faster(data_df, vector_array, split_rows=None):
    """

    :param data_df: data_df where rows are vectors
    :param vector_array: array where shape = (data_df.shape[1],)
    :param split_rows: how many rows per iter, in order to manage memory use
    :return:
    """
    df = data_df.copy()
    df.reset_index(drop=True, inplace=True)
    total_rows = df.shape[0]
    if split_rows is None:
        split_rows = total_rows
    split_indices = list(
        nltk.bigrams([0] + [
            v + index * split_rows
            for index, v in enumerate([split_rows] *
                                      (total_rows // split_rows) + [total_rows % split_rows])
        ])
    )

    all_distance = []
    for start, end in tqdm(split_indices):
        partial_feature = df.iloc[df.index.tolist()[start:end]].to_numpy()
        M = partial_feature - vector_array
        all_distance.extend(list(np.sqrt(np.sum(M**2, axis=1))))
    return all_distance


def euclidean_df_array(data_df, vector_array):
    """

    :param data_df: data_df where rows are vectors
    :param vector_array: array where shape = (data_df.shape[1],)
    :return:
    """
    df = data_df.copy()
    df.reset_index(drop=True, inplace=True)
    all_distance = []
    for index in tqdm(df.index.tolist()):
        vector = df.iloc[index].to_numpy()
        all_distance.append(np.linalg.norm(vector - vector_array))
    return all_distance


def demo(print_result=False):
    feature_size = 300
    data_size = 500000

    data = np.random.rand(data_size, feature_size)
    df = pd.DataFrame(data=data)
    vector = np.random.rand(feature_size)
    split_rows = int(data_size / 5)

    print('Start calculating Euclidean distance between vector and all rows...')
    print('method 1...')
    d1 = euclidean_df_array_faster(df, vector, split_rows)
    print('method 2...')
    d2 = euclidean_df_array(df, vector)
    print('the distance between 2 results: np.linalg.norm(d_method1 - d_method2)')
    print('distance =', np.linalg.norm(np.array(d1) - np.array(d2)))
    if print_result:
        print('distances from method1')
        print(d1)
        print('distances from method2')
        print(d2)
