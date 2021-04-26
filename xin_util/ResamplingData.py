from collections import Counter
import random


def up_resample(x_train, y_train, reach_percent=.5, seed=0):
    """
    function will copy paste labels with less data,up to the reach_percent level of biggest label in set
    :param x_train:
    :param y_train:
    :param reach_percent: float, reach at least what percentage of largest set in data,
    fo labels already equal or higher to the percentage, wont be changed
    :return:
    """
    R = random.Random(seed)
    occurrence_counter = Counter(y_train)
    Label_counts = sorted(occurrence_counter.items(), key=lambda x: x[1], reverse=True)
    largest_occur = Label_counts[0][1]
    min_num = int(largest_occur * reach_percent)
    new_x_train = list(x_train).copy()
    new_y_train = list(y_train).copy()
    for label, count in occurrence_counter.items():
        if count >= min_num:
            continue
        compensate_amount = min_num - count
        label_data = []
        for x, y in zip(x_train, y_train):
            if y == label:
                label_data.append(x)
        compensation = R.choices(label_data, k=compensate_amount)
        new_x_train.extend(compensation)
        new_y_train.extend([label] * compensate_amount)
    return new_x_train, new_y_train
