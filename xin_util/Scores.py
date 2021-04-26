import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def single_label_included_score(y_pred, y_gold):
    """
    this function will computer the score by including
    Example 1:
    y_pred=[1,2,3]
    y_gold=[{4,5},{2,4},{1,3,7}]
    we can see that for position 0, label "1" is not in {4,5}, but for position 1 and 2
    labels "2", "3" are in {2,4} and {1,3,7} respectively, in this case, the overall
    score is 2/3

    Example 2:
    it can also compute the score same way but for each label
    y_pred=[1,2,3,2,3]
    y_gold=[{4,5},{2,4},{1,3,7},{2,6},{4,5}]
    in this case we see that for label "1", it appears once in y_pred, and not in y_gold
    thus accuracy for "1" is 0.
    Similarity, label "2" appears twice in y_pred, and each time it is in y_gold,
    thus accuracy for "2" is 1
    Same way, for "3" is 1/2

    :param y_pred: a list of predicted labels, must be same length as y_gold
    :param y_gold: a list of sets of labels, must be same length as y_pred
    :return:
    total_score: float of the total score calculated by example 1
    label_wise_accuracy: a dictionary,where keys are labels, values are float score of the label
                        calculated by example 2
    """
    assert len(y_pred) == len(y_gold), 'y_pred and y_gold need to have same length'
    count = 0
    label_wise_score = nltk.defaultdict(lambda: nltk.defaultdict(int))
    for index, pred in enumerate(y_pred):
        gold = set(y_gold[index])
        if pred in gold:
            count += 1
            label_wise_score[pred]['total'] += 1
            label_wise_score[pred]['correct'] += 1
        else:
            label_wise_score[pred]['total'] += 1
    label_wise_accuracy = dict()
    for label in label_wise_score.keys():
        try:
            rate = label_wise_score[label]['correct'] / label_wise_score[label]['total']
        except:
            rate = 0
        label_wise_accuracy[label] = rate
    total_score = count / len(y_gold)
    return total_score, label_wise_accuracy


def multiple_label_included_score(y_pred, y_gold, plot_dist=False, figsize=(10, 6)):
    """
    this function will computer the score by including
    Example 1:
    y_pred=[{1,3},{2,8},{3,7}]
    y_gold=[{4,5},{2,4},{1,3,7}]
    we can see that for position 0, label "1" or "3" is not in {4,5}, but for position 1 and 2
    labels "2", "3" and "7" are in {2,4} and {1,3,7} respectively, in this case, the overall
    score is 2/3

    Example 2:
    it can also compute the score same way but for each label
    y_pred=[{1,6},{2,4},{3,9},{2,8},{3,4}]
    y_gold=[{4,5},{2,4},{1,3,7},{2,6},{4,5,10}]
    in this case we see that for label "1", it appears once in y_pred, and not in y_gold
    thus accuracy for "1" is 0.
    Similarity, label "2" appears twice in y_pred, and each time it is in y_gold,
    thus accuracy for "2" is 1
    Same way, for "3" is 1/2, for "4" is 1, for "6" is 0, for "8" and "9" are 0s


    :param y_pred: a list of sets of predicted labels, must be same length as y_gold
    :param y_gold: a list of sets of labels, must be same length as y_pred
    :param plot_dist: boolean, plot for distribution of which prediction of n predictions for each data point
                      gives the correct prediction only apply when all number of predictions are the same,
                      and predictions are in a list not set
    :return:
    total_score: float of the total score calculated by example 1
    label_wise_accuracy: a dictionary,where keys are labels, values are float score of the label
                        calculated by example 2
    """
    assert len(y_pred) == len(y_gold), 'y_pred and y_gold need to have same length'
    count = 0
    num_of_pred = len(y_pred[0])
    plot_dict = dict((i + 1, 0) for i in range(num_of_pred))
    label_wise_score = nltk.defaultdict(lambda: nltk.defaultdict(int))
    for index, pred in enumerate(y_pred):
        pred1 = set(pred)
        gold = set(y_gold[index])
        Intersection = pred1.intersection(gold)
        if Intersection:
            count += 1
        if plot_dist:
            for prediction in Intersection:
                i = pred.index(prediction) + 1
                plot_dict[i] += 1
        for label in pred:
            if label in Intersection:
                label_wise_score[label]['total'] += 1
                label_wise_score[label]['correct'] += 1
            else:
                label_wise_score[label]['total'] += 1
    label_wise_accuracy = dict()
    for label in label_wise_score.keys():
        try:
            rate = label_wise_score[label]['correct'] / label_wise_score[label]['total']
        except:
            rate = 0
        label_wise_accuracy[label] = rate
    total_score = count / len(y_gold)
    if plot_dist:
        data = []
        for position, how_many in plot_dict.items():
            data.append((position, how_many))
        data = sorted(data, key=lambda x: x[0])
        df = pd.DataFrame(data, columns=['position', 'count'])
        sns.set(style="darkgrid")
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            x="position", y="count", data=df, color='lightsalmon', orient='v', saturation=.1
        )
        for p in ax.patches:
            ax.annotate('{0}'.format(int(p.get_height())), (p.get_x() + .35, p.get_height()))
        for item in ax.get_xticklabels():
            item.set_rotation(0)
        plt.title(
            'Count of ith position made the correct prediction. {0} in Total'.format(num_of_pred)
        )
        plt.show()
    return total_score, label_wise_accuracy


def single_label_normal_score(y_pred, y_gold):
    """
    this function will computer the score by simple compare exact or not
    Example 1:
    y_pred=[1,2,3]
    y_gold=[2,2,3]
    score is 2/3

    Example 2:
    it can also compute the score same way but for each label
    y_pred=[1,2,3,2,3]
    y_gold=[2,2,3,1,3]
    in this case we see that for label "1", it appears once in y_pred, and not in y_gold
    thus accuracy for "1" is 0.
    Similarity, label "2" appears twice in y_pred, and once it is in y_gold,
    thus accuracy for "2" is 1/2
    Same way, for "3" is 1

    :param y_pred: a list of labels, must be same length as y_gold
    :param y_gold: a list of labels, must be same length as y_pred
    :return:
    total_score: float of the total score calculated by example 1
    label_wise_accuracy: a dictionary,where keys are labels, values are float score of the label
                        calculated by example 2
    """
    assert len(y_pred) == len(y_gold), 'y_pred and y_gold need to have same length'
    count = 0
    label_wise_score = nltk.defaultdict(lambda: nltk.defaultdict(int))
    for index, pred in enumerate(y_pred):
        gold = y_gold[index]
        if pred == gold:
            count += 1
            label_wise_score[pred]['total'] += 1
            label_wise_score[pred]['correct'] += 1
        else:
            label_wise_score[pred]['total'] += 1
    label_wise_accuracy = dict()
    for label in label_wise_score.keys():
        try:
            rate = label_wise_score[label]['correct'] / label_wise_score[label]['total']
        except:
            rate = 0
        label_wise_accuracy[label] = rate
    total_score = count / len(y_gold)
    return total_score, label_wise_accuracy


def single_label_f_score(y_pred, y_gold):
    """
    this function will computer the F-score
    :param y_pred: a list of labels, must be same length as y_gold
    :param y_gold: a list of labels, must be same length as y_pred
    :return:
    total_score: float of the total score calculated by average all f-scores for all labels
    label_wise_accuracy: a dictionary,where keys are labels, values are float f-score
    """
    assert len(y_pred) == len(y_gold), 'y_pred and y_gold need to have same length'
    Y_P = list(y_pred).copy()
    Y_G = list(y_gold).copy()
    Y_P.extend(Y_G)
    all_labels = set(Y_P)
    label_pairs = []
    for index, pred in enumerate(y_pred):
        label_pairs.append([pred, y_gold[index]])
    labels_pairs_dict = nltk.defaultdict(list)
    for label_ in all_labels:
        for pred, gold in label_pairs:
            if pred == label_ or gold == label_:
                labels_pairs_dict[label_].append((pred, gold))
    labels_f_score = dict()
    for label in labels_pairs_dict.keys():
        FP = 0
        FN = 0
        TP = 0
        pair_list = labels_pairs_dict[label]
        for (pred, gold) in pair_list:
            if pred == label and gold != label:
                FP += 1
            elif pred != label and gold == label:
                FN += 1
            elif pred == label and gold == label:
                TP += 1
        try:
            precision = TP / (TP + FP)
        except:
            precision = 0
        try:
            recall = TP / (TP + FN)
        except:
            recall = 0
        try:
            F_score = 2 * precision * recall / (precision + recall)
        except:
            F_score = 0
        labels_f_score[label] = F_score
    SUM = 0
    for l, Fscore in labels_f_score.items():
        SUM += Fscore
    total_score = SUM / len(labels_f_score)
    return total_score, labels_f_score


def multiple_label_intersection_score(y_pred, y_gold):
    """
    this function will computer the score by including
    Example 1:
    y_pred=[{1,3},{2,8},{3,7,9}]
    y_gold=[{4,5,6},{2,4,8},{1,3,7}]
    we can see that for position 0, the intersection is none, so intersection size is 0,
    and the size of pred is 2, the size of gold is 3, thus the score is (0/2+0/3)/2=0.
    Similarly for position 2 is (2/2+2/3)/2=5/6, for position 3 is (2/3+2/3)/2=2/3

    Example 2:
    it can also compute the score same way but for each label
    y_pred=[{1,6},{2,4},{3,9},{2,8},{3,4}]
    y_gold=[{4,5},{2,4},{1,3,7},{2,6},{4,5,10}]
    in this case we see that for label "1", it appears once in y_pred, and not in y_gold,
    since or appearing has a penalty, we minute score for that. So score for 1 is -1
    Similarity, label "2" appears twice in y_pred, and each time it is in y_gold,
    thus accuracy for "2" is 1+1/2=1
    Same way, for "3" is 1-1/2=0, for "4" is 1+1/2=1, for "6" is -1-1/2=-1, for "8" -1/2=-0.5
    and "9" -1/2=-0.5

    :param y_pred: a list of sets of predicted labels, must be same length as y_gold
    :param y_gold: a list of sets of labels, must be same length as y_pred
    :return:
    total_score: float of the total score calculated by example 1
    label_wise_accuracy: a dictionary,where keys are labels, values are float score of the label
                        calculated by example 2
    """
    assert len(y_pred) == len(y_gold), 'y_pred and y_gold need to have same length'
    label_wise_score = nltk.defaultdict(lambda: nltk.defaultdict(int))
    All_score = []
    for index, pred in enumerate(y_pred):
        pred = set(pred)
        gold = set(y_gold[index])
        Intersection = pred.intersection(gold)
        forward_score = len(Intersection) / len(pred)
        backward_score = len(Intersection) / len(gold)
        score = (backward_score + forward_score) / 2
        All_score.append(score)
        all = pred.union(gold)
        for label in all:
            if label in Intersection:
                label_wise_score[label]['total'] += 1
                label_wise_score[label]['correct'] += 1
            else:
                label_wise_score[label]['total'] += 1
                label_wise_score[label]['correct'] -= 1
    label_wise_accuracy = dict()
    for label in label_wise_score.keys():
        try:
            rate = label_wise_score[label]['correct'] / label_wise_score[label]['total']
        except:
            rate = 0
        label_wise_accuracy[label] = rate
    total_score = sum(All_score) / len(All_score)
    return total_score, label_wise_accuracy
