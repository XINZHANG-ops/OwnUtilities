import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .TextProcess import text_tokens
from xin_util.Scores import single_label_f_score
from tqdm import tqdm

tqdm.pandas()
import nltk as nltk
from sklearn.model_selection import train_test_split
from nltk.classify import apply_features


def process_sow(text):
    Text = text_tokens(
        text,
        lower_bound_percentage=0,
        higher_bound_percentage=1,
        minimal_word_length=0,
        remove_punctuations=False,
        remove_non_letter_characters=True,
        lemmatize_the_words=False,
        stemmer_the_words=True,
        part_of_speech_filter=False,
        english_text_filter=False,
        stop_words_filter=True,
        other_words_filter=False,
        remove_adjacent_tokens=False,
        tokens_form=False
    )
    return Text


def process_sow_quick(text):
    Text = text_tokens(
        text,
        lower_bound_percentage=0,
        higher_bound_percentage=1,
        minimal_word_length=0,
        remove_punctuations=False,
        remove_non_letter_characters=False,
        lemmatize_the_words=False,
        stemmer_the_words=False,
        part_of_speech_filter=False,
        english_text_filter=False,
        stop_words_filter=False,
        other_words_filter=False,
        remove_adjacent_tokens=False,
        tokens_form=False
    )
    return Text


def Naive_Bayes_classify(
    input_df,
    feature_column,
    label_column,
    process_text=False,
    test_size=.2,
    random_state=1,
    feature_fct=None,
    most_informative_features=10,
    show_info=True,
    return_fscore=True
):
    df = input_df.copy()
    if process_text:
        df[feature_column] = df[feature_column].progress_apply(process_sow)
    else:
        df[feature_column] = df[feature_column].apply(process_sow_quick)
    features = list(df[feature_column])
    labels = list(df[label_column])
    data_set = [(f, labels[index]) for index, f in enumerate(features)]
    if feature_fct:
        get_features = feature_fct
    else:

        def get_features(text):
            features = {}
            tokens = text.split(' ')
            features["sow length"] = len(tokens)
            fd = nltk.FreqDist(tokens)
            most_common = fd.most_common(3)
            top = most_common[0][0]
            sec = most_common[1][0]
            thr = most_common[2][0]
            features["1st word"] = top
            features["2nd word"] = sec
            features["3rd word"] = thr
            return features

    X_train, X_test = train_test_split(data_set, test_size=test_size, random_state=random_state)
    train_set = apply_features(get_features, X_train)
    test_set = apply_features(get_features, X_test)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    y_pred = []
    y_test = []
    for i in range(len(list(test_set))):
        y_pred.append(classifier.classify(test_set[i][0]))
        y_test.append(test_set[i][1])
    f, cf = single_label_f_score(y_gold=y_test, y_pred=y_pred)
    if show_info:
        print('f-score:', f)
        print('label wise f-score', cf)
        print(classifier.show_most_informative_features(most_informative_features))
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        labels = list(set(labels))
        sns.heatmap(
            conf_mat, annot=True, cmap="Blues", fmt='d', xticklabels=labels, yticklabels=labels
        )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title("NaiveBayes CONFUSION MATRIX", size=16)
    if return_fscore:
        return f, cf
