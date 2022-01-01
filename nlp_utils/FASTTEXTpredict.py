from tqdm import tqdm

tqdm.pandas()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .TextProcess import text_tokens
from xin_util.Scores import single_label_f_score
from sklearn.model_selection import train_test_split
import fasttext


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


def to_txt(data, label, name, mode='w'):
    Data = []
    for index, d in enumerate(data):
        Data.append(['__label__' + str(label[index]), d])
    df = pd.DataFrame(Data, columns=['Token', 'Statement'])
    df.to_csv(r'{}'.format(name), header=None, index=None, sep=' ', mode=mode)


def fasttext_classify(
    input_df,
    feature_column,
    label_column,
    process_text=False,
    test_size=.2,
    random_state=1,
    word_dim=100,
    epoch=10,
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
    labels = [str(l) for l in labels]
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    to_txt(X_train, y_train, f'fasttext_train.txt', mode='w')
    model = fasttext.train_supervised(
        f'fasttext_train.txt',
        minCount=1,
        minCountLabel=1,
        wordNgrams=1,
        lr=.5,
        lrUpdateRate=100,
        dim=word_dim,
        ws=5,
        minn=3,
        maxn=6,
        epoch=epoch
    )
    y_pred = [model.predict(text)[0][0][9:] for text in X_test]
    f, cf = single_label_f_score(y_gold=y_test, y_pred=y_pred)
    if show_info:
        print('f-score:', f)
        print('label wise f-score', cf)
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        labels = list(set(labels))
        sns.heatmap(
            conf_mat, annot=True, cmap="Blues", fmt='d', xticklabels=labels, yticklabels=labels
        )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title("FASTTEXT CONFUSION MATRIX", size=16)
    if return_fscore:
        return f, cf
