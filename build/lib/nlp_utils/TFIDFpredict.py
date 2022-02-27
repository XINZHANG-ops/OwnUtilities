from tqdm import tqdm

tqdm.pandas()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from .TextProcess import text_tokens
from xin_util.Scores import single_label_f_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


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


def tf_idf_classify(
    input_df,
    feature_column,
    label_column,
    process_text=False,
    test_size=.2,
    random_state=1,
    classifier='RF',
    return_fscore=True,
    show_info=True,
    **kwargs
):
    """

    :param input_df:
    :param feature_column:
    :param label_column:
    :param process_text:
    :param test_size:
    :param random_state:
    :param classifier: RF,DT,LSVC, LR,NB,KNN
    :param kwargs:
    :return:
    """
    df = input_df.copy()
    if process_text:
        df[feature_column] = df[feature_column].progress_apply(process_sow)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df[feature_column])
    labels = list(df[label_column])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    if classifier == 'RF':
        model = RandomForestClassifier(**kwargs)
    elif classifier == 'DT':
        model = DecisionTreeClassifier(**kwargs)
    elif classifier == 'LSVC':
        model = LinearSVC(**kwargs)
    elif classifier == 'LR':
        model = LogisticRegression(**kwargs)
    elif classifier == 'NB':
        model = GaussianNB(**kwargs)
    elif classifier == 'KNN':
        model = KNeighborsClassifier(**kwargs)
    else:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
        plt.title("TFIDF CONFUSION MATRIX", size=16)
    if return_fscore:
        return f, cf
