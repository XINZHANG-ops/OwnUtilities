#importing libraries
from tqdm import tqdm
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem import WordNetLemmatizer
from xin_util.Scores import single_label_f_score
from sklearn.metrics import confusion_matrix


def tf_idf_classify(
    input_df,
    feature_column,
    label_column,
    test_size=.2,
    return_fscore=True,
    show_info=True,
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

    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # stopword removal and lemmatization
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    msk = np.random.rand(len(df)) < 1 - test_size
    train_df = df[msk]
    val_df = df[~msk]

    train_X = []
    test_X = []

    train_y = train_df[label_column].tolist()
    test_y = val_df[label_column].tolist()

    labels = list(df[label_column])
    labels = [str(l) for l in labels]

    # text pre processing
    for text in tqdm(train_df[feature_column]):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        train_X.append(review)

    # text pre processing
    for text in tqdm(val_df[feature_column]):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        test_X.append(review)

    # tf idf
    tf_idf = TfidfVectorizer()
    # applying tf idf to training data
    X_train_tf = tf_idf.fit_transform(train_X)
    # applying tf idf to training data
    X_train_tf = tf_idf.transform(train_X)

    # transforming test data into tf-idf matrix
    X_test_tf = tf_idf.transform(test_X)

    # naive bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, train_y)

    # predicted y
    y_pred = naive_bayes_classifier.predict(X_test_tf)

    f, cf = single_label_f_score(y_gold=test_y, y_pred=y_pred)

    if show_info:
        print('f-score:', f)
        print('label wise f-score', cf)
        conf_mat = confusion_matrix(test_y, y_pred)
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
