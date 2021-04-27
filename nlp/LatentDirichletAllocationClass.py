import numpy as np
from .TextProcess import text_tokens
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


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


class MyLDA:
    def __init__(
        self,
        input_df,
        feature_column,
        process_text=False,
        random_state=1,
        max_features=1000,
        alpha=None,
        eta=None,
        n_topics=10,
        batch_size=128,
        max_iter=10,
        verbose=1
    ):
        df = input_df.copy()
        if process_text:
            df[feature_column] = df[feature_column].progress_apply(process_sow)
        else:
            df[feature_column] = df[feature_column].apply(process_sow_quick)
        features = list(df[feature_column])
        count_vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
        X_train = features
        count_train = count_vectorizer.fit_transform(X_train)
        lda = LDA(
            n_components=n_topics,
            doc_topic_prior=alpha,
            topic_word_prior=eta,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        lda.fit(count_train)
        self.lda = lda
        self.count_vectorizer = count_vectorizer
        self.X_train = X_train

    def get_topics(self, number_words=10):
        lda = self.lda
        count_vectorizer = self.count_vectorizer
        words = count_vectorizer.get_feature_names()
        topic_dict = dict()
        for topic_idx, topic in enumerate(lda.components_):
            topic_dict[topic_idx] = [words[i] for i in topic.argsort()[:-number_words - 1:-1]]
        return topic_dict

    def predict_topic(self, list_of_texts):
        count_vectorizer = self.count_vectorizer
        lda = self.lda
        count_test = count_vectorizer.transform(list_of_texts)
        doc_topic_dist_unnormalized = np.array(lda.transform(count_test))
        return doc_topic_dist_unnormalized.argmax(axis=1)
