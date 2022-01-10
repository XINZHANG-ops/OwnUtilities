import matplotlib
import string
from nltk.stem.snowball import SnowballStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from numpy import *
import numpy as np
import csv
import nltk as nltk
import pandas as pd
from nltk import ngrams
import re
import matplotlib.pyplot as plt
from numpy import *
from xin_util.CreateDIYdictFromDataFrame import CreateDIYdictFromDataFrame
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS

from nltk.corpus import words as W
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

tqdm.pandas()

##########################################################################
# Word Resources
##########################################################################
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')

some_other_words = ['your exclude words']

Englishtext = set(W.words())

my_stopwords = {'able'}
stop_words = set([word for word in stopwords.words('english')])
stop_words = stop_words.union(STOPWORDS).union(STOP_WORDS).union(my_stopwords)

##########################################################################
# Data Cleaning functions for projects SOW
##########################################################################
'''
This function expect no nan in the inout datafram, so replace nan with empty string

projectSowData is a pandas dataframe where the columns need to
 include projectID and StatementOfWork

it will return a pandas df and a dictionary, the df is the filtered dataframe
the dictionary is the one with removed id and SOW
'''


def remove_test_project_from_sow(
    projectSowData, projectID_column='ProjectID', StatmentOfWork_column='StatementOfWork'
):
    df = projectSowData.copy()
    removed_dict = dict()
    pattern = re.compile(
        r'^test|\stest$|\stesting\s|\stesting|testing\s|example|template', re.IGNORECASE
    )
    for index, row in projectSowData.iterrows():
        # <=20 since some sow with test in it but not a test project, when it is actually quite long
        if re.findall(pattern, row[StatmentOfWork_column]) \
                and len(nltk.word_tokenize(row[StatmentOfWork_column]))<=20:
            df = df.drop(index)
            removed_dict[row[projectID_column]] = row[StatmentOfWork_column]
    return df, removed_dict


# remove sow with word example in it
def remove_example_project_from_df(
    projectSowData, projectID_column='ProjectID', target_remove_column='StatementOfWork'
):
    df = projectSowData.copy()
    removed_dict = dict()
    pattern = re.compile(r'example', re.IGNORECASE)
    for index, row in projectSowData.iterrows():
        # <=20 since some sow with test in it but not a test project, when it is actually quite long
        if re.findall(pattern, row[target_remove_column]):
            df = df.drop(index)
            removed_dict[row[projectID_column]] = row[target_remove_column]
    return df, removed_dict


'''
Remove Projects by Sow length
'''


def remove_projects_by_sow_length(
    projectSowData,
    lower_bound=None,
    upper_bound=None,
    projectID_column='ProjectID',
    StatmentOfWork_column='StatementOfWork'
):
    """A Function that will remove the projects by length of SOW
    This function expect no nan in the inout datafram, so replace nan with empty string
    """
    '''
    #####Inputs:

    projectSowData: A pandas dataframe of including projectID and SOW

    lower_bound: An int for least amount of tokens a sow need to have

    upper_bound: An int for most amount of tokens a sow can have

    #####Output:
    it will return a pandas df and a dictionary, the df is the filtered dataframe
    the dictionary is the one with removed id and SOW
    '''

    df = projectSowData.copy()
    removed_dict = dict()
    if lower_bound is None and upper_bound is None:
        return df, removed_dict
    elif upper_bound is None:
        for index, row in projectSowData.iterrows():
            if len(nltk.word_tokenize(row[StatmentOfWork_column])) < lower_bound:
                df = df.drop(index)
                removed_dict[row[projectID_column]] = row[StatmentOfWork_column]
        return df, removed_dict
    elif lower_bound is None:
        for index, row in projectSowData.iterrows():
            if len(nltk.word_tokenize(row[StatmentOfWork_column])) > upper_bound:
                df = df.drop(index)
                removed_dict[row[projectID_column]] = row[StatmentOfWork_column]
        return df, removed_dict
    else:
        for index, row in projectSowData.iterrows():
            L = len(nltk.word_tokenize(row[StatmentOfWork_column]))
            if lower_bound <= L <= upper_bound:
                df = df.drop(index)
                removed_dict[row[projectID_column]] = row[StatmentOfWork_column]
        return df, removed_dict


'''
Remove Certain type of projects? Keep RFP.
'''


def remove_outlier_by_cc_freq(
    projectccdf, minimal_freq=1, projectID_column='ProjectID', TokenValue='Token1Value'
):
    """A Function that will remove outlier cc from dataframe.
    filter out Commodity Codes appear specific number of times in the data,
    if the SOW only has that code, remove the SOW as well

    Just keep in mind that we expect different CC codes how different rows for a projectID
    """
    '''
    #####Inputs:

    projectccdf: A pandas dataframe of including projectID and TokenValue
    
    minimal_freq: An int to select the minimal frequency of ccs to keep

    projectID_column: The column name for projectid

    TokenValue: The column name for tokenvalue, not necessary be Token1Value, since if we
                want to predict to 2nd level, we can combine Token1value and Token2value as a 
                new column name

    #####Output:
    it will return a pandas df and a dictionary, the df is the filtered dataframe
    the dictionary is the one with removed id and CCs
    '''
    df = projectccdf.copy()
    OBJ = CreateDIYdictFromDataFrame(projectccdf)
    DICT = OBJ.DIY_dict([TokenValue, projectID_column], convert_to=set)
    removed_dict = dict()
    remove_codes = {code for code, pset in DICT.items() if len(pset) < minimal_freq}
    for index, row in projectccdf.iterrows():
        token = row[TokenValue]
        if token in remove_codes:
            df = df.drop(index)
            removed_dict[row[projectID_column]] = row[TokenValue]
    return df, removed_dict


def remove_data_points_for_CC_level(
    projectSowData, keepiDigit, remove_string='00', projectID_column='ProjectID'
):
    '''
    This function expect no nan in the inout datafram, so replace nan with empty string
    This function is used for further level model data clean. So if we want predict on second
    level commodity codes, we want to removes rows that with token2value is '00'

    :param projectSowData: a pandas dataframe where the columns need to
           include tokenvalue_column_nameand column and ProjectID column
    :param keepiDigit: a string indicates the column name for commodity code level,
           This is depend on which level model you are trying to build
    :param remove_string: a string to indicate what is look like for empty commodity code
    :param projectID_column: a string for the column name for projectid
    :return:
    it will return a pandas df and a dictionary, the df is the filtered dataframe
    the dictionary is the one with removed id and commodity codes
    '''
    df = projectSowData.copy()
    removed_dict = nltk.defaultdict(list)
    for index, row in projectSowData.iterrows():
        if row[keepiDigit].endswith('-' + remove_string):
            df = df.drop(index)
            removed_dict[row[projectID_column]].append(row[keepiDigit])
    return df, removed_dict


##########################################################################
# Methods to tokenize text data (data preparation methods)
##########################################################################
def lemmatizer():
    wnl = nltk.WordNetLemmatizer()
    return wnl


def SnowballStemmer_():
    stemmer = SnowballStemmer("english")
    return stemmer


def remove_adjacent(seq):
    i = 1
    n = len(seq)
    while i < n:
        if seq[i] == seq[i - 1]:
            del seq[i]
            n -= 1
        else:
            i += 1


def text_tokens(
    text,
    lower_bound_percentage=0,
    higher_bound_percentage=1,
    minimal_word_length=0,
    lower_case=False,
    remove_punctuations=False,
    remove_non_letter_characters=False,
    lemmatize_the_words=False,
    stemmer_the_words=False,
    add_pos_feature=False,
    url_filter=False,
    parentheses_filter=False,
    prime_s_filter=False,
    number_filter=False,
    part_of_speech_filter=False,
    english_text_filter=False,
    stop_words_filter=False,
    other_words_filter=False,
    remove_adjacent_tokens=False,
    tokens_form=True,
    stop_words=stop_words,
    some_other_words=some_other_words
):
    if lower_case:
        text = text.lower()
        #Englishtext = set(w.lower() for w in W.words())
    text = re.sub(r'\n', "", text)
    if url_filter:
        url_pattern = re.compile(
            r'((http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-zA-Z0-9]+([\-\.]{1}[a-zA-Z0-9]+)*\.[a-zA-Z]{2,5}(:[0-9]{1,5})?(\/.*)?)'
        )
        text = re.sub(url_pattern, " ", text)
    if parentheses_filter:
        parentheses_pattern = re.compile(r'(\([^)]+\))')
        text = re.sub(parentheses_pattern, " ", text)
    if prime_s_filter:
        prime_s_pattern = r"('s|\?s)"
        text = re.sub(prime_s_pattern, "", text)
    if remove_punctuations:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if remove_non_letter_characters:
        text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    if number_filter:
        text = re.sub(r'[0-9]', " ", text)
    tokens = nltk.word_tokenize(text)
    howmany_tokens = len(tokens)
    if stop_words_filter:
        tokens = [token for token in tokens if token not in stop_words]
    tokens = tokens[int(howmany_tokens *
                        lower_bound_percentage):int(ceil(howmany_tokens * higher_bound_percentage))]
    if part_of_speech_filter:
        token_pos = nltk.pos_tag(tokens)
        tokens = [word for (word, pos) in token_pos if pos.startswith('N')]
    if add_pos_feature:
        token_pos = nltk.pos_tag(tokens)
        tokens = [word + '_' + pos for (word, pos) in token_pos]
    if english_text_filter:
        if add_pos_feature:
            tokens = [token for token in tokens if token.split('_')[0] in Englishtext]
        else:
            tokens = [token for token in tokens if token in Englishtext]
    if lemmatize_the_words:
        if add_pos_feature:
            tokens = [
                lemmatizer().lemmatize(token.split('_')[0]) + '_' + token.split('_')[1]
                for token in tokens
            ]
        else:
            tokens = [lemmatizer().lemmatize(token) for token in tokens]
        #stop_words = set([lemmatizer().lemmatize(word) for word in stopwords.words('english')])
        some_other_words = set([lemmatizer().lemmatize(word) for word in some_other_words])
    if stemmer_the_words:
        if add_pos_feature:
            tokens = [
                SnowballStemmer_().stem(token.split('_')[0]) + '_' + token.split('_')[1]
                for token in tokens
            ]
        else:
            tokens = [SnowballStemmer_().stem(token) for token in tokens]
        #stop_words = set([SnowballStemmer_().stem(word) for word in stopwords.words('english')])
        some_other_words = set([SnowballStemmer_().stem(word) for word in some_other_words])
    if add_pos_feature:
        tokens = [token for token in tokens if len(token.split('_')[0]) >= minimal_word_length]
    else:
        tokens = [token for token in tokens if len(token) >= minimal_word_length]
    if other_words_filter:
        if add_pos_feature:
            tokens = [token for token in tokens if token.split('_')[0] not in some_other_words]
        else:
            tokens = [token for token in tokens if token not in some_other_words]
    if remove_adjacent_tokens:
        remove_adjacent(tokens)
    if tokens_form:
        return tokens
    else:
        return ' '.join(tokens)


def text_tokens_2(
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
    tokens_form=True,
    stop_words=stop_words,
    some_other_words=some_other_words
):
    text = text.lower()
    if remove_punctuations:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if remove_non_letter_characters:
        text = re.sub(r'[^a-zA-Z]', " ", text)
    tokens = nltk.word_tokenize(text)
    howmany_tokens = len(tokens)
    tokens = tokens[int(howmany_tokens *
                        lower_bound_percentage):int(ceil(howmany_tokens * higher_bound_percentage))]
    if part_of_speech_filter:
        token_pos = nltk.pos_tag(tokens)
        tokens = [word for (word, pos) in token_pos if pos.startswith('N') or pos.startswith('J')]
    if english_text_filter:
        tokens = [token for token in tokens if token in Englishtext]
    if lemmatize_the_words:
        tokens = [lemmatizer().lemmatize(token) for token in tokens]
        stop_words = set([lemmatizer().lemmatize(word) for word in stopwords.words('english')])
        some_other_words = set([lemmatizer().lemmatize(word) for word in some_other_words])
    if stemmer_the_words:
        tokens = [SnowballStemmer_().stem(token) for token in tokens]
        stop_words = set([SnowballStemmer_().stem(word) for word in stopwords.words('english')])
        some_other_words = set([SnowballStemmer_().stem(word) for word in some_other_words])
    tokens = [token for token in tokens if len(token) >= minimal_word_length]
    if other_words_filter:
        tokens = [token for token in tokens if token not in some_other_words]
    p = nltk.pos_tag(tokens)
    grammar = r"""
         NP: {(<DT>|<JJ>*)<NN.*>+(<CC><NN.*>+)?}    # noun phrase chunks
         VP: {<TO>?<VB.*>}          # verb phrase chunks
         PP: {<IN>}                 # prepositional phrase chunks
         CLAUSE: {<VP>?<NP>+}
         """
    cp = nltk.RegexpParser(grammar)
    if p:
        result = cp.parse(p)
        tree = result.subtrees()
        goodones = []
        badones = []
        for sub in tree:
            if sub.label() == 'CLAUSE':
                if len(list(sub)) >= 3:
                    goodones.append(sub)
                else:
                    badones.append(sub)
        tokens = []
        if goodones:
            for g in goodones:
                for w, po in g.leaves():
                    tokens.append(w)
        else:
            for b in badones:
                for w, po in b.leaves():
                    tokens.append(w)
        if stop_words_filter:
            tokens = [token for token in tokens if token not in stop_words]
        if remove_adjacent_tokens:
            remove_adjacent(tokens)
        if tokens_form:
            return tokens
        else:
            return ' '.join(tokens)
    else:
        return []


'''
list_of_texts is a list of strings (list of sow)
where the method is tokenize method are the 2 functions above text_tokens and text_tokens_2
return a list of lists of tokens of the SOWs
if skip_empty, means if the token list is empty of a sow, it also return a removed index of remove positions
in the original list_of_texts, this will be helpful for the label list, we can use the removed_empty_index remove
corresponding labels

pre_processed_dict is a dictionary where key and values are string, when input this dictionary,
the text in the keys will directly use the value as the processed text to save time
'''


def get_tokens_for_all(list_of_texts, method, skip_empty=True, pre_processed_dict=None, **kwargs):
    data_list = []
    removed_empty_index = []
    if pre_processed_dict is None:
        processed_dict = dict()
    else:
        processed_dict = pre_processed_dict.copy()
    for index, text in enumerate(tqdm(list_of_texts)):
        try:
            Tokens = pre_processed_dict[text]
        except:
            Tokens = method(text, **kwargs)
            processed_dict[text] = Tokens
        if skip_empty:
            if Tokens:
                data_list.append(Tokens)
            else:
                removed_empty_index.append(index)
        else:
            data_list.append(Tokens)
    return data_list, processed_dict, removed_empty_index


#####################################################################################
# Get Ngrams from Tokens
#####################################################################################


def Ngrams_for_tokens(Tokens, n=3):
    Ngrams = ngrams(Tokens, n)
    NgramL = []
    Ngram_freq = nltk.FreqDist(Ngrams)
    for k, v in Ngram_freq.items():
        NgramL.append((k, v))
    NgramFreqTable = pd.DataFrame(
        NgramL, columns=['Ngram', 'freq']
    ).sort_values(
        by='freq', ascending=False
    )
    return NgramFreqTable


#####################################################################################
#Following Functions To get vector Data from Tokens
#####################################################################################


def normalization(data):
    mins = []
    maxs = []
    number_of_datafeatures = data.shape[1]
    number_of_samples = data.shape[0]
    for i in range(number_of_datafeatures):
        maxs = maxs + [data[:, i].max()]
        mins = mins + [data[:, i].min()]
    maxs = np.array(maxs)
    mins = np.array(mins)
    minM = tile(mins, (number_of_samples, 1))
    max_minM = tile(maxs - mins, (number_of_samples, 1))
    normal = (data - minM) / max_minM
    return normal


'''
get_data_from_tokens(data,vector_size=5, window=2, min_count=1, workers=4))
target_list_of_lists_of_tokens must be a subset of train_list_of_list_of_tokens, which means
all train_list_of_list_of_tokens must be in train_list_of_list_of_tokens
'''


def get_data_from_d2v(
    train_list_of_list_of_tokens, target_list_of_lists_of_tokens, return_model=False, **kwargs
):
    datalist = []
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_list_of_list_of_tokens)]
    model = Doc2Vec(documents, **kwargs)
    for index in range(len(target_list_of_lists_of_tokens)):
        datalist.append(model.docvecs[index])
    data = np.array(datalist)
    if return_model:
        return model, data
    else:
        return data


'''
model = Word2Vec(list_of_lists_of_tokens, size=5, window=5, min_count=1, workers=4)


target_list_of_lists_of_tokens must be a subset of train_list_of_list_of_tokens to use tfidf_weight,
since tfidf_weight obtained by training on train_list_of_list_of_tokens

The data in the end is a list of vectors. Each vector is the data for the doc, obtained by summing up the w2v in the doc
'''


def get_data_from_w2v(
    train_list_of_list_of_tokens,
    target_list_of_lists_of_tokens,
    tfidf_weight=False,
    normalize_vector_for_TF_IDF=True,
    return_model=False,
    **kwargs
):
    datalist = []
    model = Word2Vec(train_list_of_list_of_tokens, **kwargs)
    Aword = train_list_of_list_of_tokens[0][0]
    dim = len(model.wv.get_vector(Aword))
    if tfidf_weight:
        train_data_dict_for_tf_idf = dict()
        for index, token_list in enumerate(train_list_of_list_of_tokens):
            SOW = ' '.join(token_list)
            train_data_dict_for_tf_idf[SOW] = index
        tfidf_vec = get_tfidf_from_context(
            train_list_of_list_of_tokens, normalize_vector=normalize_vector_for_TF_IDF
        )
    for doc in target_list_of_lists_of_tokens:
        docvec = np.zeros(dim)
        for word in doc:
            wordvec = model.wv.get_vector(word)
            if tfidf_weight:
                doc_SOW = ' '.join(doc)
                index = train_data_dict_for_tf_idf[doc_SOW]
                wordweight = tfidf_vec[index][word]
                docvec += wordvec * wordweight
            else:
                docvec += wordvec
        datalist.append(list(docvec))
    data = np.array(datalist)
    if return_model:
        return model, data
    else:
        return data


'''
model = FastText(size=4, window=3, min_count=1, workers=4)
'''


def get_data_from_ft_tf_idf(
    train_list_of_list_of_tokens,
    target_list_of_lists_of_tokens,
    tfidf_weight=False,
    normalize_vector_for_TF_IDF=True,
    return_model=False,
    **kwargs
):
    datalist = []
    model = FastText(**kwargs)
    model.build_vocab(sentences=train_list_of_list_of_tokens)
    model.train(
        sentences=train_list_of_list_of_tokens,
        total_examples=len(train_list_of_list_of_tokens),
        epochs=kwargs.get('epochs', 10)
    )
    Aword = train_list_of_list_of_tokens[0][0]
    dim = len(model.wv.get_vector(Aword))
    if tfidf_weight:
        train_data_dict_for_tf_idf = dict()
        for index, token_list in enumerate(train_list_of_list_of_tokens):
            SOW = ' '.join(token_list)
            train_data_dict_for_tf_idf[SOW] = index
        tfidf_vec = get_tfidf_from_context(
            train_list_of_list_of_tokens, normalize_vector=normalize_vector_for_TF_IDF
        )
    for doc in target_list_of_lists_of_tokens:
        docvec = np.zeros(dim)
        for word in doc:
            wordvec = model.wv.get_vector(word)
            if tfidf_weight:
                doc_SOW = ' '.join(doc)
                index = train_data_dict_for_tf_idf[doc_SOW]
                wordweight = tfidf_vec[index][word]
                docvec += wordvec * wordweight
            else:
                docvec += wordvec
        datalist.append(list(docvec))
    data = np.array(datalist)
    if return_model:
        return model, data
    else:
        return data


class gensim_ft:
    def __init__(self, **kwargs):
        model = FastText(**kwargs)
        self.model = model

    def fit(self, train_list_of_list_of_text, epoch=10):
        train_list_of_list_of_tokens = [text.split(' ') for text in train_list_of_list_of_text]
        self.model.build_vocab(sentences=train_list_of_list_of_tokens)
        self.model.train(
            sentences=train_list_of_list_of_tokens,
            total_examples=len(train_list_of_list_of_tokens),
            epochs=epoch
        )

    def transform_data(self, list_of_list_of_text):
        list_of_list_of_tokens = [text.split(' ') for text in list_of_list_of_text]
        Aword = list_of_list_of_tokens[0][0]
        dim = len(self.model.wv.get_vector(Aword))
        datalist = []
        for doc in list_of_list_of_tokens:
            docvec = np.zeros(dim)
            for word in doc:
                wordvec = self.model.wv.get_vector(word)
                docvec += wordvec
            docvec /= len(doc)
            datalist.append(list(docvec))
        data = np.array(datalist)
        return data


# class facebook__unsupft:
#     @staticmethod
#     def to_txt(data, label, name, mode='w'):
#         Data = []
#         for index, d in enumerate(data):
#             Data.append(['__label__' + label[index], d])
#         df = pd.DataFrame(Data, columns=['Token', 'Statement'])
#         df.to_csv(r'{}'.format(name), header=None, index=None, sep=' ', mode=mode)
#


# this allows input the trained w2v model and get the data for our target
def get_data_from_w2v_model(model, target_list_of_lists_of_tokens, ignore_new_words=True):
    datalist = []
    dim = model.vector_size
    for index, doc in enumerate(target_list_of_lists_of_tokens):
        docvec = np.zeros(dim)
        for word in doc:
            if ignore_new_words:
                try:
                    wordvec = model.wv.get_vector(word)
                except:
                    wordvec = np.zeros(dim)
            else:
                wordvec = model.wv.get_vector(word)
            wordweight = 1 / len(doc)
            docvec += wordvec * wordweight
        datalist.append(list(docvec))
    data = np.array(datalist)
    return data


# get tfidf from list of lists of tokens
# remember we keep tokens with gigh tf-idf
# It returns the a list of dictionaries, the order of dictionaries corresponding to the order of input list_of_list_of_tokens
def get_tfidf_from_context(list_of_list_of_tokens, normalize_vector=True):
    outputlist = []
    dct = Dictionary(list_of_list_of_tokens)
    corpus = [dct.doc2bow(line) for line in list_of_list_of_tokens]
    model = TfidfModel(corpus)
    for index, text in enumerate(list_of_list_of_tokens):
        vector = model[corpus[index]]
        docdict = dict()
        if normalize_vector:
            vector = [(id, value**2) for id, value in vector]
        for wordid, tfidf in vector:
            docdict[dct[wordid]] = tfidf
        outputlist.append(docdict)
    return outputlist


#####################################################################################
# Some useful functions
#####################################################################################
def get_single_label_maodel_accuracy_for_each_label(trained_model, x_test, y_test):
    """A Function that will give the accuracy for each label in the y_test, the model should be
       single label model"""
    '''
    #####Inputs:
    
    model: should be a mahcine learning algorithm that with method predict, etc model.predict(x_test)=y_predict
    
    x_test: numpy array for test data
    
    y_test: numpy array for test labels

    #####Output:
    A dictionary, where the keys are the labels in y_test, the values are the accuracy for that 
    particular label predicted by the model on x_test
    '''
    y_predict = trained_model.predict(x_test)
    Accuracy_dict = dict()
    unique_labels = set(y_test)
    for label in unique_labels:
        Accuracy_dict[label] = [0, 0]
    for index, true_label in enumerate(y_test):
        pred_label = y_predict[index]
        Accuracy_dict[true_label][1] += 1
        if true_label == pred_label:
            Accuracy_dict[true_label][0] += 1
        else:
            pass
    for label, count_list in Accuracy_dict.items():
        Accuracy_dict[label] = count_list[0] / count_list[1]
    return Accuracy_dict


def get_keys_words_use_tf_idf1(document, num_words):
    """
    This function use tf_idf to find keywords for a corpus
    :param document: A list of strings
    :param num_words: an int, number of keywords for all doc, or a float in (0,1) for percentage of remaining
    :return: a list of string
    """
    tfidf_vectorizer1 = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer1.fit_transform(document)
    total_words = len(tfidf_vectorizer1.get_feature_names())
    if isinstance(num_words, int):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=num_words)
        tfidf_vectorizer.fit_transform(document)
        key_words = tfidf_vectorizer.get_feature_names()
        return key_words
    if isinstance(num_words, float):
        num_words = int(total_words * num_words)
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=num_words)
        tfidf_vectorizer.fit_transform(document)
        key_words = tfidf_vectorizer.get_feature_names()
        return key_words


def get_keys_words_use_tf_idf2(document, num_words):
    """
    This function use tf_idf to find keywords for a corpus
    :param document: A list of strings
    :param num_words: an int, number of keywords for all doc, or a float in (0,1) for percentage of remaining
    :return: a list of string
    """
    tfidf_vectorizer1 = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer1.fit_transform(document)
    all_words = set(tfidf_vectorizer1.get_feature_names())
    total_words = len(all_words)
    if isinstance(num_words, int):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=total_words - num_words)
        tfidf_vectorizer.fit_transform(document)
        key_words = list(all_words.difference(set(tfidf_vectorizer.get_feature_names())))
        return key_words
    if isinstance(num_words, float):
        num_words = int(total_words * num_words)
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=total_words - num_words)
        tfidf_vectorizer.fit_transform(document)
        key_words = list(all_words.difference(set(tfidf_vectorizer.get_feature_names())))
        return key_words


def get_key_words_for_dataframe(
    projectccdf,
    key_word_method=get_keys_words_use_tf_idf1,
    remain=.9,
    projectID_column='ProjectID',
    StatementOfWork_column='StatementOfWork'
):
    """

    :param projectccdf: target dataframe, expect to have StatementOfWork, and ProjectID column (ProjectID column can be index as well)
    :param key_word_method: get_keys_words_use_tf_idf1 or get_keys_words_use_tf_idf2
    :param remain: a float or an int. Means take how many words as key words, out of all words from corpora
    :param projectID_column: a string, the name of ProjectID column
    :param StatementOfWork_column: a string, the name of StatementOfWork column
    :return:
    a dataframe that was processed by the method of key_word_method
    """
    df = projectccdf.copy()
    OBJ = CreateDIYdictFromDataFrame(projectccdf)
    DICT = OBJ.DIY_dict([projectID_column, StatementOfWork_column], convert_to=set)
    document = [list(sow_set)[0] for pid, sow_set in DICT.items()]
    key_words = set(key_word_method(document, num_words=remain))

    def process(text):
        tokens = text.split(' ')
        tokens = [t for t in tokens if t in key_words]
        return ' '.join(tokens)

    df[StatementOfWork_column] = df[StatementOfWork_column].progress_apply(process)
    return df


def find_key_words_by_w2v_with_cc_titles(
    projectccdf,
    pretrained_map_dict,
    remain=.9,
    projectID_column='ProjectID',
    StatmentOfWork_column='StatementOfWork',
    cc_title_column='Title'
):
    """
    This function is trying to find keywords in sow that have similar meaning of their cc titles using method of
    word2vec

    :param projectccdf: target dataframe, need to have projectID column, StatmentOfWork column and cc_title_column
    :param pretrained_map_dict: a dictionary where the keys are strings(words), and values are array of word vectors
    :param remain: a float, tell percentage of of words remain for each sow
    :param projectID_column: string, the name of projectID_column
    :param StatmentOfWork_column: string, the name of StatmentOfWork_column
    :param cc_title_column: string, the name of cc_title_column
    :return:
    a filtered dataframe, and a dict of filtered words for each projectID
    """
    df = projectccdf.copy()
    OBJ = CreateDIYdictFromDataFrame(projectccdf)
    PID2TITLEDICT = OBJ.DIY_dict([projectID_column, cc_title_column], convert_to=set)
    PID2SOWDICT = OBJ.DIY_dict([projectID_column, StatmentOfWork_column], convert_to=set)
    removed_dict = nltk.defaultdict(lambda: nltk.defaultdict(dict))
    pretrained = pretrained_map_dict
    dim = len(pretrained[list(pretrained.keys())[0]])
    PID_filter_sow_dict = dict()
    for PID, title_set in tqdm(PID2TITLEDICT.items()):
        sow1 = list(PID2SOWDICT[PID])[0].split(' ')
        sow = [w for w in sow1 if w in pretrained]
        removed_not_in_pretrained = [w for w in sow1 if w not in pretrained]
        title = ' '.join(title_set).split(' ')
        title = [w for w in title if w in pretrained]
        word_dist_dict = nltk.defaultdict(lambda: 0)
        for sow_word in sow:
            sow_word_vec = pretrained[sow_word].reshape(1, dim)
            for title_word in title:
                title_word_vec = pretrained[title_word].reshape(1, dim)
                dist = cosine_distances(sow_word_vec, title_word_vec)[0][0]
                word_dist_dict[sow_word] += dist
        ranked_sow_word = [k for k, v in sorted(word_dist_dict.items(), key=lambda item: item[1])]
        ranked_sow_word_set = set(ranked_sow_word[:int(len(ranked_sow_word) * remain)])
        removed_words_by_rank = set(ranked_sow_word).difference(ranked_sow_word_set)
        sow_string = ' '.join([w for w in sow if w in ranked_sow_word_set])
        PID_filter_sow_dict[PID] = sow_string
        removed_dict[PID]['removed_not_in_pretrained'] = set(removed_not_in_pretrained)
        removed_dict[PID]['removed_words_by_dist'] = removed_words_by_rank
    for pid, text in tqdm(PID_filter_sow_dict.items()):
        df.loc[df[projectID_column] == pid, StatmentOfWork_column] = text
    return df, removed_dict


def get_key_words_for_dataframe_own_tf_idf(
    projectccdf,
    remain=.9,
    projectID_column='ProjectID',
    StatmentOfWork_column='StatementOfWork',
    normalize_vector=True
):
    """
    This function is trying to filter out low tfidf words from each sow
    :param projectccdf: the target dataframe, need to have projectID_column and StatmentOfWork_column
    :param remain: a float, the percentage of remaining words, for each sow. Not calculated by all words
    :param projectID_column: string, the name of projectid column
    :param StatmentOfWork_column: string, the name of sow column
    :param normalize_vector: boolean, whether to normalize the tfidfvectors when calculating
    :return:
    a data frame and a dictionary. The dataframe is the modified dataframe, the dictionary where keys are
    projectIDs, values are words removed of that projectID
    """
    df = projectccdf.copy()
    OBJ = CreateDIYdictFromDataFrame(projectccdf)
    PID2SOWDICT = OBJ.DIY_dict([projectID_column, StatmentOfWork_column], convert_to=set)
    PIDS = list(PID2SOWDICT.keys())
    doc = []
    removed_dict = nltk.defaultdict(set)
    PID_filter_sow_dict = dict()
    for PID in PIDS:
        sow_set = PID2SOWDICT[PID]
        doc.append(list(sow_set)[0].split(' '))
    outputlist = get_tfidf_from_context(doc, normalize_vector=normalize_vector)
    for index, value_dict in enumerate(outputlist):
        id = PIDS[index]
        sorted_tokens = [
            k for k, v in sorted(value_dict.items(), key=lambda item: item[1], reverse=True)
        ]
        remained = set(sorted_tokens[:int(len(sorted_tokens) * remain)])
        real_tokens = list(PID2SOWDICT[id])[0].split(' ')
        modified_tokens = []
        removed_set = set()
        for t in real_tokens:
            if t in remained:
                modified_tokens.append(t)
            else:
                removed_set.add(t)
        text = ' '.join(modified_tokens)
        PID_filter_sow_dict[id] = text
        removed_dict[id] = removed_set
    for pid, text in tqdm(PID_filter_sow_dict.items()):
        df.loc[df[projectID_column] == pid, StatmentOfWork_column] = text
    return df, removed_dict
