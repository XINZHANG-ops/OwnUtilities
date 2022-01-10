import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from .TextProcess import text_tokens
from xin_util.Scores import single_label_f_score


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


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def Onehot_Network_classify(
    input_df,
    feature_column,
    label_column,
    process_text=False,
    test_size=.2,
    random_state=1,
    validation_rate=.1,
    MAX_NB_WORDS=10000,
    hidden_layer_size=300,
    second_layer=None,
    batch_size=100,
    epochs=10,
    show_model_summary=True,
    plot_network=True,
    show_validation_plot=True
):
    """

    :param input_df:
    :param feature_column:
    :param label_column:
    :param process_text:
    :param test_size:
    :param random_state:
    :param validation_rate:
    :param MAX_NB_WORDS:
    :param hidden_layer_size:
    :param second_layer: int or none
    :param batch_size:
    :param epochs:
    :param show_model_summary:
    :param plot_network:
    :param show_validation_plot:
    :return:
    """
    df = input_df.copy()
    if process_text:
        df[feature_column] = df[feature_column].progress_apply(process_sow)
    else:
        df[feature_column] = df[feature_column].apply(process_sow_quick)
    features = list(df[feature_column])
    labels = list(df[label_column])
    label_names = set(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        features, onehot_encoded_labels, test_size=test_size, random_state=random_state
    )
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(X_train + X_test)
    word_seq_train = tokenizer.texts_to_sequences(X_train)
    word_seq_test = tokenizer.texts_to_sequences(X_test)
    x_train = vectorize_sequences(word_seq_train, MAX_NB_WORDS)
    x_test = vectorize_sequences(word_seq_test, MAX_NB_WORDS)
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation='relu', input_shape=(MAX_NB_WORDS, )))
    if second_layer:
        model.add(Dense(second_layer, activation='relu'))
    model.add(Dense(len(label_names), activation='softmax'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if show_model_summary:
        print(model.summary())
    if plot_network:
        plot_model(model, show_shapes=True)
        plt.show()
    val_size = int(len(x_train) * validation_rate)
    x_val = x_train[:val_size]
    patial_x_train = x_train[val_size:]
    y_val = y_train[:val_size]
    patial_y_train = y_train[val_size:]
    history = model.fit(
        patial_x_train,
        patial_y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val)
    )
    if show_validation_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], lw=2.0, color='b', label='train')
        plt.plot(history.history['val_loss'], lw=2.0, color='r', label='val')
        plt.title('NN sentiment')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend(loc='upper right')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], lw=2.0, color='b', label='train')
        plt.plot(history.history['val_accuracy'], lw=2.0, color='r', label='val')
        plt.title('NN sentiment')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.show()

    y_test = list(
        label_encoder.inverse_transform(
            onehot_encoder.inverse_transform(y_test).astype(int).ravel()
        )
    )
    y_pred = list(
        label_encoder.inverse_transform(
            np.argmax(model.predict(x_test), axis=1).astype(int).ravel()
        )
    )
    f, cf = single_label_f_score(y_gold=y_test, y_pred=y_pred)
    print('f-score:', f)
    print('label wise f-score', cf)
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("CONFUSION MATRIX", size=16)
