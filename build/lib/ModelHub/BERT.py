"""
!pip install tensorflow-addons
!pip install -q -U tensorflow-text
!pip install -q -U tf-models-official
!pip install -U tfds-nightly

"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # A dependency of the preprocessing model
# import tensorflow_addons as tfa  # For metrics
# from official.nlp import optimization

tf.get_logger().setLevel('ERROR')
"""
Configure TFHub to read checkpoints directly from TFHub's Cloud Storage buckets. This is only recommended when running TFHub models on TPU.
Without this setting TFHub would download the compressed file and extract the checkpoint locally. Attempting to load from these local files will fail with the following error:
```
InvalidArgumentError: Unimplemented: File system scheme '[local]' not implemented
```
This is because the [TPU can only read directly from Cloud Storage buckets](https://cloud.google.com/tpu/docs/troubleshooting#cannot_use_local_filesystem).
Note: This setting is automatic in Colab.
"""
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"


class BERT_FineTune:
    def __init__(self, handle_encoder, handle_preprocess):
        if tf.config.list_physical_devices('TPU'):
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            strategy = tf.distribute.TPUStrategy(cluster_resolver)
            print('Using TPU')
        elif tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
            print('Using GPU')
        else:
            strategy = None
            print('Running on CPU is not recommended.')
        self.strategy = strategy
        self.handle_encoder = handle_encoder
        self.handle_preprocess = handle_preprocess

    @staticmethod
    def make_bert_preprocess_model(handle_preprocess, sentence_features, seq_length=128):
        """Returns Model mapping string features to BERT inputs.

        Args:
        sentence_features: a list with the names of string-valued features.
        seq_length: an integer that defines the sequence length of BERT inputs.

        Returns:
        A Keras Model that can be called on a list or dict of string Tensors
        (with the order or names, resp., given by sentence_features) and
        returns a dict of tensors for input to BERT.
        """

        input_segments = [
            tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft) for ft in sentence_features
        ]

        # Tokenize the text to word pieces.
        bert_preprocess = hub.load(handle_preprocess)
        tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
        segments = [tokenizer(s) for s in input_segments]

        # Optional: Trim segments in a smart way to fit seq_length.
        # Simple cases (like this example) can skip this step and let
        # the next step apply a default truncation to approximately equal lengths.
        truncated_segments = segments

        # Pack inputs. The details (start/end token ids, dict of output tensors)
        # are model-dependent, so this gets loaded from the SavedModel.
        packer = hub.KerasLayer(
            bert_preprocess.bert_pack_inputs, arguments=dict(seq_length=seq_length), name='packer'
        )
        model_inputs = packer(truncated_segments)
        return tf.keras.Model(input_segments, model_inputs)

    @staticmethod
    def convert_dataset(
        df,
        batch_size,
        bert_preprocess_model,
        sentence_features,
        label='label',
        shuffle=False,
        repeat=False
    ):
        AUTOTUNE = tf.data.AUTOTUNE
        in_memory_ds = dict()
        for feature in sentence_features:
            in_memory_ds[feature] = df[feature]
        in_memory_ds['label'] = df[label]

        dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds)
        num_examples = len(in_memory_ds['label'])

        if shuffle:
            dataset = dataset.shuffle(num_examples)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        return dataset, num_examples

    @staticmethod
    def build_classifier_model(encoder, num_classes, seed=0):
        tf.random.set_seed(seed)

        class Classifier(tf.keras.Model):
            def __init__(self, encoder, num_classes):
                super(Classifier, self).__init__(name="prediction")
                self.encoder = encoder
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.dense = tf.keras.layers.Dense(num_classes)

            def call(self, preprocessed_text):
                encoder_outputs = self.encoder(preprocessed_text)
                pooled_output = encoder_outputs["pooled_output"]
                x = self.dropout(pooled_output)
                x = self.dense(x)
                return x

        model = Classifier(encoder, num_classes)
        return model

    def tune(
        self,
        train_df,
        val_df=None,
        sentence_features=['sentence'],
        label='label',
        epochs=3,
        batch_size=32,
        optimizer='sgd',
        seq_length=128
    ):
        num_classes = len(train_df[label].unique())
        bert_preprocess_model = BERT_FineTune.make_bert_preprocess_model(
            self.handle_preprocess, sentence_features, seq_length
        )

        train_dataset, train_data_size = BERT_FineTune.convert_dataset(
            train_df, batch_size, bert_preprocess_model, sentence_features, label, True, True
        )

        steps_per_epoch = train_data_size // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = num_train_steps // 10

        if val_df is None:
            validation_dataset = None
            validation_steps = None
        else:
            validation_dataset, validation_data_size = BERT_FineTune.convert_dataset(
                val_df, batch_size, bert_preprocess_model, sentence_features, label, False, False
            )
            validation_steps = validation_data_size // batch_size

        if self.strategy is None:
            encoder = hub.KerasLayer(self.handle_encoder, trainable=True)

            # metric have to be created inside the strategy scope

            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
            classifier_model = BERT_FineTune.build_classifier_model(encoder, num_classes)

            # optimizer = optimization.create_optimizer(
            #     init_lr=init_lr,
            #     num_train_steps=num_train_steps,
            #     num_warmup_steps=num_warmup_steps,
            #     optimizer_type='adamw')

            classifier_model.compile(
                optimizer=optimizer, loss=loss, metrics='accuracy'
            )  # metrics=[metrics]

            classifier_model.fit(
                x=train_dataset,
                validation_data=validation_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_steps=validation_steps
            )
        else:
            with self.strategy.scope():
                encoder = hub.KerasLayer(self.handle_encoder, trainable=True)

                # metric have to be created inside the strategy scope

                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                # metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)

                classifier_model = BERT_FineTune.build_classifier_model(encoder, num_classes)

                # optimizer = optimization.create_optimizer(
                #     init_lr=init_lr,
                #     num_train_steps=num_train_steps,
                #     num_warmup_steps=num_warmup_steps,
                #     optimizer_type='adamw')

                classifier_model.compile(
                    optimizer=optimizer, loss=loss, metrics='accuracy'
                )  # metrics=[metrics]

                classifier_model.fit(
                    x=train_dataset,
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_steps=validation_steps
                )
        return classifier_model
