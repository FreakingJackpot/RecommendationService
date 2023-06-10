import os
from collections import defaultdict
import pickle

import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

from extensions.extensions import db_connections

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_RATING_COL as RATING_COL,
    DEFAULT_GENRE_COL as ITEM_FEAT_COL,
    SEED
)

from recommenders.utils import tf_utils
from recommenders.datasets.python_splitters import python_random_split
import recommenders.models.wide_deep.wide_deep_utils as wide_deep  # Set seed for deterministic result


class TrainingData(object):
    GENDERS = ['M', 'F', ]
    AGES = ['0-17', '18-24', '25-34', '35-44', '45-49', '50-55', '56+', ]

    def __init__(self, config):
        self.checkpoints_dir = config['CHECKPOINTS_DIR']
        self.model_dir = config['MODEL_DIR']
        self.encoder_path = config['ENCODER_PATH']
        self.database = db_connections.get('portal')
        self.items, self.item_feat_shape, self.users, self.train, self.test = self.prepare_training_data()

    def prepare_training_data(self):
        review_tuples = []

        features_by_movie_id = self._get_all_movies_features_map()
        features_by_user_id = self._get_all_user_features_map()

        data = self.database.execute('SELECT user_id, movie_id, rating FROM film_recommender_userreview', ())
        for review in data:
            features = features_by_movie_id.get(review[1], [])
            features += features_by_user_id.get(review[0], [])
            review_tuples.append((review[0], review[1], review[2], features))

        data = pd.DataFrame(data=review_tuples, columns=[USER_COL, ITEM_COL, RATING_COL, ITEM_FEAT_COL])

        features_encoder = sklearn.preprocessing.MultiLabelBinarizer(classes=self._get_all_features())
        data[ITEM_FEAT_COL] = features_encoder.fit_transform(data[ITEM_FEAT_COL]).tolist()

        with open(self.encoder_path, 'wb+') as f:
            pickle.dump(features_encoder, f)

        train, test = python_random_split(data, ratio=0.75, seed=SEED)

        items = data.drop_duplicates(ITEM_COL)[[ITEM_COL, ITEM_FEAT_COL]].reset_index(drop=True)
        item_feat_shape = len(items[ITEM_FEAT_COL][0])

        users = data.drop_duplicates(USER_COL)[[USER_COL]].reset_index(drop=True)

        return items, item_feat_shape, users, train, test

    def _get_all_features(self):
        sql = 'SELECT name FROM film_recommender_genre_translation WHERE language_code=\'en-us\'' \
              'UNION SELECT name FROM film_recommender_tag ' \
              'UNION SELECT name FROM account_occupation_translation WHERE language_code=\'en-us\' ' \
              'UNION SELECT DISTINCT country FROM account_customuser'

        features = [row[0] for row in self.database.execute(sql, ()) or []]
        features += self.AGES + self.GENDERS
        return features

    def _get_all_user_features_map(self):
        sql = 'SELECT cu.id, cu.age, cu.gender,cu.country,ot.name FROM account_customuser cu ' \
              'JOIN account_occupation oc ON  oc.id = cu.occupation_id ' \
              'JOIN account_occupation_translation ot ON ot.master_id = oc.id AND ot.language_code=\'en-us\''
        features_by_id = {row[0]: [feature for feature in row[1:] if feature] for row in self.database.execute(sql, ())}

        return features_by_id

    def _get_all_movies_features_map(self):
        features_by_movie_id = defaultdict(list)

        data = self.database.execute("""
            SELECT mv_gr.movie_id, tr.name FROM film_recommender_genre as genre 
            JOIN film_recommender_movie_genres as mv_gr ON genre.id = mv_gr.genre_id
            JOIN film_recommender_genre_translation as tr ON tr.master_id = genre.id AND tr.language_code='en-us'
        UNION
            SELECT mv_tg.movie_id, tag.name FROM film_recommender_tag as tag 
            JOIN film_recommender_movie_tags as mv_tg ON tag.id = mv_tg.tag_id
        """, ())

        for row in data:
            features_by_movie_id[row[0]].append(row[1])

        return features_by_movie_id


class ModelTrainer(object):
    #### Hyperparameters
    MODEL_TYPE = "wide_deep"
    STEPS = 20000  # Number of batches to train
    BATCH_SIZE = 32

    # Wide (linear) model hyperparameters
    LINEAR_OPTIMIZER = "adagrad"
    LINEAR_OPTIMIZER_LR = 0.0621  # Learning rate
    LINEAR_L1_REG = 0.0  # Regularization rate for FtrlOptimizer
    LINEAR_L2_REG = 0.0
    LINEAR_MOMENTUM = 0.0  # Momentum for MomentumOptimizer or RMSPropOptimizer

    # DNN model hyperparameters
    DNN_OPTIMIZER = "adadelta"
    DNN_OPTIMIZER_LR = 0.1
    DNN_L1_REG = 0.0  # Regularization rate for FtrlOptimizer
    DNN_L2_REG = 0.0
    DNN_MOMENTUM = 0.0  # Momentum for MomentumOptimizer or RMSPropOptimizer

    DNN_HIDDEN_LAYER_1 = 0
    DNN_HIDDEN_LAYER_2 = 64
    DNN_HIDDEN_LAYER_3 = 128
    DNN_HIDDEN_LAYER_4 = 512
    DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4] if
                        h > 0]
    DNN_USER_DIM = 32  # User embedding feature dimension
    DNN_ITEM_DIM = 16  # Item embedding feature dimension
    DNN_DROPOUT = 0.8
    DNN_BATCH_NORM = 1

    SAVE_CHECKPOINT_STEPS = max(1, STEPS // 5)

    RANDOM_SEED = SEED

    def __init__(self, training_data):
        self.training_data = training_data

        for filename in os.listdir(self.training_data.checkpoints_dir):
            file_path = os.path.join(self.training_data.checkpoints_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

        self.wide_columns, self.deep_columns = wide_deep.build_feature_columns(
            users=training_data.users[USER_COL].values,
            items=training_data.items[ITEM_COL].values,
            user_col=USER_COL,
            item_col=ITEM_COL,
            item_feat_col=ITEM_FEAT_COL,
            crossed_feat_dim=1000,
            user_dim=self.DNN_USER_DIM,
            item_dim=self.DNN_ITEM_DIM,
            item_feat_shape=training_data.item_feat_shape,
            model_type=self.MODEL_TYPE,
        )

        self.model = wide_deep.build_model(
            model_dir=self.training_data.checkpoints_dir,
            wide_columns=self.wide_columns,
            deep_columns=self.deep_columns,
            linear_optimizer=tf_utils.build_optimizer(self.LINEAR_OPTIMIZER, self.LINEAR_OPTIMIZER_LR, **{
                'l1_regularization_strength': self.LINEAR_L1_REG,
                'l2_regularization_strength': self.LINEAR_L2_REG,
                'momentum': self.LINEAR_MOMENTUM,
            }),
            dnn_optimizer=tf_utils.build_optimizer(self.DNN_OPTIMIZER, self.DNN_OPTIMIZER_LR, **{
                'l1_regularization_strength': self.DNN_L1_REG,
                'l2_regularization_strength': self.DNN_L2_REG,
                'momentum': self.DNN_MOMENTUM,
            }),
            dnn_hidden_units=self.DNN_HIDDEN_UNITS,
            dnn_dropout=self.DNN_DROPOUT,
            dnn_batch_norm=(self.DNN_BATCH_NORM == 1),
            log_every_n_iter=max(1, self.STEPS // 10),  # log 10 times
            save_checkpoints_steps=self.SAVE_CHECKPOINT_STEPS,
            seed=self.RANDOM_SEED
        )

        self.train_fn = tf_utils.pandas_input_fn(
            df=training_data.train,
            y_col=RATING_COL,
            batch_size=self.BATCH_SIZE,
            num_epochs=None,
            shuffle=True,
            seed=self.RANDOM_SEED,
        )

    def train(self):
        try:
            self.model.train(
                input_fn=self.train_fn,
                steps=self.STEPS
            )
        except tf.train.NanLossDuringTrainingError:
            import warnings
            warnings.warn(
                "Training stopped with NanLossDuringTrainingError. "
                "Try other optimizers, smaller batch size and/or smaller learning rate."
            )

        export_path = tf_utils.export_model(
            model=self.model,
            train_input_fn=self.train_fn,
            eval_input_fn=tf_utils.pandas_input_fn(df=self.training_data.test, y_col=RATING_COL),
            tf_feat_cols=self.wide_columns + self.deep_columns,
            base_dir=self.training_data.model_dir
        )


def train(config):
    training_data = TrainingData(config)
    trainer = ModelTrainer(training_data)
    trainer.train()
