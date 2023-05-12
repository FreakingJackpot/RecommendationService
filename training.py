import os
from collections import defaultdict

import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

from extensions.database import db_cursor

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
import recommenders.models.wide_deep.wide_deep_utils as wide_deep

RANDOM_SEED = SEED  # Set seed for deterministic result

#### Hyperparameters
MODEL_TYPE = "wide_deep"
STEPS = 50000  # Number of batches to train
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

DNN_HIDDEN_LAYER_1 = 0  # Set 0 to not use this layer
DNN_HIDDEN_LAYER_2 = 64  # Set 0 to not use this layer
DNN_HIDDEN_LAYER_3 = 128  # Set 0 to not use this layer
DNN_HIDDEN_LAYER_4 = 512  # Note, at least one layer should have nodes.
DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4] if
                    h > 0]
DNN_USER_DIM = 32  # User embedding feature dimension
DNN_ITEM_DIM = 16  # Item embedding feature dimension
DNN_DROPOUT = 0.8
DNN_BATCH_NORM = 1

SAVE_CHECKPOINT_STEPS = max(1, STEPS // 5)


class TrainingData(object):
    def __init__(self, config):
        self.checkpoints_dir = config['CHECKPOINTS_DIR']
        self.model_dir = config['MODEL_DIR']
        self.items, self.item_feat_shape, self.users, self.train, self.test = self.prepare_training_data(config)

    def prepare_training_data(self, config):
        review_tuples = []
        genres_encoder = sklearn.preprocessing.MultiLabelBinarizer()

        with db_cursor(config['DB_HOST'], config['DB_NAME'], config['DB_USERNAME'], config['DB_PASSWORD']) as cursor:
            genres_by_movie_id = self._get_all_genres_by_movie_id_map(cursor)

            cursor.execute('SELECT user_id, movie_id, rating FROM film_recommender_userreview')
            for review in cursor.fetchall():
                genres = genres_by_movie_id[review[1]] or ['unknown', ]
                review_tuples.append((review[0], review[1], review[2], genres))

        data = pd.DataFrame(data=review_tuples, columns=[USER_COL, ITEM_COL, RATING_COL, ITEM_FEAT_COL])

        data[ITEM_FEAT_COL] = genres_encoder.fit_transform(data[ITEM_FEAT_COL]).tolist()

        train, test = python_random_split(data, ratio=0.75, seed=SEED)

        items = data.drop_duplicates(ITEM_COL)[[ITEM_COL, ITEM_FEAT_COL]].reset_index(drop=True)
        item_feat_shape = len(items[ITEM_FEAT_COL][0])
        # Unique users in the dataset
        users = data.drop_duplicates(USER_COL)[[USER_COL]].reset_index(drop=True)

        return items, item_feat_shape, users, train, test

    @staticmethod
    def _get_all_genres_by_movie_id_map(cursor):
        genres_by_movie_id = defaultdict(list)

        cursor.execute("""
        SELECT mv_gr.movie_id ,genre.name FROM film_recommender_genre as genre 
        JOIN film_recommender_movie_genres as mv_gr ON genre.id = mv_gr.genre_id
        """)

        for row in cursor.fetchall():
            genres_by_movie_id[row[0]] = row[1]

        return


class ModelTrainer(object):
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
            user_dim=DNN_USER_DIM,
            item_dim=DNN_ITEM_DIM,
            item_feat_shape=training_data.item_feat_shape,
            model_type=MODEL_TYPE,
        )

        self.model = wide_deep.build_model(
            model_dir=self.training_data.checkpoints_dir,
            wide_columns=self.wide_columns,
            deep_columns=self.deep_columns,
            linear_optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR, **{
                'l1_regularization_strength': LINEAR_L1_REG,
                'l2_regularization_strength': LINEAR_L2_REG,
                'momentum': LINEAR_MOMENTUM,
            }),
            dnn_optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR, **{
                'l1_regularization_strength': DNN_L1_REG,
                'l2_regularization_strength': DNN_L2_REG,
                'momentum': DNN_MOMENTUM,
            }),
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_dropout=DNN_DROPOUT,
            dnn_batch_norm=(DNN_BATCH_NORM == 1),
            log_every_n_iter=max(1, STEPS // 10),  # log 10 times
            save_checkpoints_steps=SAVE_CHECKPOINT_STEPS,
            seed=RANDOM_SEED
        )

        self.train_fn = tf_utils.pandas_input_fn(
            df=training_data.train,
            y_col=RATING_COL,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=True,
            seed=RANDOM_SEED,
        )

    def train(self):
        try:
            self.model.train(
                input_fn=self.train_fn,
                steps=STEPS
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
