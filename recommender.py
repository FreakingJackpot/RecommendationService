import glob
import os
from collections import defaultdict

import cachetools.func
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
from recommenders.utils.tf_utils import pandas_input_fn_for_saved_model
from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_GENRE_COL as ITEM_FEAT_COL,
)

from extensions.extensions import db_connections

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class Predictor(object):
    update_model_rate = 70  # minutes

    def __init__(self, config):
        self.model_dir = config['MODEL_DIR']
        self.database = db_connections.get('portal')

        self._update_model()

    def _update_model(self):
        list_of_files = glob.glob(self.model_dir + '/*')
        self.latest_file = latest_file = max(list_of_files, key=os.path.getctime)

        self.model = tf.saved_model.load(latest_file + '/', tags=['serve', ])

        self.features = self._get_all_features()

    def _get_all_features(self):
        sql = 'SELECT name FROM film_recommender_genre' \
              'UNION SELECT name FROM film_recommender_tag'

        features = [row[0] for row in self.database.execute(sql, ())]
        features.append('unknown')
        return features

    def get_top_k(self, movie_ids, user_id, top_k):
        predictions = self.predict(movie_ids, user_id)

        movie_ids_with_ratings = []
        for prediction, movie_id in zip(predictions, movie_ids):
            movie_ids_with_ratings.append((movie_id, prediction))

        movie_ids_with_ratings.sort(key=lambda x: x[1], reverse=True)

        return movie_ids_with_ratings[:top_k]

    def predict(self, movie_ids, user_id):
        self._check_new_weights()
        data = self._prepare_data_from_movies(movie_ids, user_id)

        predictions = self.model.signatures["predict"](
            examples=pandas_input_fn_for_saved_model(
                df=data,
                feat_name_type={
                    USER_COL: int,
                    ITEM_COL: int,
                    ITEM_FEAT_COL: list
                }
            )()["inputs"]
        )

        return [float(prediction) * 2 for prediction in predictions['predictions']]

    def _prepare_data_from_movies(self, movie_ids, user_id):
        genres_encoder = sklearn.preprocessing.MultiLabelBinarizer(classes=self.features)
        genres_by_movie_id = self.__get_genres_for_movies(movie_ids)

        movies_to_predict = []

        for movie_id in movie_ids:
            genres = genres_by_movie_id.get(movie_id, ['unknown', ])
            movies_to_predict.append((user_id, movie_id, genres))

        data = pd.DataFrame(data=movies_to_predict, columns=[USER_COL, ITEM_COL, ITEM_FEAT_COL])

        data[ITEM_FEAT_COL] = genres_encoder.fit_transform(data[ITEM_FEAT_COL]).tolist()
        data.reset_index(drop=True, inplace=True)

        return data

    def __get_genres_for_movies(self, movie_ids):
        genres_by_movie_id = defaultdict(list)
        data = self.database.execute("""
        SELECT mv_gr.movie_id ,genre.name FROM film_recommender_genre as genre 
        JOIN film_recommender_movie_genres as mv_gr ON genre.id = mv_gr.genre_id
        WHERE mv_gr.movie_id IN %s
        """, (movie_ids,))

        for row in data:
            if row[1] in self.features:
                genres_by_movie_id[row[0]] = row[1]

        return genres_by_movie_id

    @cachetools.func.ttl_cache(maxsize=128, ttl=600)
    def _check_new_weights(self):
        list_of_files = glob.glob(self.model_dir + '/*')
        latest_file = max(list_of_files, key=os.path.getctime)

        if latest_file != self.latest_file:
            self._update_model()
