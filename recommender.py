import glob
import os
import pickle
from collections import defaultdict

import cachetools.func
import redis
import pandas as pd
import tensorflow as tf
from recommenders.utils.tf_utils import pandas_input_fn_for_saved_model
from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_GENRE_COL as ITEM_FEAT_COL,
)

from extensions.extensions import db_connections

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class PredictDataPreparer(object):
    cache_key_template = 'feats_{}'

    def __init__(self, database, encoder_path, redis_url):
        self.database = database
        self.cache = redis.Redis.from_url(redis_url, decode_responses=True)
        self.encoder_path = encoder_path
        self.features_encoder, self.features = None, None
        self.update()

    def update(self):
        with open(self.encoder_path, 'rb') as f:
            self.features_encoder = pickle.load(f)

        self.features = set(self.features_encoder.classes_)

    def prepare_data_from_user_and_movies(self, movie_ids, user_id):
        features_by_movie_id = self.__get_features_for_movies(movie_ids)

        movies_to_predict = []

        for movie_id in movie_ids:
            features = features_by_movie_id.get(movie_id, [])
            movies_to_predict.append((user_id, movie_id, features))

        data = pd.DataFrame(data=movies_to_predict, columns=[USER_COL, ITEM_COL, ITEM_FEAT_COL])

        data[ITEM_FEAT_COL] = self.features_encoder.fit_transform(data[ITEM_FEAT_COL]).tolist()
        data.reset_index(drop=True, inplace=True)

        return data

    def __get_features_for_movies(self, movie_ids):
        genres_by_movie_id = defaultdict(list)

        movies_keys = [self.cache_key_template.format(id_) for id_ in movie_ids]
        ids_to_update = []
        for key in movies_keys:
            features = self.cache.lrange(key, 0, -1)
            id_ = int(key.split('_')[1])
            if features:
                genres_by_movie_id[id_] = features
            else:
                ids_to_update.append(id_)

        if ids_to_update:
            data = self.database.execute("""
            SELECT mv_gr.movie_id ,tr.name FROM film_recommender_genre as genre 
            JOIN film_recommender_movie_genres as mv_gr ON genre.id = mv_gr.genre_id
            JOIN film_recommender_genre_translation as tr ON tr.master_id = genre.id AND tr.language_code='en-us'
            WHERE mv_gr.movie_id IN %s
            UNION 
            SELECT mv_tg.movie_id ,tr.name FROM film_recommender_tag as tag 
            JOIN film_recommender_movie_tags as mv_tg ON tag.id = mv_tg.tag_id
            JOIN film_recommender_tag_translation as tr ON tr.master_id = tag.id AND tr.language_code='en-us'
            WHERE mv_tg.movie_id IN %s
            """, (tuple(ids_to_update), tuple(ids_to_update)))

            for row in data:
                if row[1] in self.features:
                    genres_by_movie_id[row[0]].append(row[1])

            for id_ in ids_to_update:
                self.cache.lpush(self.cache_key_template.format(id_), *genres_by_movie_id[id_])

        return genres_by_movie_id


class Predictor(object):
    update_model_rate = 70  # minutes

    def __init__(self, config):
        self.model_dir = config['MODEL_DIR']
        self.data_preparer = PredictDataPreparer(db_connections.get('portal'), config['ENCODER_PATH'],
                                                 config['REDIS_URL'])

        self._update_model()

    def _update_model(self):
        list_of_files = glob.glob(self.model_dir + '/*')
        self.latest_file = max(list_of_files, key=os.path.getctime)

        self.model = tf.saved_model.load(self.latest_file + '/', tags=['serve', ])

        self.data_preparer.update()

    def predict(self, movie_ids, user_id):
        self._check_new_weights()
        data = self.data_preparer.prepare_data_from_user_and_movies(movie_ids, user_id)

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

        return [
            {'movie_id': movie_id, 'rating': predict if (predict := float(prediction)) <= 5 else 5.0} for
            movie_id, prediction
            in
            zip(movie_ids, predictions['predictions'])
        ]

    @cachetools.func.ttl_cache(maxsize=128, ttl=600)
    def _check_new_weights(self):
        list_of_files = glob.glob(self.model_dir + '/*')
        latest_file = max(list_of_files, key=os.path.getctime)

        if latest_file != self.latest_file:
            self._update_model()
