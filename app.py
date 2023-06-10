from flask import Flask, request, jsonify
import jwt
import yaml

from extensions.extensions import db_connections, celery_init_app
from auth.auth_middleware import token_required
from auth.validators import validate_user_data, validate_username_and_password
from models import User
from recommender import Predictor
from model_training import train

app = Flask(__name__)
app.config.from_pyfile('config.py')

db_connections.init_app(app)
celery = celery_init_app(app)

PREDICTOR = Predictor(app.config)

@app.route('/predictor/', methods=('GET',))
@token_required
def predict():
    args = request.args
    movie_ids = tuple(map(int, args.get('movie_ids', type=str).split(',')))
    user_id = args.get('user_id', type=int)
    predictions = PREDICTOR.predict(movie_ids, user_id)
    return jsonify({
        'predictions': predictions
    })


@app.route("/users/", methods=["POST"])
def add_user():
    try:
        user_data = request.json
        if not user_data:
            return {
                "message": "Please provide user details",
                "data": None,
                "error": "Bad request"
            }, 400

        is_validated = validate_user_data(user_data)
        if is_validated is not True:
            return dict(message='Invalid data', data=None, error=is_validated), 400

        user = User.create(**user_data)

        if not user:
            return {
                "message": "User already exists",
                "error": "Conflict",
                "data": None
            }, 409

        return {
            "message": "Successfully created new user",
            "data": user
        }, 201

    except Exception as e:
        return {
            "message": "Something went wrong",
            "error": str(e),
            "data": None
        }, 500


@app.route("/users/login", methods=["POST"])
def login():
    try:
        data = request.json

        if not data:
            return {
                "message": "Please provide user details",
                "data": None,
                "error": "Bad request"
            }, 400

        username, password = data.get('username'), data.get('password')

        is_validated = validate_username_and_password(username, password)
        if is_validated is not True:
            return dict(message='Invalid data', data=None, error=is_validated), 400

        user_id = User.login(username, password)

        try:
            # expire after 24 hrs
            token = jwt.encode({"user_id": user_id}, app.config["SECRET_KEY"], algorithm="HS256")
            return {
                'token': token,
            }

        except Exception as e:
            return {"error": "Something went wrong", "message": str(e)}, 500

    except Exception as e:
        return {"error": "Something went wrong", "message": str(e)}, 500


@app.route("/swagger/", methods=["GET"])
def swagger():
    with open('openapi.yml', 'r') as file:
        openapi = yaml.safe_load(file)

    return openapi


@app.errorhandler(404)
def forbidden(e):
    return jsonify({"message": "Endpoint Not Found", "error": str(e), "data": None}), 404


@celery.task
def train_model():
    train(app.config)


if __name__ == '__main__':
    try:
        app.run()
    except Exception as e:
        print(e)

    finally:
        db_connections.close_connections()
