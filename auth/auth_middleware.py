from functools import wraps
import jwt
from flask import request, abort
from flask import current_app

from extensions.extensions import db_connections


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        if not token:
            return {
                "message": "Authentication Token is missing!",
                "data": None,
                "error": "Unauthorized"
            }, 401
        try:
            data = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])

            db_connection = db_connections.get()
            current_user = db_connection.execute(
                """SELECT approved FROM account_serviceuser WHERE id = %s """, (data["user_id"],)
            )

            if not current_user:
                return {"message": "Invalid Authentication token!", "data": None, "error": "Unauthorized"}, 401

            if not current_user[0][0]:
                abort(403)

        except Exception as e:
            return {"message": "Something went wrong", "data": None, "error": str(e)}, 500

        return f(*args, **kwargs)

    return decorated
