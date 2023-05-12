from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash

from extensions.extensions import db_connections


class User:
    @classmethod
    def create(cls, username, password):
        password = generate_password_hash(password)
        data = db_connections.get().modify_data("INSERT INTO users (username,password) VALUES (%s,%s)",
                                                (username, password,))
        return data

    @classmethod
    def login(cls, username, password):
        connection = db_connections.get()
        data = connection.execute("SELECT id, password FROM users WHERE username = %s", (username,))

        if data and check_password_hash(data[0][1], password):
            return data[0][0]
