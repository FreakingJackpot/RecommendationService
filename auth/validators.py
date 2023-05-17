from extensions.extensions import db_connections
from werkzeug.security import generate_password_hash, check_password_hash

required_data_fields = ['username', 'password']


def validate_user_data(user_data):
    for field in required_data_fields:
        value = user_data.get(field)
        if not value:
            return False

    return True


def validate_username_and_password(username, password):
    connection = db_connections.get()
    data = connection.execute("SELECT password FROM portal_serviceuser WHERE username = %s", (username,))

    if data:
        return check_password_hash(data[0][0], password)

    return False
