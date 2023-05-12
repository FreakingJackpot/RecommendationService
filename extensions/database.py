from contextlib import contextmanager

import psycopg2


class DatabaseConnections:
    def __init__(self):
        self.app = None
        self.connections = {}

    def init_app(self, app):
        self.app = app
        self._add_connections_from_config()

    def _add_connections_from_config(self):
        for database_name, info in self.app.config['DATABASES'].items():
            self.connections[database_name] = Database(**info)

    def close_connections(self):
        for connection in self.connections.values():
            connection.close_connection()

    def get(self, name='service'):
        return self.connections[name]


class Database:
    def __init__(self, host, name, user, password, life_time):
        self.host = host
        self.name = name
        self.user = user
        self.password = password

        self.max_age = life_time

        self.connection = None

        self._connect()

    def _connect(self):
        self.connection = psycopg2.connect(
            host=self.host,
            database=self.name,
            user=self.user,
            password=self.password
        )

    def close_connection(self):
        if self.connection and not self.connection.closed:
            self.connection.close()

    def reconnect(self):
        self.close_connection()
        self._connect()

    def execute(self, sql, params, retries=3):
        data = None

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                data = cursor.fetchall()
        except psycopg2.OperationalError:
            if self.connection.closed:
                try:
                    self.reconnect()
                except psycopg2.OperationalError:
                    pass

            if retries:
                self.execute(sql, params, retries - 1)

        return data

    def modify_data(self, sql, params):
        result = False

        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            self.connection.commit()
            result = True

        return result


@contextmanager
def db_cursor(host, database, user, password):
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    cursor = conn.cursor()

    yield cursor

    cursor.close()
    conn.close()
