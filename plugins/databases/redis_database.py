import redis


from analyser.plugins.database import Database, DatabaseManager

default_config = {
    "db": 0,
    "host": "localhost",
    "port": 6379,
}


@DatabaseManager.export("redis")
class RedisDatabase(Database, config=default_config, version="0.1"):
    def init(self, config):
        r = redis.Redis(host="localhost", port=6379, db=0)

    def set(self, id: str, data: str) -> bool:
        pass

    def get(self, id: str) -> str:
        pass
