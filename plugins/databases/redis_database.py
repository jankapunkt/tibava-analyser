from typing import Any, List, Iterator

import redis
import msgpack

from analyser.plugins.database import Database, DatabaseManager

default_config = {"db": 0, "host": "analyser_redisai", "port": 6379, "tag": "data"}


class Batcher:
    def __init__(self, iterable, n=1):
        self.iterable = iterable
        self.n = n

    def __iter__(self):
        l = len(self.iterable)
        for ndx in range(0, l, self.n):
            yield self.iterable[ndx : min(ndx + self.n, l)]


@DatabaseManager.export("redis")
class RedisDatabase(Database, config=default_config, version="0.1"):
    def __init__(self, config=None):
        super().__init__(config)
        self.r = redis.Redis(host=self.config.get("host"), port=self.config.get("port"), db=self.config.get("db"))

    def set(self, id: str, data: Any) -> bool:
        packed = msgpack.packb(data)
        tag = self.config.get("tag")
        self.r.set(f"{tag}:{id}", packed)

    def delete(self, id: str) -> bool:
        tag = self.config.get("tag")
        return self.r.delete(f"{tag}:{id}")

    def get(self, id: str) -> Any:
        tag = self.config.get("tag")
        packed = self.r.get(f"{tag}:{id}")
        return msgpack.unpackb(packed)

    def keys(self) -> List[str]:
        tag = self.config.get("tag")
        start = len(f"{tag}:")
        keys = self.r.scan_iter(f"{tag}:*", 500)

        # print([x for x in Batcher(keys, 2)])
        return [key[start:].decode("utf-8") for key in keys]

    def __iter__(self) -> Iterator:

        tag = self.config.get("tag")
        start = len(f"{tag}:")
        keys = list(self.r.scan_iter(f"{tag}:*", 500))
        while len(keys) > 0:
            batch_keys = keys[:500]
            keys = keys[500:]

            values = self.r.mget(batch_keys)
            for k, v in zip(batch_keys, values):
                yield k[start:].decode("utf-8"), msgpack.unpackb(v)
