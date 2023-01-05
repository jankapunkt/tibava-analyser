import logging
from typing import Dict

from analyser.utils.plugin import Plugin
from analyser.utils.plugin import Factory


class Cache(Plugin):
    def __init__(self, config=None):
        super().__init__(config)


class CacheManager(Factory):
    _plugins = {}

    @classmethod
    def export(cls, name: str):
        def export_helper(plugin):
            cls._plugins[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def build(cls, name: str, config: Dict = None):
        if name not in cls._plugins:
            logging.error(f"Unknown inference server: {name}")
            return None

        return cls._plugins[name](config)
