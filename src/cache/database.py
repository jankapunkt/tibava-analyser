import importlib
import os
import re
import sys
import logging
import uuid

# from typing import Union
from typing import Dict
from analyser.plugins.plugin import Plugin

from packaging import version
from analyser.plugins.manager import PluginManager


class Database(Plugin):
    def __init__(self, config=None):
        super().__init__(config)


class DatabaseManager(PluginManager):
    _plugins = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.find()
        self.plugin_list = self.init_plugins()

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._plugins[name] = plugin
            return plugin

        return export_helper

    def plugins(self):
        return self._plugins

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("analyser.plugins.databases.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def __call__(self, database_name):

        run_id = uuid.uuid4().hex[:4]
        if database_name not in self._plugins:
            return None

        plugin_to_run = None
        for plugin_candidate in self.plugin_list:
            if plugin_candidate.get("plugin").name == database_name:
                plugin_to_run = plugin_candidate["plugin"]
        if plugin_to_run is None:
            logging.error(f"[DatabaseManager] {run_id} plugin: {database_name} not found")
            return None

        return plugin_to_run
