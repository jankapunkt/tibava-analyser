import importlib
import os
import re
import sys
import logging

# from typing import Union

from analyser.plugins.plugin import Plugin

from packaging import version


class PluginManager:
    def __init__(self, configs=None):
        self.configs = configs
        if configs is None:
            self.configs = []

    def plugins(self):
        return {}

    def init_plugins(self, plugins=None, configs=None):
        if plugins is None:
            plugins = list(self.plugins().keys())

        # TODO add merge tools
        if configs is None:
            configs = self.configs

        plugin_list = []
        plugin_name_list = [x.lower() for x in plugins]

        for plugin_name, plugin_class in self.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue
            plugin_has_config = False
            plugin_config = {"params": {}}
            for x in configs:
                if x["plugin"].lower() == plugin_name.lower():
                    plugin_config.update(x)
                    plugin_has_config = True
            if not plugin_has_config:
                continue
            plugin = plugin_class(config=plugin_config["params"])
            plugin_list.append({"plugin": plugin, "plugin_cls": plugin_class, "config": plugin_config})
        return plugin_list


class AnalyserPluginManager(PluginManager):
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

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "analyser")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("analyser.plugins.analyser.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def __call__(self, plugin, inputs, parameters=None):
        if plugin not in self._plugins:
            return None

        plugin_to_run = None
        for plugin_candidate in self.plugin_list:

            if plugin_candidate.get("plugin").name == plugin:
                plugin_to_run = plugin_candidate["plugin"]
        if plugin_to_run is None:
            return None
        logging.debug(f"manager.py: {plugin_to_run}")
        logging.debug(f"manager.py: {inputs}")
        logging.debug(f"manager.py: {parameters}")
        results = plugin_to_run(inputs, parameters)
        return results


class VideoPlugin(Plugin):
    _type = "video"

    def __init__(self, **kwargs):
        super(VideoPlugin, self).__init__(**kwargs)

    def __call__(self, images):
        return self.call(images)
