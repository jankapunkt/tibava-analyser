import importlib
import os
import re
import sys
import logging
import uuid


from packaging import version
from typing import Union, Dict, Any

from analyser.utils import convert_name


class Plugin:
    @classmethod
    def __init_subclass__(
        cls,
        config: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None,
        version: str = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._default_config = config
        cls._version = version
        cls._parameters = parameters
        cls._name = convert_name(cls.__name__)

    def __init__(self, config=None):
        self._config = self._default_config
        if config is not None:
            self._config.update(config)

    @property
    def config(self):
        return self._config

    @classmethod
    @property
    def default_config(cls):
        return cls._default_config

    @classmethod
    @property
    def version(cls):
        return cls._version

    @classmethod
    @property
    def name(cls):
        return cls._name


class Manager:
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
