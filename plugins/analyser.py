import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Type
import traceback
import sys
import importlib
from numpy import require

import numpy.typing as npt

from analyser.plugins.plugin import Plugin, Manager
from analyser.utils import convert_name
from analyser.data import PluginData, VideoData, AudioData, ImageData
from analyser import analyser_pb2


class AnalyserPluginCallback:
    def update(self, **kwargs):
        pass


class AnalyserProgressCallback(AnalyserPluginCallback):
    def __init__(self, shared_memory) -> None:
        self.shared_memory = shared_memory

    def update(self, progress=0.0, **kwargs):
        self.shared_memory["progress"] = progress


class AnalyserPlugin(Plugin):
    @classmethod
    def __init_subclass__(
        cls,
        parameters: Dict[str, Any] = None,
        requires: Dict[str, Type[PluginData]] = None,
        provides: Dict[str, Type[PluginData]] = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._requires = requires
        cls._provides = provides
        cls._parameters = parameters
        cls._name = convert_name(cls.__name__)

    def __init__(self, config=None):
        self._config = self._default_config
        if config is not None:
            self._config.update(config)

    @classmethod
    @property
    def requires(cls):
        return cls.requires

    @classmethod
    @property
    def provides(cls):
        return cls.provides

    @classmethod
    def update_callbacks(cls, callbacks, **kwargs):
        if callbacks is None or not isinstance(callbacks, (list, set)):
            return
        for x in callbacks:
            x.update(**kwargs)

    @classmethod
    def serialize_class(cls):
        result = analyser_pb2.PluginInfo()
        result.name = cls._name
        result.version = cls._version

        for k, v in cls._parameters.items():
            p = result.parameters.add()
            p.name = k
            p.default = f"{v}"
            if isinstance(v, str):
                p.type = analyser_pb2.STRING_TYPE
            elif isinstance(v, int):
                p.type = analyser_pb2.INT_TYPE
            elif isinstance(v, float):
                p.type = analyser_pb2.FLOAT_TYPE

        for k, v in cls._requires.items():
            r = result.requires.add()
            r.name = k
            if v == VideoData:
                r.type = analyser_pb2.VIDEO_DATA
            elif v == ImageData:
                r.type = analyser_pb2.IMAGE_DATA
            else:
                pass

        for k, v in cls._provides.items():
            r = result.provides.add()
            r.name = k
            if v == VideoData:
                r.type = analyser_pb2.VIDEO_DATA
            elif v == ImageData:
                r.type = analyser_pb2.IMAGE_DATA

        return result

    def __call__(
        self,
        inputs: Dict[str, PluginData],
        parameters: Dict[str, Any] = None,
        callbacks: List[AnalyserPluginCallback] = None,
    ) -> Dict[str, PluginData]:
        input_parameters = self._parameters
        if parameters is not None:
            input_parameters.update(parameters)
        logging.info(f"[Plugin] {self._name} starting")
        try:
            result = self.call(inputs, input_parameters, callbacks=callbacks)

        except Exception as e:
            logging.error(f"[Plugin] {self._name} {repr(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()

            traceback.print_exception(
                exc_type,
                exc_value,
                exc_traceback,
                limit=2,
                file=sys.stdout,
            )
            return {}
        logging.info(f"[Plugin] {self._name} done")
        return result


class AnalyserPluginManager(Manager):
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

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysers")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("analyser.plugins.analysers.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def __call__(self, plugin, inputs, parameters=None, callbacks=None):

        run_id = uuid.uuid4().hex[:4]
        if plugin not in self._plugins:
            return None

        plugin_to_run = None
        for plugin_candidate in self.plugin_list:
            if plugin_candidate.get("plugin").name == plugin:
                plugin_to_run = plugin_candidate["plugin"]
        if plugin_to_run is None:
            logging.error(f"[AnalyserPluginManager] {run_id} plugin: {plugin} not found")
            return None

        logging.info(f"[AnalyserPluginManager] {run_id} plugin: {plugin_to_run}")
        logging.info(f"[AnalyserPluginManager] {run_id} data: {[{k:x.id} for k,x in inputs.items()]}")
        logging.info(f"[AnalyserPluginManager] {run_id} parameters: {parameters}")
        results = plugin_to_run(inputs, parameters, callbacks)
        logging.info(f"[AnalyserPluginManager] {run_id} results: {[{k:x.id} for k,x in results.items()]}")
        return results
