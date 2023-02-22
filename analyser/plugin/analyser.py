import logging
import os
import re
import uuid
from typing import Dict, List, Any, Type, Callable
import traceback
import sys
import importlib
import time

from analyser.utils import convert_name
from analyser.utils.plugin import Plugin, Manager
from analyser.data import Data, DataManager
from analyser import analyser_pb2
from analyser.plugin.callback import AnalyserPluginCallback
from analyser.utils import get_hash_for_plugin
from analyser.cache import Cache


class AnalyserPlugin(Plugin):
    @classmethod
    def __init_subclass__(
        cls,
        parameters: Dict[str, Any] = None,
        requires: Dict[str, Type[Data]] = None,
        provides: Dict[str, Type[Data]] = None,
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
        return cls._requires

    @classmethod
    @property
    def provides(cls):
        return cls._provides

    @classmethod
    def update_callbacks(cls, callbacks: List[AnalyserPluginCallback], **kwargs):
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
            elif isinstance(v, bool):
                p.type = analyser_pb2.BOOL_TYPE

        for k, v in cls._requires.items():
            r = result.requires.add()
            r.name = k
            # if v == VideoData:
            #     r.type = analyser_pb2.VIDEO_DATA
            # elif v == ImageData:
            #     r.type = analyser_pb2.IMAGE_DATA
            # else:
            #     pass

        for k, v in cls._provides.items():
            r = result.provides.add()
            r.name = k
            # if v == VideoData:
            #     r.type = analyser_pb2.VIDEO_DATA
            # elif v == ImageData:
            #     r.type = analyser_pb2.IMAGE_DATA

        return result

    def __call__(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        input_parameters = self._parameters
        if parameters is not None:
            input_parameters.update(parameters)
        logging.info(f"[Plugin] {self._name} starting")
        try:
            result = self.call(inputs, data_manager, input_parameters, callbacks=callbacks)

        except Exception as e:
            raise e
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

    def __init__(self, cache: Cache = None, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
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

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plugins")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("analyser.plugin.plugins.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def __call__(
        self,
        plugin: str,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ):
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

        cached = False
        if self.cache:
            results = {}
            logging.info(f"[AnalyserPluginManager] Cache {plugin_to_run}")
            logging.info(f"[AnalyserPluginManager] Cache {plugin_to_run.provides}")
            logging.info(f"[AnalyserPluginManager] Cache {plugin_to_run.requires}")

            cached = True
            for output in plugin_to_run.provides:
                result_hash = get_hash_for_plugin(
                    plugin=plugin,
                    output=output,
                    inputs=[x.id for _, x in inputs.items()],
                    parameters=parameters,
                    version=plugin_to_run.version,
                    config=plugin_to_run.config,
                )

                logging.info(f"[AnalyserPluginManager] Cache {result_hash}")
                cached_data = self.cache.get(result_hash)
                if cached_data is None:
                    cached = False
                    break

                logging.info(f"[AnalyserPluginManager] Cache get {result_hash} {cached_data}")
                results[output] = data_manager.load(cached_data.get("data_id"))

        if not cached:
            logging.info(f"[AnalyserPluginManager] {run_id} plugin: {plugin_to_run}")
            logging.info(f"[AnalyserPluginManager] {run_id} data: {[{k:x.id} for k,x in inputs.items()]}")
            logging.info(f"[AnalyserPluginManager] {run_id} parameters: {parameters}")
            results = plugin_to_run(inputs, data_manager, parameters, callbacks)
            logging.info(f"[AnalyserPluginManager] {run_id} results: {[{k:x.id} for k,x in results.items()]}")

        if self.cache:
            for output, data in results.items():
                result_hash = get_hash_for_plugin(
                    plugin=plugin,
                    output=output,
                    inputs=[x.id for _, x in inputs.items()],
                    parameters=parameters,
                    version=plugin_to_run.version,
                    config=plugin_to_run.config,
                )
                logging.info(f"[AnalyserPluginManager] Cache set {result_hash} {data.id}")

                self.cache.set(result_hash, {"data_id": data.id, "time": time.time(), "type": "plugin_result"})

        return results
