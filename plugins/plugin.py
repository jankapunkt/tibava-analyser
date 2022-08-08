import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Type
import traceback
import sys
from numpy import require

import numpy.typing as npt

from analyser.utils import convert_name
from analyser.data import PluginData, VideoData, AudioData, ImageData
from analyser import analyser_pb2


class PluginCallback:
    def update(self, **kwargs):
        pass


class ProgressCallback(PluginCallback):
    def __init__(self, shared_memory) -> None:
        self.shared_memory = shared_memory

    def update(self, progress=0.0, **kwargs):
        self.shared_memory["progress"] = progress


class Plugin:
    @classmethod
    def __init_subclass__(
        cls,
        config: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None,
        version: str = None,
        requires: Dict[str, Type[PluginData]] = None,
        provides: Dict[str, Type[PluginData]] = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._default_config = config
        cls._version = version
        cls._requires = requires
        cls._provides = provides
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
                print(v)

            print(f"WTF {r} {r.type == analyser_pb2.VIDEO_DATA}")

        for k, v in cls._provides.items():
            r = result.provides.add()
            r.name = k
            if v == VideoData:
                r.type = analyser_pb2.VIDEO_DATA
            elif v == ImageData:
                r.type = analyser_pb2.IMAGE_DATA

        return result

    def __call__(
        self, inputs: Dict[str, PluginData], parameters: Dict[str, Any] = None, callbacks: List[PluginCallback] = None
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
