import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Type
from numpy import require

import numpy.typing as npt

from analyser.utils import convert_name
from analyser.data import PluginData, VideoData, AudioData, ImageData
from analyser import analyser_pb2


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
                print("VIDEO_DATA")
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

    def __call__(self, inputs: Dict[str, PluginData]) -> Dict[str, PluginData]:
        return self.call(inputs)
