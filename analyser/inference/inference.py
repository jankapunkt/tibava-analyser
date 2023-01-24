from enum import Enum
import uuid
import logging

from typing import Dict


def generate_id():
    return uuid.uuid4().hex


class Backend(Enum):
    PYTORCH = 1
    TENSORFLOW = 2
    ONNX = 3


class Device(Enum):
    CPU = 1
    GPU = 2


class InferenceServer:
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
