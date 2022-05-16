import os
import re
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Type, Iterator

import numpy.typing as npt
import numpy as np
import json

from analyser import analyser_pb2
from datetime import datetime


@dataclass
class PluginData:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    last_access: datetime = field(default_factory=lambda: datetime.now())
    type: str = field(default="PluginData", init=False)

    def dump(self):
        return {"id": self.id, "last_access": self.last_access.timestamp(), "type": self.type}

    def load(self, data):
        self.id = data.get("id")
        self.last_access = datetime.fromtimestamp(data.get("last_access"))


@dataclass
class VideoData(PluginData):
    path: str = None
    ext: str = None
    type: str = field(default="VideoData", init=False)

    def dump(self):
        dump = super().dump()
        return {**dump, "path": self.path, "ext": self.ext, "type": self.type}

    def load(self, data):
        super().load(data)
        self.path = data.get("path")
        self.ext = data.get("ext")


@dataclass
class ImageData(PluginData):
    path: str = None
    time: float = None
    ext: str = None


@dataclass
class Shot:
    start: float
    end: float


@dataclass
class ShotsData(PluginData):
    shots: List[Shot] = field(default_factory=list)


@dataclass
class AudioData(PluginData):
    path: str = None
    ext: str = None
    type: str = field(default="AudioData", init=False)

    def dump(self):
        dump = super().dump()
        return {**dump, "path": self.path, "ext": self.ext, "type": self.type}

    def load(self, data):
        super().load(data)
        self.path = data.get("path")
        self.ext = data.get("ext")


@dataclass
class ScalarData(PluginData):
    x: npt.NDArray = field(default_factory=np.ndarray)
    time: List[float] = field(default_factory=list)


@dataclass
class HistData(PluginData):
    x: npt.NDArray = field(default_factory=np.ndarray)
    time: List[float] = field(default_factory=list)


def data_from_proto_stream(proto, data_dir=None):
    if proto.type == analyser_pb2.VIDEO_DATA:
        data = VideoData()
        if hasattr(proto, "ext"):
            data.ext = proto.ext
        if data_dir:
            data.path = os.path.join(data_dir)

        return data


def data_from_proto(proto, data_dir=None):
    if proto.type == analyser_pb2.VIDEO_DATA:
        data = VideoData()
        if hasattr(proto, "ext"):
            data.ext = proto.ext
        if data_dir:
            data.path = os.path.join(data_dir)

        return data


class DataManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_from_stream(self, data: Iterator[Any]) -> PluginData:

        datastream = iter(data)
        firstpkg = next(datastream)
        data = None
        if firstpkg.type == analyser_pb2.VIDEO_DATA:
            data = VideoData()
            if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
                data.ext = firstpkg.ext
            else:
                data.ext = "mp4"
            if self.data_dir:
                data.path = os.path.join(self.data_dir, f"{data.id}.{data.ext}")

            with open(data.path, "wb") as f:
                f.write(firstpkg.data_encoded)  # write first package
                for x in datastream:
                    f.write(x.data_encoded)

        if data is not None:
            with open(os.path.join(self.data_dir, f"{data.id}.json"), "w") as f:
                f.write(json.dumps(data.dump(), indent=2))

        return data

    def dump_to_stream(self, data: PluginData):

        if data.type == "VideoData":

            chunk_size = 1024

            with open(data.path, "rb") as bytestream:
                while True:
                    chunk = bytestream.read(chunk_size)
                    if not chunk:
                        break
                    yield {"type": analyser_pb2.VIDEO_DATA, "data_encoded": chunk}
        elif data.type == "AudioData":

            chunk_size = 1024
            with open(data.path, "rb") as bytestream:
                while True:
                    chunk = bytestream.read(chunk_size)
                    if not chunk:
                        break
                    yield {"type": analyser_pb2.AUDIO_DATA, "data_encoded": chunk}

    def load(self, data_id):
        if len(data_id) != 32:
            return None

        if not re.match(r"^[a-f0-9]{32}$", data_id):
            return None

        data_path = os.path.join(self.data_dir, f"{data_id}.json")

        if not os.path.exists(data_path):
            return None

        with open(data_path, "r") as f:
            data_raw = json.load(f)

        if data_raw["type"] == "VideoData":
            data = VideoData()
            data.load(data_raw)

        elif data_raw["type"] == "AudioData":
            data = AudioData()
            data.load(data_raw)
        else:
            logging.error(f"[DataManager::load] unknow type {data_raw['type']}")
            return None

        return data

    def save(self, data):
        with open(os.path.join(self.data_dir, f"{data.id}.json"), "w") as f:
            f.write(json.dumps(data.dump(), indent=2))
