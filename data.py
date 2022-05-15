import os
import re
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

    def dump(self):
        print(self.last_access, flush=True)
        return {"id": self.id, "last_access": self.last_access.timestamp()}

    def load(self, data):
        self.id = data.get("id")
        self.last_access = datetime.fromtimestamp(data.get("last_access"))


@dataclass
class VideoData(PluginData):
    path: str = None
    ext: str = None

    def dump(self):
        dump = super().dump()
        return {**dump, "path": self.path, "ext": self.ext}

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

    def load_from_stream(self, data: Iterator[Any]):

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
                json.dumps(data.dump())

        return data

    def load(self, data_id):
        if len(data_id) != 32:
            return None

        if not re.match(r"^[a-f0-9]{32}$", data_id):
            return None

        
        print(data_id)
