from __future__ import annotations
import os
import re
import logging
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Type, Iterator
import json


import numpy.typing as npt
import numpy as np

from analyser import analyser_pb2


@dataclass(kw_only=True, frozen=True)
class PluginData:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    last_access: datetime = field(default_factory=lambda: datetime.now())
    type: str = field(default="PluginData")

    def dumps(self):
        return {"id": self.id, "last_access": self.last_access.timestamp(), "type": self.type}

    @classmethod
    def loads(cls, data: dict):
        return PluginData(
            id=data.get("id"), last_access=datetime.fromtimestamp(data.get("last_access")), type=data.get("type")
        )
        # self.id = data.get("id")
        # self.last_access = datetime.fromtimestamp(data.get("last_access"))

    def save(self, data_dir: str):
        data_path = os.path.join(data_dir, f"{self.id}.json")
        with open(data_path, "w") as f:
            f.write(self.dumps())

    @classmethod
    def load(cls, data_dir: str, id: str, load_blob: bool = False) -> PluginData:
        if len(id) != 32:
            return None

        if not re.match(r"^[a-f0-9]{32}$", id):
            return None

        data_path = os.path.join(data_dir, f"{id}.json")

        with open(data_path, "r") as f:
            return cls.loads(f.read())


@dataclass(kw_only=True, frozen=True)
class VideoData(PluginData):
    path: str = None
    data_dir: str = None
    ext: str = None
    type: str = field(default="VideoData")

    def __post_init__(self):
        if not self.path:
            if self.data_dir and self.ext:
                object.__setattr__(self, "path", os.path.join(self.data_dir, f"{self.id}.{self.ext}"))

    def dumps(self):
        dump = super().dumps()
        return {**dump, "path": self.path, "ext": self.ext, "type": self.type}

    @classmethod
    def loads(cls, data):
        return VideoData(**super().loads(data).dumps(), path=data.get("path"), ext=data.get("ext"))


@dataclass(kw_only=True, frozen=True)
class ImageData(PluginData):
    path: str = None
    time: float = None
    ext: str = None


@dataclass(kw_only=True, frozen=True)
class ImagesData(PluginData):
    images: List[ImageData] = field(default_factory=list)


@dataclass(kw_only=True, frozen=True)
class ShotData(PluginData):
    start: float
    end: float


@dataclass(kw_only=True, frozen=True)
class ShotsData(PluginData):
    shots: List[ShotData] = field(default_factory=list)


@dataclass(kw_only=True, frozen=True)
class AudioData(PluginData):
    path: str = None
    data_dir: str = None
    ext: str = None
    type: str = field(default="AudioData")

    def __post_init__(self):
        if not self.path:
            if self.data_dir and self.ext:
                object.__setattr__(self, "path", os.path.join(self.data_dir, f"{self.id}.{self.ext}"))

    def dumps(self):
        dump = super().dumps()
        return {**dump, "path": self.path, "ext": self.ext, "type": self.type}

    @classmethod
    def loads(cls, data):
        return AudioData(**super().loads(data).dumps(), path=data.get("path"), ext=data.get("ext"))


@dataclass(kw_only=True, frozen=True)
class ScalarData(PluginData):
    y: npt.NDArray = field(default_factory=np.ndarray)
    time: List[float] = field(default_factory=list)

    def dumps(self):
        dump = super().dumps()
        return {**dump, "path": self.path, "ext": self.ext, "type": self.type}

    def load(self, data):
        super().load(data)

    def save_blob(self, path):
        pass

    def load_blob(self, path):
        pass


@dataclass(kw_only=True, frozen=True)
class HistData(PluginData):
    y: npt.NDArray = field(default_factory=np.ndarray)
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
                ext = firstpkg.ext
            else:
                ext = "mp4"
            if self.data_dir:
                data.path = os.path.join(self.data_dir, f"{data.id}.{data.ext}")

            data = VideoData(ext=ext, data_dir=self.data_dir)

            with open(data.path, "wb") as f:
                f.write(firstpkg.data_encoded)  # write first package
                for x in datastream:
                    f.write(x.data_encoded)

        if data is not None:
            with open(os.path.join(self.data_dir, f"{data.id}.json"), "w") as f:
                f.write(json.dumps(data.dumps(), indent=2))

        return data

    def dump_to_stream(self, data: PluginData):
        print("#####")
        print(data, flush=True)
        if data.type == "VideoData":

            chunk_size = 1024

            with open(data.path, "rb") as bytestream:
                while True:
                    chunk = bytestream.read(chunk_size)
                    if not chunk:
                        break
                    yield {"type": analyser_pb2.VIDEO_DATA, "data_encoded": chunk}
        elif data.type == "AudioData":
            print(f"Audiostream {data}", flush=True)
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
        print(data_raw, flush=True)
        if data_raw.get("type") == "VideoData":
            data = VideoData.loads(data_raw)
        elif data_raw.get("type") == "AudioData":
            data = AudioData.loads(data_raw)
        else:
            logging.error(f"[DataManager::load] unknow type {data_raw['type']}")
            return None

        return data

    def save(self, data):
        with open(os.path.join(self.data_dir, f"{data.id}.json"), "w") as f:
            f.write(json.dumps(data.dumps(), indent=2))
