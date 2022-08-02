from __future__ import annotations
import os
import re
import logging
import uuid
import json
import tempfile
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Type, Iterator, Union


import msgpack
import msgpack_numpy as m

import numpy.typing as npt
import numpy as np

from analyser import analyser_pb2

from analyser.utils import ByteFIFO


def create_data_path(data_dir, data_id, file_ext):
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"{data_id}.{file_ext}")
    return data_path


def generate_id():
    return uuid.uuid4().hex


class DataManager:
    _data_name_lut = {}
    _data_enum_lut = {}

    def __init__(self, data_dir=None):
        if not data_dir:
            data_dir = tempfile.mkdtemp()
        self.data_dir = data_dir

    @classmethod
    def export(cls, name: str, enum_value: int):
        def export_helper(data):
            cls._data_name_lut[name] = data
            cls._data_enum_lut[enum_value] = data
            return data

        return export_helper

    @classmethod
    def _load_from_stream(cls, data_dir: str, data: Iterator[Any], save_meta=True) -> PluginData:
        logging.debug(f"data.py (load_from_stream): {data}")
        datastream = iter(data)
        firstpkg = next(datastream)

        def data_generator():
            yield firstpkg
            for x in datastream:
                yield x

        data = None
        if firstpkg.type not in cls._data_enum_lut:
            return None
        data = cls._data_enum_lut[firstpkg.type].load_from_stream(data_dir=data_dir, stream=data_generator())

        if save_meta and data is not None:
            with open(create_data_path(data_dir, data.id, "json"), "w") as f:
                f.write(json.dumps(data.dumps(), indent=2))

        return data

    def load_from_stream(self, data: Iterator[Any], save_meta=True) -> PluginData:
        return self._load_from_stream(self.data_dir, data, save_meta)

    def dump_to_stream(self, data: PluginData):
        return data.dump_to_stream()

    @classmethod
    def _load(self, data_dir: str, data_id: str) -> PluginData:
        data = PluginData.load(data_dir=data_dir, id=data_id, load_blob=False)
        if data.type not in self._data_name_lut:
            logging.error(f"[DataManager::load] unknow type {data.type}")
            return None

        return self._data_name_lut[data.type].load(data_dir=data_dir, id=data_id)

    def load(self, data_id: str) -> PluginData:
        return self._load(self.data_dir, data_id)

    def save(self, data):
        data.save(self.data_dir, save_blob=True)


@dataclass(kw_only=True, frozen=True)
class PluginData:
    id: str = field(default_factory=generate_id)
    last_access: datetime = field(default_factory=lambda: datetime.now())
    type: str = field(default="PluginData")
    path: str = None
    data_dir: str = None
    ext: str = None

    def __post_init__(self):
        if not self.path:
            if self.data_dir and self.ext:
                object.__setattr__(self, "path", create_data_path(self.data_dir, self.id, self.ext))

    def dumps(self):
        return {"id": self.id, "last_access": self.last_access.timestamp(), "type": self.type, "ext": self.ext}

    def save(self, data_dir: str, save_blob: bool = True) -> bool:
        logging.debug("[PluginData::save]")
        try:
            if not self.save_blob(data_dir):
                return False
            data_path = create_data_path(data_dir, self.id, "json")
            with open(data_path, "w") as f:
                f.write(json.dumps(self.dumps()))
        except Exception as e:
            logging.error(f"[PluginData::save] {e}")
            logging.error(f"[PluginData::save] {traceback.format_exc()}")
            logging.error(f"[PluginData::save] {traceback.print_stack()}")
            return False
        return True

    def save_blob(self, data_dir=None, path=None) -> bool:
        return True

    @classmethod
    def load(cls, data_dir: str, id: str, load_blob: bool = True) -> PluginData:
        logging.debug(f"[PluginData::load] {id}")
        if len(id) != 32:
            return None

        if not re.match(r"^[a-f0-9]{32}$", id):
            return None

        data_path = create_data_path(data_dir, id, "json")

        data = {}
        with open(data_path, "r") as f:
            data = {**json.load(f), "data_dir": data_dir}

        data_args = cls.load_args(data)
        blob_args = dict()
        if load_blob:
            blob_args = cls.load_blob_args(data_args)

        return cls(**data_args, **blob_args)

    @classmethod
    def load_args(cls, data: dict) -> dict:
        return dict(
            id=data.get("id"),
            last_access=datetime.fromtimestamp(data.get("last_access")),
            type=data.get("type"),
            data_dir=data.get("data_dir"),
        )

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        return {}

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        return cls(data_dir=data_dir)


@DataManager.export("VideoData", analyser_pb2.VIDEO_DATA)
@dataclass(kw_only=True, frozen=True)
class VideoData(PluginData):
    path: str = None
    data_dir: str = None
    ext: str = None
    type: str = field(default="VideoData")

    def dumps(self):
        dump = super().dumps()
        return {**dump, "path": self.path, "ext": self.ext, "type": self.type}

    @classmethod
    def load_args(cls, data: dict):
        data_dict = super().load_args(data)
        return dict(**data_dict, path=data.get("path"), ext=data.get("ext"))

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        return {}

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "mp4"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        return cls(id=data_id, ext=ext, data_dir=data_dir)

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        with open(self.path, "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.VIDEO_DATA, "data_encoded": chunk, "ext": self.ext}


@dataclass(kw_only=True, frozen=True)
class ImageData(PluginData):
    time: float = None
    delta_time: float = field(default=None)
    ext: str = field(default="jpg")


@DataManager.export("ImagesData", analyser_pb2.IMAGES_DATA)
@dataclass(kw_only=True, frozen=True)
class ImagesData(PluginData):
    images: List[ImageData] = field(default_factory=list)
    ext: str = field(default="msg")
    type: str = field(default="ImagesData")

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[ImagesData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                # TODO use dump
                f.write(
                    msgpack.packb(
                        {
                            "images": [
                                {"time": image.time, "delta_time": image.delta_time, "ext": image.ext, "id": image.id}
                                for image in self.images
                            ]
                        }
                    )
                )
        except Exception as e:
            logging.error(f"ImagesData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[ImagesData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            packdata = msgpack.unpackb(f.read())
            dictdata = {
                "images": [
                    ImageData(
                        time=x["time"],
                        delta_time=x["delta_time"],
                        id=x["id"],
                        ext=x["ext"],
                        data_dir=data.get("data_dir"),
                    )
                    for x in packdata["images"]
                ]
            }
        return dictdata

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        unpacker = msgpack.Unpacker()
        unpacker.feed(firstpkg.data_encoded)
        images = []
        for x in stream:
            unpacker.feed(x.data_encoded)
            for image in unpacker:
                image_id = generate_id()
                image_path = create_data_path(data_dir, image_id, image.get("ext"))
                with open(image_path, "wb") as f:
                    f.write(image.get("image"))
                images.append(
                    ImageData(
                        data_dir=data_dir,
                        id=image_id,
                        ext=image.get("ext"),
                        time=image.get("time"),
                        delta_time=image.get("delta_time"),
                    )
                )

        data = cls(images=images)
        data.save_blob(data_dir=data_dir)
        return data

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        buffer = ByteFIFO()
        for image in self.images:
            with open(create_data_path(self.data_dir, image.id, image.ext), "rb") as f:
                image_raw = f.read()
            dump = msgpack.packb(
                {"time": image.time, "delta_time": image.delta_time, "ext": image.ext, "image": image_raw}
            )
            buffer.write(dump)

            while len(buffer) > chunk_size:
                chunk = buffer.read(chunk_size)
                # if not chunk:
                #     break

                yield {"type": analyser_pb2.IMAGES_DATA, "data_encoded": chunk, "ext": self.ext}

        chunk = buffer.read(chunk_size)
        if chunk:
            yield {"type": analyser_pb2.IMAGES_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        return {
            "images": [
                {"time": image.time, "delta_time": image.delta_time, "ext": image.ext, "id": image.id}
                for image in self.images
            ]
        }


@dataclass(kw_only=True, frozen=True)
class Shot:
    start: float
    end: float


@DataManager.export("ShotsData", analyser_pb2.SHOTS_DATA)
@dataclass(kw_only=True, frozen=True)
class ShotsData(PluginData):
    type: str = field(default="ShotsData")
    ext: str = field(default="msg")
    shots: List[Shot] = field(default_factory=list)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[ShotsData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                # TODO use dump
                f.write(msgpack.packb({"shots": [{"start": x.start, "end": x.end} for x in self.shots]}))
        except Exception as e:
            logging.error(f"ScalarData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[ShotsData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read())
            data = {"shots": [Shot(start=x["start"], end=x["end"]) for x in data["shots"]]}
        return data

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.SHOTS_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        return {"shots": [{"start": x.start, "end": x.end} for x in self.shots]}


@DataManager.export("AudioData", analyser_pb2.AUDIO_DATA)
@dataclass(kw_only=True, frozen=True)
class AudioData(PluginData):
    type: str = field(default="AudioData")

    def dumps(self):
        data_dict = super().dumps()
        return {**data_dict, "path": self.path, "ext": self.ext, "type": self.type}

    @classmethod
    def load_args(cls, data: dict):
        data_dict = super().load_args(data)
        return dict(**data_dict, path=data.get("path"), ext=data.get("ext"))

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        return {}

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "mp3"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        return cls(id=data_id, ext=ext, data_dir=data_dir)

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        with open(self.path, "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.AUDIO_DATA, "data_encoded": chunk, "ext": self.ext}


@DataManager.export("HistData", analyser_pb2.HIST_DATA)
@dataclass(kw_only=True, frozen=True)
class HistData(PluginData):
    type: str = field(default="HistData")
    ext: str = field(default="msg")
    hist: npt.NDArray = field()
    time: List[float] = field(default_factory=list)
    delta_time: float = field(default=None)
    name: str = field(default=None)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[ScalarData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {"hist": self.hist, "time": self.time, "delta_time": self.delta_time}, default=m.encode
                    )
                )
        except Exception as e:
            logging.error(f"ScalarData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[ScalarData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read(), object_hook=m.decode)
        return data

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.HIST_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        return {"hist": self.hist.tolist(), "time": self.time, "delta_time": self.delta_time}


@dataclass(kw_only=True, frozen=True)
class Annotation:
    start: float
    end: float
    labels: list


@DataManager.export("AnnotationData", analyser_pb2.ANNOTATION_DATA)
@dataclass(kw_only=True, frozen=True)
class AnnotationData(PluginData):
    type: str = field(default="AnnotationData")
    ext: str = field(default="msg")
    annotations: List[Annotation] = field(default_factory=list)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[AnnotationData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                # TODO use dump
                f.write(
                    msgpack.packb(
                        {
                            "annotations": [
                                {"start": x.start, "end": x.end, "labels": x.labels} for x in self.annotations
                            ]
                        }
                    )
                )
        except Exception as e:
            logging.error(f"AnnotationData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[AnnotationData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read())
            data = {
                "annotations": [
                    Annotation(start=x["start"], end=x["end"], labels=x["labels"]) for x in data["annotations"]
                ]
            }
        return data

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.ANNOTATION_DATA, "data_encoded": chunk, "ext": self.ext}


@DataManager.export("ScalarData", analyser_pb2.SCALAR_DATA)
@dataclass(kw_only=True, frozen=True)
class ScalarData(PluginData):
    type: str = field(default="ScalarData")
    ext: str = field(default="msg")
    y: npt.NDArray = field()
    time: List[float] = field(default_factory=list)
    delta_time: float = field(default=None)
    name: str = field(default=None)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[ScalarData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                f.write(
                    msgpack.packb({"y": self.y, "time": self.time, "delta_time": self.delta_time}, default=m.encode)
                )
        except Exception as e:
            logging.error(f"ScalarData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[ScalarData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read(), object_hook=m.decode)
        return data

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.SCALAR_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        if hasattr(self.y, "tolist"):
            return {"y": self.y.tolist(), "time": self.time, "delta_time": self.delta_time}

        return {"y": self.y, "time": self.time, "delta_time": self.delta_time}


@DataManager.export("RGBData", analyser_pb2.RGB_DATA)
@dataclass(kw_only=True, frozen=True)
class RGBData(PluginData):
    type: str = field(default="RGBData")
    ext: str = field(default="msg")
    colors: npt.NDArray = field(default_factory=np.ndarray)
    time: List[float] = field(default_factory=list)
    delta_time: float = field(default=None)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[RGBData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {"colors": self.colors, "time": self.time, "delta_time": self.delta_time}, default=m.encode
                    )
                )
        except Exception as e:
            logging.error(f"RGBData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[RGBData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read(), object_hook=m.decode)
        return data

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.RGB_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        if hasattr(self.colors, "tolist"):
            colors = self.colors.tolist()
        else:
            colors = self.colors
        return {"colors": colors, "time": self.time, "delta_time": self.delta_time}


@DataManager.export("ListData", analyser_pb2.LIST_DATA)
@dataclass(kw_only=True, frozen=True)
class ListData(PluginData):
    type: str = field(default="ListData")
    ext: str = field(default="msg")
    data: List[PluginData] = field(default_factory=list)
    index: List[Union[str, int]] = field(default=None)

    def __post_init__(self):
        if not self.index:
            object.__setattr__(self, "index", list(range(len(self.data))))

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[ListData::save_blob]")
        try:
            for d in self.data:
                d.save(data_dir)
        except Exception as e:
            logging.error(f"ListData::save_blob {e}")
            return False

        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                # TODO use dump
                f.write(msgpack.packb({"data": [{"id": d.id} for d in self.data], "index": self.index}))
        except Exception as e:
            logging.error(f"ListData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[ListData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data_decoded = msgpack.unpackb(f.read(), object_hook=m.decode)

        return {
            "data": [DataManager._load(data.get("data_dir"), d.get("id")) for d in data_decoded.get("data")],
            "index": data_decoded.get("index"),
        }

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        class DataYielder:
            def __init__(self, stream):
                self.cache = []
                self.stream = stream
                self.empty = False
                self.index = {}

            def push(self, data):
                self.cache.append(data)

            def get_next(self):
                if len(self.cache) > 0:
                    return self.cache.pop()

                return next(stream)

            def __iter__(self):
                try:
                    firstpkg = self.get_next()
                    firstpkg_decoded = msgpack.unpackb(firstpkg.data_encoded)
                    self.index[firstpkg_decoded.get("i")] = firstpkg_decoded.get("id")
                    chunk = firstpkg_decoded.get("chunk")

                    yield analyser_pb2.DownloadDataResponse(**chunk)
                    while True:
                        pkg = self.get_next()
                        pkg_decoded = msgpack.unpackb(pkg.data_encoded)

                        self.index[pkg_decoded.get("i")] = pkg_decoded.get("id")
                        if firstpkg_decoded.get("i") == pkg_decoded.get("i"):
                            chunk = pkg_decoded.get("chunk")

                            yield analyser_pb2.DownloadDataResponse(**chunk)
                        else:
                            self.push(pkg)
                            break
                except StopIteration as e:
                    self.empty = True
                    return

        yielder = DataYielder(stream)

        data = []
        while not yielder.empty:
            data.append(DataManager._load_from_stream(data_dir, yielder))

        data_obj = cls(
            data_dir=data_dir,
            data=data,
            index=list(map(lambda x: x[1], sorted(yielder.index.items(), key=lambda x: x[0]))),
        )

        data_obj.save(data_dir=data_dir)
        return data_obj

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        for i, (id, d) in enumerate(zip(self.index, self.data)):
            for chunk in d.dump_to_stream(chunk_size=chunk_size):

                yield {
                    "type": analyser_pb2.LIST_DATA,
                    "data_encoded": msgpack.packb({"i": i, "id": id, "chunk": chunk}),
                    "ext": self.ext,
                }

    def dumps_to_web(self):
        return {"y": self.y.tolist(), "time": self.time}


@dataclass(kw_only=True, frozen=True)
class BboxData(PluginData):
    image_id: int = None
    time: float = None
    delta_time: float = field(default=None)
    x: int = None
    y: int = None
    w: int = None
    h: int = None


@DataManager.export("BboxesData", analyser_pb2.BBOXES_DATA)
@dataclass(kw_only=True, frozen=True)
class BboxesData(PluginData):
    type: str = field(default="BboxesData")
    ext: str = field(default="msg")
    bboxes: List[BboxData] = field(default_factory=list)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[BboxesData::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {
                            "bboxes": [
                                {
                                    "x": bbox.x,
                                    "y": bbox.y,
                                    "w": bbox.w,
                                    "h": bbox.h,
                                    "time": bbox.time,
                                    "delta_time": bbox.delta_time,
                                }
                                for bbox in self.bboxes
                            ]
                        },
                        default=m.encode,
                    )
                )
        except Exception as e:
            logging.error(f"BboxesData::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[BboxesData::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read(), object_hook=m.decode)
            bboxes = {"bboxes": [BboxData(**bbox) for bbox in data.get("bboxes")]}
        return bboxes

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.BBOXES_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        if hasattr(self.bboxes, "tolist"):
            bboxes = self.bboxes.tolist()  # TODO
        else:
            bboxes = self.bboxes
        return {"bboxes": bboxes}


@dataclass(kw_only=True, frozen=True)
class StringData(PluginData):
    text: str = None


@dataclass(kw_only=True, frozen=True)
class ImageEmbedding(PluginData):
    image_id: int = None
    time: float = None
    delta_time: float = field(default=None)
    embedding: npt.NDArray = field(default_factory=np.ndarray)


@dataclass(kw_only=True, frozen=True)
class TextEmbedding(PluginData):
    text_id: int = None
    text: str = None
    embedding: npt.NDArray = field(default_factory=np.ndarray)


@DataManager.export("ImageEmbeddings", analyser_pb2.IMAGE_EMBEDDING_DATA)
@dataclass(kw_only=True, frozen=True)
class ImageEmbeddings(PluginData):
    type: str = field(default="ImageEmbeddings")
    ext: str = field(default="msg")
    embeddings: List[ImageEmbedding] = field(default_factory=list)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[ImageEmbeddings::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {
                            "embeddings": [
                                {
                                    "image_id": embd.image_id,
                                    "time": embd.time,
                                    "delta_time": embd.delta_time,
                                    "embedding": embd.embedding,
                                }
                                for embd in self.embeddings
                            ]
                        },
                        default=m.encode,
                    )
                )
        except Exception as e:
            logging.error(f"ImageEmbeddings::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[ImageEmbeddings::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read(), object_hook=m.decode)
            embeddings = {"embeddings": [ImageEmbedding(**embd) for embd in data.get("embeddings")]}
        return embeddings

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.IMAGE_EMBEDDING_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        if hasattr(self.embeddings, "tolist"):
            embeddings = self.embeddings.tolist()
        else:
            embeddings = self.embeddings
        return {"embeddings": embeddings}


@DataManager.export("TextEmbeddings", analyser_pb2.TEXT_EMBEDDING_DATA)
@dataclass(kw_only=True, frozen=True)
class TextEmbeddings(PluginData):
    type: str = field(default="TextEmbeddings")
    ext: str = field(default="msg")
    embeddings: List[TextEmbedding] = field(default_factory=list)

    def save_blob(self, data_dir=None, path=None):
        logging.debug(f"[TextEmbeddings::save_blob]")
        try:
            with open(create_data_path(data_dir, self.id, "msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {
                            "embeddings": [
                                {
                                    "text_id": embd.text_id,
                                    "text": embd.text,
                                    "embedding": embd.embedding,
                                }
                                for embd in self.embeddings
                            ]
                        },
                        default=m.encode,
                    )
                )
        except Exception as e:
            logging.error(f"TextEmbeddings::save_blob {e}")
            return False
        return True

    @classmethod
    def load_blob_args(cls, data: dict) -> dict:
        logging.debug(f"[TextEmbeddings::load_blob_args]")
        with open(create_data_path(data.get("data_dir"), data.get("id"), "msg"), "rb") as f:
            data = msgpack.unpackb(f.read(), object_hook=m.decode)
            embeddings = {"embeddings": [TextEmbedding(**embd) for embd in data.get("embeddings")]}
        return embeddings

    @classmethod
    def load_from_stream(cls, data_dir: str, stream: Iterator[bytes]) -> PluginData:
        firstpkg = next(stream)
        if hasattr(firstpkg, "ext") and len(firstpkg.ext) > 0:
            ext = firstpkg.ext
        else:
            ext = "msg"

        data_id = generate_id()
        path = create_data_path(data_dir, data_id, ext)

        with open(path, "wb") as f:
            f.write(firstpkg.data_encoded)
            for x in stream:
                f.write(x.data_encoded)

        data_args = {"id": data_id, "ext": ext, "data_dir": data_dir}

        return cls(**data_args, **cls.load_blob_args(data_args))

    def dump_to_stream(self, chunk_size=1024) -> Iterator[dict]:
        self.save(self.data_dir)
        with open(create_data_path(self.data_dir, self.id, "msg"), "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"type": analyser_pb2.TEXT_EMBEDDING_DATA, "data_encoded": chunk, "ext": self.ext}

    def dumps_to_web(self):
        if hasattr(self.embeddings, "tolist"):
            embeddings = self.embeddings.tolist()
        else:
            embeddings = self.embeddings
        return {"embeddings": embeddings}
