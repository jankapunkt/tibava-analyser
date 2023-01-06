import zipfile
import logging
import yaml
from dataclasses import dataclass, field
from typing import Callable, Optional

import uuid


def generate_id():
    return uuid.uuid4().hex


class FSHandler:
    pass


class ZipFSHandler:
    def __init__(self):
        pass

    def open(self):
        pass

    def close(self):
        pass


@dataclass(kw_only=True)
class Data:
    id: str = field(default_factory=generate_id)
    version: str = field(default="1.0")
    type: str = field(default="PluginData")
    name: Optional[str]
    ref_id: Optional[str]

    def _register_fs_handler(self, fs_handler: FSHandler) -> None:
        self.fs_handler = fs_handler

    def __enter__(self):
        if hasattr(self, "fs_handler") and self.fs_handler:
            self.fs_handler.open(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "fs_handler") and self.fs_handler:
            self.fs_handler.close(self)
        # self.fs_handler.close()
        # if self.zipfile is None:
        #     return

        # if self.mode == "w":
        #     dump_meta = self.dump_meta()
        #     with self.open_file("meta.yml", mode="w") as f:
        #         f.write(yaml.dump(dump_meta).encode())

        # self.zipfile.close()
        # self.zipfile = None


class ContainerData:
    def __init__(self, file_path: str = None, id: str = None, type: str = None, mode: str = None):
        if mode is None:
            mode = "w"
        assert mode == "w" or mode == "r", "No valid mode for ZipData"

        if id is None:
            id = generate_id()
        self.id = id

        self.file_path = file_path
        self.type = type

        self.mode = mode
        self.zipfile = zipfile.ZipFile(self.file_path, self.mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.zipfile is None:
            return

        if self.mode == "w":
            dump_meta = self.dump_meta()
            with self.open_file("meta.yml", mode="w") as f:
                f.write(yaml.dump(dump_meta).encode())

        self.zipfile.close()
        self.zipfile = None

    def open_file(self, path: str, mode: str = "r"):
        if self.zipfile is None:
            logging.error("")
            return None

        return self.zipfile.open(path, mode=mode, force_zip64=True)

    def dump_meta(self):
        return {
            "type": self.type,
            "id": self.id,
        }

    def extract_all(self, path: str, path_generator: Callable[[str], str] = None):
        pass
