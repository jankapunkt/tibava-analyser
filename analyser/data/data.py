import zipfile
import logging
import yaml

from typing import Callable

import uuid


def generate_id():
    return uuid.uuid4().hex


class Data:
    pass


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
