import zipfile
import logging
import yaml
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Callable, Optional, Dict

import uuid


class FSHandler:
    pass


class ZipFSHandler(FSHandler):
    def __init__(self, path: str, mode: str = "r") -> None:
        if mode is None:
            mode = "w"
        assert mode == "w" or mode == "r", "No valid mode for ZipData"
        # if os.path.exists(path):
        #     if mode != "r":
        #         raise FileExistsError()
        # else:
        #     if mode == "r":
        #         raise FileNotFoundError()

        self.path = path
        self.zipfile = None
        self.mode = mode

    def open(self, data) -> None:
        logging.info(f"open {self.path}")
        self.zipfile = zipfile.ZipFile(self.path, self.mode)
        if self.mode == "r":
            data.load()

    def mkdir(self, path) -> None:
        if self.zipfile is None:
            raise Exception()
        logging.info(f"mkdir {self.path}")
        self.zipfile.mkdir(path)

    def close(self, data) -> None:
        logging.info(f"close {self.path}")
        if self.zipfile is None:
            return

        if self.mode == "r":
            self.zipfile.close()
            self.zipfile = None
            return

        data.save()

        # data_split = split_data_in_paths(data)
        # logging.info(data_split)

        # dump_meta = data._meta_data()
        # with self.open_file("meta.yml", mode="w") as f:
        #     f.write(yaml.dump(data_split[]).encode())

        # logging.info(data._data())

        self.zipfile.close()
        self.zipfile = None

    def open_file(self, filename: str, mode: str = "r"):
        logging.info(f"{self.__class__.__name__} open_file {self.path} {filename}")
        if self.zipfile is None:
            logging.error("")
            return None

        if mode == "w" and self.mode == "r":
            raise ValueError

        return self.zipfile.open(filename, mode=mode, force_zip64=True)


class LocalFSHandler(FSHandler):
    def __init__(self, fs: FSHandler, path: str) -> None:
        self.fs = fs
        self.path = path

    def open(self, data) -> None:
        logging.info(f"open {self.path}")
        # if self.fs.mode != "r":
        #     self.fs.mkdir(self.path)
        if self.fs.mode == "r":
            data.load()

    @property
    def mode(self):
        return self.fs.mode

    def mkdir(self, path) -> None:
        if self.fs is None:
            raise Exception()
        logging.info(f"mkdir {self.path}")
        self.zipfile.mkdir(path)

    def close(self, data) -> None:
        logging.info(f"close {self.path}")
        if self.fs is None:
            return

        if self.fs.mode == "r":
            return

        data.save()

    def open_file(self, filename: str, mode: str = "r"):
        logging.info(f"{self.__class__.__name__} open_file {self.path} {filename}")
        if self.fs is None:
            logging.error("")
            return None

        if mode == "w" and self.fs.mode == "r":
            raise ValueError

        return self.fs.open_file(os.path.join(self.path, filename), mode=mode)
