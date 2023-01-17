import logging
from typing import List, Union
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from ..fs_handler import LocalFSHandler
from analyser import analyser_pb2


@DataManager.export("ListData", analyser_pb2.LIST_DATA)
@dataclass(kw_only=True)
class ListData(Data):
    type: str = field(default="ListData")
    data: List[str] = field(default_factory=list)
    index: List[Union[str, int]] = field(default_factory=list)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("list_data.yml")
        self.index = data.get("index")
        self.data = data.get("data")

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        self.save_dict("list_data.yml", {"index": self.index, "data": self.data})

    def create_data(self, data_type: str, index: str = None):
        assert self.fs.mode == "w", "Data packet is open read only"
        assert data_type in DataManager._data_name_lut, f"Unknown data type {data_type}"

        data = DataManager._data_name_lut[data_type]()
        data._register_fs_handler(LocalFSHandler(self.fs, data.id))

        self.data.append(data.id)
        if index is None:
            index = len(self.index)
        self.index.append(index)

        return data

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        for i, data_id in zip(self.index, self.data):
            data = Data()
            data._register_fs_handler(LocalFSHandler(self.fs, data_id))
            data_type = None
            with data:
                data_type = data.type

            assert data_type in DataManager._data_name_lut, f"Unknown data type {data_type}"

            data = DataManager._data_name_lut[data_type]()
            data._register_fs_handler(LocalFSHandler(self.fs, data_id))

            yield i, data

    def to_dict(self) -> dict:
        # TODO
        result = {**super().to_dict(), "data": [], "index": []}
        for i, data in self:
            with data:
                result["index"].append(i)
                result["data"].append(data.to_dict())

        return result
