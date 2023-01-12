import logging
from typing import List, Union
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from analyser import analyser_pb2


@DataManager.export("ListData", analyser_pb2.LIST_DATA)
@dataclass(kw_only=True)
class ListData(Data):
    type: str = field(default="ListData")
    data: List[Data] = field(default_factory=list)
    index: List[Union[str, int]] = field(default=None)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("kpss_data.yml")
        self.kpss = [KpsData(**x) for x in data.get("kpss")]

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        self.save_dict("kpss_data.yml", {"kpss": [x.to_dict() for x in self.kpss]})

    def to_dict(self) -> dict:
        return {**super().to_dict(), "kpss": [x.to_dict() for x in self.kpss]}
