import logging
from typing import List
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from analyser import analyser_pb2


@dataclass(kw_only=True)
class StringData(Data):
    text: str = None

    def to_dict(self) -> dict:
        return {"text": self.text}
