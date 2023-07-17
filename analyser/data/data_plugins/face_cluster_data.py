import logging
from typing import List
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from analyser import analyser_pb2

@DataManager.export("FaceClusterData", analyser_pb2.FACE_CLUSTER_DATA)
@dataclass(kw_only=True)
class FaceClusterData(Data):
    type: str = field(default="FaceClusterData")
    clusters: List[List[str]] = field(default_factory=list)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("face_cluster_data.yml")
        self.clusters = data.get("facecluster")

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        self.save_dict(
            "face_cluster_data.yml",
            {"facecluster": self.clusters},
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "facecluster": self.clusters,
        }
