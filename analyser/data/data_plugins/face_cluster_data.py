import logging
from typing import List
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from .face_data import FacesData, FaceData
from .keypoint_data import KpssData, KpsData
from .bounding_box_data import BboxesData, BboxData
from .image_data import ImagesData, ImageData
from analyser import analyser_pb2

@DataManager.export("FaceClusterData", analyser_pb2.FACE_CLUSTER_DATA)
@dataclass(kw_only=True)
class FaceClusterData(Data):
    type: str = field(default="FaceClusterData")
    clusters: List[List[str]] = field(default_factory=list)
    faces: FacesData = field(default_factory=list)
    kpss: KpssData = field(default_factory=list)
    bboxes: BboxesData = field(default_factory=list)
    images: ImagesData = field(default_factory=list)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("face_cluster_data.yml")
        self.clusters = data.get("facecluster")
        self.faces = [FaceData(**x) for x in data.get("faces")]
        self.kpss = [KpsData(**x) for x in data.get("kpss")]
        self.bboxes = [BboxData(**x) for x in data.get("bboxes")]
        self.images = [ImageData(**x) for x in data.get("images")]

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        self.save_dict(
            "face_cluster_data.yml",
            {
                "facecluster": self.clusters, 
                "faces": [face.to_dict() for face in self.faces.faces],
                "kpss": [kp.to_dict() for kp in self.kpss.kpss],
                "bboxes": [box.to_dict() for box in self.bboxes.bboxes],
                "images": [image.to_dict() for image in self.images.images],
            },
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "facecluster": self.clusters,
            "faces": [face.to_dict() for face in self.faces],
            "kpss": [kp.to_dict() for kp in self.kpss],
            "bboxes": [box.to_dict() for box in self.bboxes],
            "images": [image.to_dict() for image in self.images],
        }
