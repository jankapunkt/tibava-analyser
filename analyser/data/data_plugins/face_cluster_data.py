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
from .image_embedding import ImageEmbeddings, ImageEmbedding
from analyser import analyser_pb2

@dataclass(kw_only=True)
class Cluster(Data):
    face_refs: List[str] = field(default_factory=list)
    embedding_repr: List[ImageEmbedding] = field(default_factory=list)

    def to_dict(self) -> dict:
        meta = super().to_dict()
        return {
            **meta,
            "face_refs": self.face_refs,
            "embedding_repr": [x.to_dict() for x in self.embedding_repr],
        }
    
@DataManager.export("FaceClusterData", analyser_pb2.FACE_CLUSTER_DATA)
@dataclass(kw_only=True)
class FaceClusterData(Data):
    type: str = field(default="FaceClusterData")
    clusters: List[Cluster] = field(default_factory=list)
    faces: FacesData = field(default_factory=list)
    kpss: KpssData = field(default_factory=list)
    bboxes: BboxesData = field(default_factory=list)
    images: ImagesData = field(default_factory=list)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("face_cluster_data.yml")
        self.clusters = [Cluster(**x) for x in data.get("facecluster")]
        for cluster in self.clusters:
            cluster.embedding_repr = [ImageEmbedding(**x) for x in cluster.embedding_repr]
            for img_emb in cluster.embedding_repr:
                img_emb.embedding = np.asarray(img_emb.embedding)

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
                "facecluster": [c.to_dict() for c in self.clusters], 
                "faces": [face.to_dict() for face in self.faces.faces],
                "kpss": [kp.to_dict() for kp in self.kpss.kpss],
                "bboxes": [box.to_dict() for box in self.bboxes.bboxes],
                "images": [image.to_dict() for image in self.images.images],
            },
        )

    def to_dict(self) -> dict:
        print(type(self.clusters[0]))
        return {
            **super().to_dict(),
            "facecluster": [c.to_dict() for c in self.clusters],
            "faces": [face.to_dict() for face in self.faces],
            "kpss": [kp.to_dict() for kp in self.kpss],
            "bboxes": [box.to_dict() for box in self.bboxes],
            "images": [image.to_dict() for image in self.images],
        }
