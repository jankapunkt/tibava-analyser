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
from analyser.proto import analyser_pb2
from .image_embedding import ImageEmbedding


@dataclass(kw_only=True)
class Cluster(Data):
    object_refs: List[str] = field(default_factory=list)
    embedding_repr: List[npt.NDArray] = field(default_factory=list)

    def to_dict(self) -> dict:
        meta = super().to_dict()
        return {
            **meta,
            "object_refs": self.object_refs,
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
        self.clusters = [Cluster(**x) for x in data.get("cluster")]

        self.faces = [FaceData(**x) for x in data.get("faces")]
        self.kpss = [KpsData(**x) for x in data.get("kpss")]
        self.bboxes = [BboxData(**x) for x in data.get("bboxes")]
        self.images = [ImageData(**x) for x in data.get("images")]

        with self.fs.open_file("face_cluster_embeddings.npz", "r") as f:
            embeddings = np.load(f)

        cluster_feature_lut = data.get("cluster_feature_lut")

        for i in range(len(self.clusters)):
            self.clusters[i].embedding_repr = embeddings[
                cluster_feature_lut[i][0] : cluster_feature_lut[i][1]
            ]

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        cluster_feature_lut = {}

        i = 0
        for j, cluster in enumerate(self.clusters):
            cluster_feature_lut[j] = (i, i + len(cluster.embedding_repr))
            i += len(cluster.embedding_repr)

        self.save_dict(
            "face_cluster_data.yml",
            {
                "cluster": [c.to_dict() for c in self.clusters],
                "cluster_feature_lut": cluster_feature_lut,
                "faces": [face.to_dict() for face in self.faces.faces],
                "kpss": [kp.to_dict() for kp in self.kpss.kpss],
                "bboxes": [box.to_dict() for box in self.bboxes.bboxes],
                "images": [image.to_dict() for image in self.images.images],
            },
        )

        with self.fs.open_file("face_cluster_embeddings.npz", "w") as f:
            np.save(
                f, np.concatenate([x.embedding_repr for x in self.clusters], axis=0)
            )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "cluster": [c.to_dict() for c in self.clusters],
            "faces": [face.to_dict() for face in self.faces],
            "kpss": [kp.to_dict() for kp in self.kpss],
            "bboxes": [box.to_dict() for box in self.bboxes],
            "images": [image.to_dict() for image in self.images],
        }
