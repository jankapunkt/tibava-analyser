from typing import Iterator
from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import (
    KpssData,
    FacesData,
    ImagesData,
    BboxesData,
    ImageEmbedding,
    ImageEmbeddings,
    VideoData,
)
import logging
import numpy as np
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"input_size": (640, 640)}

requires = {"video": VideoData, "kpss": KpssData}
provides = {"features": ImageEmbeddings}


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "insightface_w600k_r50",
    "model_device": "cpu",
    "model_file": "/models/insightface_feature_extraction/w600k_r50.onnx",
}

default_parameters = {"min_face_height": 0.1}

requires = {
    "images": ImagesData,
    "kpss": KpssData,
    "faces": FacesData,
    "bboxes": BboxesData,
}
provides = {
    "images": ImagesData,
    "kpss": KpssData,
    "faces": FacesData,
    "bboxes": BboxesData,
}


@AnalyserPluginManager.export("face_size_filter")
class FaceSizeFilter(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["images"] as images_data, inputs["kpss"] as kpss_data, inputs[
            "faces"
        ] as faces_data:
            kpss = kpss_data.kpss
            faces = faces_data.faces
            assert len(kpss) > 0

            image_lut = {image.id: image for image in images_data}
            face_image_lut = {face.id: face.ref_id for face in faces}
            kps_face_lut = {kps.ref_id: kps for kps in kpss}

            def get_iterator():
                for face_id, kps in kps_face_lut.items():
                    if face_id not in face_image_lut:
                        continue
                    image_id = face_image_lut[face_id]
                    if image_id not in image_lut:
                        continue

                    image_data = image_lut[image_id]

                    image = images_data.load_image(image_data)

                    yield {"frame": image, "kps": kps, "face_id": face_id}

            return self.get_facial_features(
                iterator=get_iterator(),
                num_faces=len(kps_face_lut),
                parameters=parameters,
                data_manager=data_manager,
                callbacks=callbacks,
            )
