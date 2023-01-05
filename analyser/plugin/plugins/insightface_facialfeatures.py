from typing import Iterator
from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import (
    KpssData,
    FacesData,
    ImagesData,
    ImageEmbedding,
    ImageEmbeddings,
    VideoData,
    create_data_path,
    generate_id,
)
import cv2
import imageio.v3 as iio
import logging
import numpy as np
import traceback

# from skimage import transform as trans
import sys

from analyser.inference import InferenceServer

src1 = np.array(
    [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]], dtype=np.float32
)
# <--left
src2 = np.array(
    [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]], dtype=np.float32
)

# ---frontal
src3 = np.array(
    [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]], dtype=np.float32
)

# -->right
src4 = np.array(
    [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]], dtype=np.float32
)

# -->right profile
src5 = np.array(
    [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]], dtype=np.float32
)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

arcface_src = np.expand_dims(arcface_src, axis=0)


class InsightfaceFeatureExtractor(AnalyserPlugin):
    def __init__(self, config=None):
        super().__init__(config)
        inference_config = self.config.get("inference", None)

        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))
        # copied from insightface for w600_r50.onnx model
        self.input_size = tuple([112, 112])
        self.input_std = 127.5
        self.input_mean = 127.5

    def estimate_norm(self, lmk, image_size=112, mode="arcface"):
        assert lmk.shape == (5, 2)
        # tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float("inf")
        if mode == "arcface":
            if image_size == 112:
                src = arcface_src
            else:
                src = float(image_size) / 112 * arcface_src
        else:
            src = src_map[image_size]
        for i in np.arange(src.shape[0]):
            # tform.estimate(lmk, src[i])
            # M = tform.params[0:2, :]
            M = cv2.estimateAffinePartial2D(lmk, src[i])[0]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, img, landmark, image_size=112, mode="arcface"):
        M, _ = self.estimate_norm(landmark, image_size, mode)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )

        return self.server({"data": blob}, ["embedding"])["embedding"]

    def get_facial_features(self, iterator, num_faces, parameters, callbacks):
        try:
            features = []

            # iterate through images to get face_images and bboxes
            for i, face in enumerate(iterator):
                kps = face.get("kps")
                frame = face.get("frame")
                h, w = frame.shape[0:2]
                landmark = np.column_stack([kps.x, kps.y])
                landmark *= (w, h)  # revert normalization done in insightface_detector.py

                image_id = generate_id()
                aimg = self.norm_crop(face.get("frame"), landmark=landmark)
                output_path = create_data_path(self.config.get("data_dir"), image_id, "jpg")
                iio.imwrite(output_path, aimg)

                features.append(
                    ImageEmbedding(
                        ref_id=face.get("face_id"),
                        embedding=self.get_feat(aimg).flatten(),
                        time=kps.time,
                        delta_time=1 / parameters.get("fps"),
                    )
                )

                self.update_callbacks(callbacks, progress=i / num_faces)

            self.update_callbacks(callbacks, progress=1.0)
            return {"features": ImageEmbeddings(embeddings=features)}

        except Exception as e:
            logging.error(f"InsightfaceDetector: {repr(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()

            traceback.print_exception(
                exc_type,
                exc_value,
                exc_traceback,
                limit=2,
                file=sys.stdout,
            )
        return {}


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "insightface_w600k_r50",
    "model_device": "cpu",
    "model_file": "/models/insightface_feature_extraction/w600k_r50.onnx",
}

default_parameters = {"input_size": (640, 640)}

requires = {"video": VideoData, "kpss": KpssData}
provides = {"features": ImageEmbeddings}


@AnalyserPluginManager.export("insightface_video_feature_extractor")
class InsightfaceVideoFeatureExtractor(
    InsightfaceFeatureExtractor,
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters, callbacks=None):
        try:
            kpss = inputs["kpss"].kpss
            parameters["fps"] = 1 / kpss[0].delta_time
            assert len(kpss) > 0

            faceid_lut = {}
            for kps in kpss:
                faceid_lut[kps.id] = kps.ref_id

            # decode video to extract kps for frames with detected faces
            video_decoder = VideoDecoder(path=inputs["video"].path, fps=parameters.get("fps"))
            kps_dict = {}
            num_faces = 0
            for kps in kpss:
                if kps.time not in kps_dict:
                    kps_dict[kps.time] = []
                num_faces += 1
                kps_dict[kps.time].append(kps)

            def get_iterator(video_decoder, kps_dict):
                # TODO: change VideoDecoder class to be able to directly seek the video for specific frames
                # WORKAROUND: loop over the whole video and store frames whenever there is a face detected
                for frame in video_decoder:
                    t = frame["time"]
                    if t in kps_dict:
                        for kps in kps_dict[t]:
                            face_id = faceid_lut[kps.id] if kps.id in faceid_lut else None
                            yield {"frame": frame["frame"], "kps": kps, "face_id": face_id}

            iterator = get_iterator(video_decoder, kps_dict)
            return self.get_facial_features(
                iterator=iterator, num_faces=num_faces, parameters=parameters, callbacks=callbacks
            )

        except Exception as e:
            logging.error(f"InsightfaceVideoFeatureExtractor: {repr(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()

            traceback.print_exception(
                exc_type,
                exc_value,
                exc_traceback,
                limit=2,
                file=sys.stdout,
            )
        return {}


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "insightface_w600k_r50",
    "model_device": "cpu",
    "model_file": "/models/insightface_feature_extraction/w600k_r50.onnx",
}

default_parameters = {"input_size": (640, 640)}

requires = {"images": ImagesData, "kpss": KpssData}
provides = {"features": ImageEmbeddings}


@AnalyserPluginManager.export("insightface_image_feature_extractor")
class InsightfaceImageFeatureExtractor(
    InsightfaceVideoFeatureExtractor,
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters, callbacks=None):
        try:
            kpss = inputs["kpss"].kpss
            assert len(kpss) > 0

            faceid_lut = {}
            for kps in kpss:
                faceid_lut[kps.id] = kps.ref_id

            image_paths = [
                create_data_path(inputs["images"].data_dir, image.id, image.ext) for image in inputs["images"].images
            ]

            kps_dict = {}
            num_faces = 0
            for kps in kpss:
                if kps.ref_id not in image_paths:
                    continue

                if kps.ref_id not in kps_dict:
                    kps_dict[kps.ref_id] = []

                num_faces += 1
                kps_dict[kps.ref_id].append(kps)

            def get_iterator(kps_dict):
                for image_path in kps_dict:
                    image = iio.imread(image_path)

                    for kps in kps_dict[image_path]:
                        face_id = faceid_lut[kps.id] if kps.id in faceid_lut else None
                        yield {"frame": image, "kps": kps, "face_id": face_id}

            iterator = get_iterator(kps_dict)
            return self.get_facial_features(
                iterator=iterator, num_faces=num_faces, parameters=parameters, callbacks=callbacks
            )

        except Exception as e:
            logging.error(f"InsightfaceImageDetector: {repr(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()

            traceback.print_exception(
                exc_type,
                exc_value,
                exc_traceback,
                limit=2,
                file=sys.stdout,
            )
        return {}
