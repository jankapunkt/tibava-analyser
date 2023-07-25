from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder

from analyser.inference import InferenceServer
from analyser.data import BboxData, BboxesData, FaceData, FacesData, KpsData, KpssData, ImageData, ImagesData, VideoData
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict

from analyser.utils import VideoDecoder
import cv2
import imageio.v3 as iio
import logging
import numpy as np
import sys
import traceback
import time


class InsightfaceDetectorTorch(AnalyserPlugin):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        inference_config = self.config.get("inference", None)

        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def detect(
        self,
        frame,
        input_size=(640, 640),
        det_thresh=0.5,
        nms_thresh=0.4,
        fps=10,
    ):
        img = frame.get("frame")
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        start_time = time.time()
        result = self.server(
            {"data": np.expand_dims(det_img, axis=0), "det_thresh": det_thresh, "nms_thresh": nms_thresh},
            ["boxes", "scores", "kpss"],
        )

        bboxes = result["boxes"] / det_scale
        kpss = result["kpss"] / det_scale
        scores = result["scores"]

        # create bbox, kps, and face objects (added to original code)
        bbox_list = []
        kps_list = []
        for i in range(len(scores)):
            x, y = round(max(0, bboxes[i][0])), round(max(0, bboxes[i][1]))
            w, h = round(bboxes[i][2] - x), round(bboxes[i][3] - y)
            det_score = scores[i]

            # number of pixels that is required to be considered a (useful) face
            threshold = 7500

            if (w * h < threshold):
                continue

            # store bbox
            bbox = {
                "x": float(x / img.shape[1]),
                "y": float(y / img.shape[0]),
                "w": float(w / img.shape[1]),
                "h": float(h / img.shape[0]),
                "det_score": float(det_score),
                "time": frame.get("time"),
                "delta_time": 1 / fps,
            }
            bbox_list.append(bbox)

            # store facial keypoints (kps)
            kps = {
                "x": [x.item() / img.shape[1] for x in kpss[i, :, 0]],
                "y": [y.item() / img.shape[0] for y in kpss[i, :, 1]],
                "time": frame.get("time"),
                "delta_time": 1 / fps,
            }
            kps_list.append(kps)

        return bbox_list, kps_list

    def predict_faces(self, iterator, num_frames, parameters, data_manager, callbacks):
        with data_manager.create_data("ImagesData") as images_data, data_manager.create_data(
            "BboxesData"
        ) as bboxes_data, data_manager.create_data("FacesData") as faces_data, data_manager.create_data(
            "KpssData"
        ) as kpss_data:
            # iterate through images to get face_images and bboxes
            for i, frame in enumerate(iterator):
                self.update_callbacks(callbacks, progress=i / num_frames)
                frame_bboxes, frame_kpss = self.detect(
                    frame,
                    parameters.get("input_size"),
                    det_thresh=parameters.get("det_thresh"),
                    nms_thresh=parameters.get("nms_thresh"),
                    fps=parameters.get("fps"),
                )

                for i in range(len(frame_bboxes)):
                    # store bboxes, kpss, and faces
                    face = FaceData(ref_id=frame.get("ref_id", None))
                    bbox = BboxData(**frame_bboxes[i], ref_id=face.id)
                    kps = KpsData(**frame_kpss[i], ref_id=face.id)

                    # faces.append(face)
                    # bboxes.append(bbox)
                    # kpss.append(kps)
                    bboxes_data.bboxes.append(bbox)
                    faces_data.faces.append(face)
                    kpss_data.kpss.append(kps)

                    # store face image
                    frame_image = frame.get("frame")
                    h, w = frame_image.shape[:2]

                    # draw kps
                    # for i in range(len(kps.x)):
                    #     x = round(kps.x[i] * w)
                    #     y = round(kps.y[i] * h)
                    #     frame_image[y - 1 : y + 1, x - 1 : x + 1, :] = [0, 255, 0]

                    # write faceimg
                    face_image = frame_image[
                        round(bbox.y * h) : round((bbox.y + bbox.h) * h),
                        round(bbox.x * w) : round((bbox.x + bbox.w) * w),
                        :,
                    ]

                    images_data.save_image(
                        face_image,
                        ext="jpg",
                        time=frame.get("time"),
                        delta_time=1 / parameters.get("fps"),
                        ref_id=face.id,
                    )
            self.update_callbacks(callbacks, progress=1.0)

            return {"images": images_data, "bboxes": bboxes_data, "kpss": kpss_data, "faces": faces_data}


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "insightface_torch",
    "model_device": "cpu",
    "model_file": "/models/insightface_detector_torch/scrfd_10g_bnkps.pth",
}

default_parameters = {"fps": 2, "det_thresh": 0.5, "nms_thresh": 0.4, "input_size": (640, 640)}

requires = {
    "video": VideoData,
}

provides = {"images": ImagesData, "bboxes": BboxesData, "kpss": KpssData, "faces": FacesData}


@AnalyserPluginManager.export("insightface_video_detector_torch")
class InsightfaceVideoDetectorTorch(
    InsightfaceDetectorTorch,
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
        with inputs["video"] as input_data:
            with input_data.open_video() as f_video:
                video_decoder = VideoDecoder(
                    f_video, fps=parameters.get("fps"), extension=f".{input_data.ext}", ref_id=input_data.id
                )

                # decode video to extract bboxes per frame
                # video_decoder = VideoDecoder(path=inputs["video"].path, fps=parameters.get("fps"))

                num_frames = video_decoder.duration() * video_decoder.fps()

                return self.predict_faces(
                    iterator=video_decoder,
                    num_frames=num_frames,
                    parameters=parameters,
                    data_manager=data_manager,
                    callbacks=callbacks,
                )


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "insightface_torch",
    "model_device": "cpu",
    "model_file": "/models/insightface_detector_torch/scrfd_10g_bnkps.pth",
}

default_parameters = {"fps": 1, "det_thresh": 0.5, "nms_thresh": 0.4, "input_size": (640, 640)}

requires = {
    "images": ImagesData,
}

provides = {"images": ImagesData, "bboxes": BboxesData, "kpss": KpssData, "faces": FacesData}


@AnalyserPluginManager.export("insightface_image_detector_torch")
class InsightfaceImageDetectorTorch(
    InsightfaceDetectorTorch,
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
        with inputs["images"] as input_data:

            def image_generator():
                for image in input_data:
                    frame = input_data.load_image(image)

                    yield {"frame": frame, "time": 0, "ref_id": image.id}

            images = image_generator()
            return self.predict_faces(
                iterator=images,
                num_frames=len(input_data),
                parameters=parameters,
                data_manager=data_manager,
                callbacks=callbacks,
            )
