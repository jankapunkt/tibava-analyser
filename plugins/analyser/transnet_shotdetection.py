from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import Shot, ShotsData, VideoData, generate_id
from analyser.plugins import Plugin
from analyser.utils import InferenceServer

import ffmpeg
import numpy as np

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "transnet",
    "model_device": "cpu",
    "model_file": "/models/transnet_shotdetection/transnet.pt",
}

default_parameters = {"threshold": 0.5}

requires = {
    "video": VideoData,
}

provides = {
    "shots": ShotsData,
}


@AnalyserPluginManager.export("transnet_shotdetection")
class TransnetShotdetection(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]

        self.server = InferenceServer(
            model_file=self.model_file,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            device=self.model_device,
        )

    def predict_frames(self, frames: np.ndarray, callbacks):
        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr : ptr + 100]
                progress = ptr / len(padded_inputs)
                ptr += 50
                yield progress, out[np.newaxis]

        predictions = []
        # max_iter = len(input_iterator())
        for progress, inp in input_iterator():
            # (131362, 27, 48, 3)
            self.update_callbacks(callbacks, progress=progress)

            result = self.server({"data": inp}, ["single_frame_pred", "all_frames_pred"])

            if result is not None:
                single_frame_pred = result.get(f"single_frame_pred")
                all_frames_pred = result.get(f"all_frames_pred")

                predictions.append((single_frame_pred[0, 25:75, 0], all_frames_pred[0, 25:75, 0]))

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[: len(frames)], all_frames_pred[: len(frames)]  # remove extra padded frames

    def predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def call(self, inputs, parameters, callbacks=None):
        self.update_callbacks(callbacks, progress=0.0)
        video_stream, err = (
            ffmpeg.input(inputs["video"].path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )

        video_decoder = VideoDecoder(inputs["video"].path)
        video_decoder.fps

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

        prediction, _ = self.predict_frames(video, callbacks)

        shot_list = self.predictions_to_scenes(prediction, parameters.get("threshold"))

        data = ShotsData(
            shots=[
                Shot(start=x[0].item() / video_decoder.fps(), end=x[1].item() / video_decoder.fps()) for x in shot_list
            ]
        )

        self.update_callbacks(callbacks, progress=1.0)

        return {"shots": data}
