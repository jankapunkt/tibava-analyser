from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import Shot, ShotsData, ImagesData, generate_id
from analyser.plugins import Plugin
import ffmpeg
import os
import redisai as rai
import ml2rt
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
    "images": ImagesData,
}

provides = {
    # "emotions": ProbData,
}


@AnalyserPluginManager.export("face_emotion")
class FaceEmotion(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]

        self.con = rai.Client(host=self.host, port=self.port)

        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name,
            backend="torch",
            device=self.model_device,
            data=model,
            batch=16,
        )

    def predict_frames(self, frames: np.ndarray):

        job_id = generate_id()

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
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():

            self.con.tensorset(f"data_{job_id}", inp)
            result = self.con.modelrun(
                self.model_name, f"data_{job_id}", [f"single_frame_pred_{job_id}", f"all_frames_pred_{job_id}"]
            )

            single_frame_pred = self.con.tensorget(f"single_frame_pred_{job_id}")
            all_frames_pred = self.con.tensorget(f"all_frames_pred_{job_id}")

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

    def call(self, inputs, parameters):

        output_data = ShotsData(ext="msg", data_dir=self.config.get("data_dir"))
        video_stream, err = (
            ffmpeg.input(inputs["video"].path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

        print(video.shape, flush=True)

        prediction, _ = self.predict_frames(video)

        shot_list = self.predictions_to_scenes(prediction, parameters.get("threshold"))

        data = ShotsData(shots=[Shot(start=x[0].item(), end=x[1].item()) for x in shot_list])
        # print(probabilities)

        return {"shots": data}
