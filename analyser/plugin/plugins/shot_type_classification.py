from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder, image_pad
from analyser.data import ListData, ScalarData, VideoData, ListData
from analyser.inference import InferenceServer
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict

import numpy as np


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "shot_type_classifier",
    "model_device": "cpu",
    "model_file": "/models/shot_type_classification/shot_type_classifier_e9-s3199_cpu.pt",
    "image_resolution": 224,
}

default_parameters = {"fps": 5}

requires = {
    "video": VideoData,
}

provides = {
    "probs": ListData,
}


@AnalyserPluginManager.export("shot_type_classifier")
class ShotTypeClassifier(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.image_resolution = self.config["image_resolution"]

        inference_config = self.config.get("inference", None)

        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=self.image_resolution, fps=parameters.get("fps")
        )

        # video_decoder.fps

        predictions = []
        time = []

        num_frames = video_decoder.duration() * video_decoder.fps()
        for i, frame in enumerate(video_decoder):

            self.update_callbacks(callbacks, progress=i / num_frames)
            frame = image_pad(frame["frame"])

            result = self.server({"data": np.expand_dims(frame, 0)}, ["prob"])
            if result is not None:
                predictions.append(result["prob"].tolist())
                time.append(i / parameters.get("fps"))
        # predictions = zip(*predictions)
        probs = ListData(
            data=[
                ScalarData(y=np.asarray(y), time=time, delta_time=1 / parameters.get("fps")) for y in zip(*predictions)
            ],
            index=["p_ECU", "p_CU", "p_MS", "p_FS", "p_LS"],
        )

        # predictions: list(np.array) in form of [(p_ECU, p_CU, p_MS, p_FS, p_LS), ...] * #frames
        # times: list in form [0 / fps, 1 / fps, ..., #frames/fps]

        self.update_callbacks(callbacks, progress=1.0)
        return {"probs": probs}
