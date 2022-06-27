from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder, image_pad
from analyser.data import ListData, ScalarData, VideoData, ListData, generate_id
from analyser.plugins import Plugin
from analyser.utils import InferenceServer

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
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.image_resolution = self.config["image_resolution"]

        self.server = InferenceServer(
            model_file=self.model_file, model_name=self.model_name, host=self.host, port=self.port
        )

    def call(self, inputs, parameters):
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=self.image_resolution, fps=parameters.get("fps")
        )

        # video_decoder.fps

        predictions = []
        time = []
        for i, frame in enumerate(video_decoder):
            frame = image_pad(frame["frame"])

            result = self.server({"data": frame}, ["prob"])
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
        return {"probs": probs}
