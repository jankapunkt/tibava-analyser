from lzma import PRESET_DEFAULT
from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder, image_pad, image_resize
from analyser.data import ScalarData, VideoData, ProbData, generate_id
from analyser.plugins import Plugin
import redisai as rai
import ml2rt
import numpy as np
import logging

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "shot_type_classifier",
    "model_device": "cpu",
    "model_file": "/models/shot_type_classification/shot_type_classifier_e9-s3199_cpu.pt",
}

default_parameters = {}

requires = {
    "video": VideoData,
}

provides = {
    "probs": ProbData,
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

        self.con = rai.Client(host=self.host, port=self.port)

        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name,
            backend="torch",
            device=self.model_device,
            data=model,
            batch=16,
        )

    def call(self, inputs, parameters):

        video_decoder = VideoDecoder(inputs["video"].path)
        video_decoder.fps
        job_id = generate_id()

        predictions = []
        times = []
        for i, frame in enumerate(video_decoder):
            if i % 100 == 0:
                logging.info(f"shot_type_classification.py: {i} frames processed")
            frame = image_pad(frame["frame"])
            frame = image_resize(frame, size=(224, 224))
            self.con.tensorset(f"data_{job_id}", frame)

            _ = self.con.modelrun(self.model_name, f"data_{job_id}", f"prob_{job_id}")
            prediction = self.con.tensorget(f"prob_{job_id}")
            predictions.append(prediction)
            times.append(i / video_decoder.fps())

        # predictions: list(list) in form of [p_ECU, p_CU, p_MS, p_FS, p_LS] * #frames
        # times: list in form [0 / fps, 1 / fps, ..., #frames/fps]
        logging.info("shot_type_classification.py: return result")
        data = ProbData(
            probs=ScalarData(y=predictions, time=times),
            labels=["Extreme Close-Up", "Close-Up", "Medium Shot", "Full Shot", "Long Shot"],
            shortlabels=["ECU", "CU", "MS", "FS", "LS"],
        )

        return {"probs": data}
