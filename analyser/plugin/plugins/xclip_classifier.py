from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import VideoData, generate_id, ListData, ScalarData


from analyser.inference import InferenceServer
import numpy as np
import cv2

import csv

# NOTE: The xclip model is slightly modified such that it only takes stacked frames as input.
# The text input are the pre-computed clip embeddings of the kinetics 600 dataset labels.
# https://github.com/microsoft/VideoX/tree/master/X-CLIP

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "xclip_classifier",
    "model_device": "gpu",
    "model_file": "/models/xclip/xclip_kin600_16_8.onnx",
    "image_resolution": (224, 224),
    "classes_file": "/models/xclip/kinetics_600_labels.csv",
    "seq_len": 8,
}

default_parameters = {
    "fps": 5,
}

requires = {
    "video": VideoData,
}

provides = {
    "logits": ListData,
}


@AnalyserPluginManager.export("xclip_classifier")
class XCLIPClassifier(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.image_resolution = self.config["image_resolution"]
        self.seq_len = self.config["seq_len"]

        self.classes = self.read_classes(self.config["classes_file"])

        self.server = InferenceServer(
            model_file=self.model_file,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            device=self.model_device,
            backend="ONNX",
        )

    def read_classes(self, classes_file):
        classes = []

        with open(classes_file, "r") as csvfile:
            content = csv.DictReader(csvfile)
            for line in content:
                classes.append(line["name"])
        return classes

    def call(self, inputs, parameters, callbacks=None):
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=self.image_resolution, fps=parameters.get("fps")
        )

        num_frames = video_decoder.duration() * video_decoder.fps()

        logits = []
        time = []

        frames = []
        # TODO: For now, we divide the video into N chunks of seq_len
        # Think about using Shots data
        for frame in video_decoder:
            frames.append(frame["frame"])
            if len(frames) == self.seq_len:
                frames = np.stack(frames)
                batch = np.transpose(frames, (0, 3, 1, 2))  # [T, C, H, W]
                batch = np.expand_dims(batch, 0)  # [B, T, C, H, W]

                res = self.server({"data": batch.astype(np.float32)}, ["logits"])["logits"]
                logits.append(res)

                frames = []

                time.append(frame["time"])

            self.update_callbacks(callbacks, progress=frame["index"] / num_frames)

        self.update_callbacks(callbacks, progress=1.0)

        logits = np.concatenate(logits)

        return {"logits": ListData(data=[ScalarData(y=logits, time=time)], index=[self.classes])}
