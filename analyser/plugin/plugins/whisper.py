from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder, image_pad
from analyser.data import AudioData, AnnotationData, Annotation
from analyser.inference import InferenceServer
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict

import numpy as np
import librosa


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"sr": 16000, "chunk_length": 30}

requires = {
    "audio": AudioData,
}

provides = {
    "annotations": AnnotationData,
}


@AnalyserPluginManager.export("whisper")
class Whisper(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        inference_config = self.config.get("inference", None)
        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["audio"] as input_data, data_manager.create_data("AnnotationData") as output_data:
            with input_data.open_audio("r") as f_audio:
                y, sr = librosa.load(f_audio, sr=parameters.get("sr"))

                chunk_start = 0
                chunk_size = parameters.get("sr") * parameters.get("chunk_length")
                start = 0
                while chunk_start < len(y):
                    chunk_y = y[chunk_start : chunk_start + chunk_size]
                    result = self.server({"data": chunk_y}, ["text", "times"])
                    output_data.annotations.append(
                        Annotation(
                            start=start, end=start + parameters.get("chunk_length"), labels=[str(result["text"])]
                        )
                    )  # Maybe store max_mean_class_prob as well?
                    chunk_start += chunk_size
                    start += parameters.get("chunk_length")

                self.update_callbacks(callbacks, progress=1.0)
                return {"annotations": output_data}
