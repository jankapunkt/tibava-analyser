from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.data import AudioData, AnnotationData, Annotation

# from analyser.inference import InferenceServer
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict
import logging

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


@AnalyserPluginManager.export("whisper_surf")
class WhisperSurf(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # inference_config = self.config.get("inference", None)
        # self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

        self.model = None

        # self.config refers to what is defined in
        # inference_ray/deploy.yml
        self.model_name = self.config.get("model", "openai/whisper-base")

    def call(
            self,
            inputs: Dict[str, Data],
            data_manager: DataManager,
            parameters: Dict = None,
            callbacks: Callable = None,
    ) -> Dict[str, Data]:
        # prepare data
        # call surf api
        # post process data
        # return
        # output_data = data_manager.create_data("AnnotationData")
        logging.error('[wisper_surf]: create new AnnotationData')
        with data_manager.create_data("AnnotationData") as output_data:
            logging.error('[wisper_surf]: output data created, add timestamp1')
            output_data.annotations.append(
                Annotation(start="00:00", end="00:01",
                           labels=["foo", "bar", "baz", "moo"])
            )
            logging.error('[wisper_surf]: output data created, add timestamp2')
            output_data.annotations.append(
                Annotation(start="00:00", end="00:01",
                           labels=["foo", "bar", "baz", "moo"])
            )
            logging.error('[wisper_surf]: output data created, add timestamp3')
            output_data.annotations.append(
                Annotation(start="00:00", end="00:01",
                           labels=["foo", "bar", "baz", "moo"])
            )
            logging.error('[wisper_surf]: update callbacks')
            self.update_callbacks(callbacks, progress=1.0)
            logging.error('[wisper_surf]: done')
            return {"annotations": output_data}
