from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import BboxesData, ListData, ScalarData
from analyser.plugins import Plugin

import numpy as np
import pickle

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_file": "/models/naivebayes_facesize/naivebayes_facesize.pkl",
}

default_parameters = {"reduction": "max"}

requires = {
    "bboxes": BboxesData,
}

provides = {
    "probs": ListData,
    "facesizes": ScalarData,
}


@AnalyserPluginManager.export("insightface_facesize")
class InsightfaceFacesize(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]

        with open(self.config["model_file"], "rb") as pklfile:
            self.model = pickle.load(pklfile)

    def call(self, inputs, parameters, callbacks=None):
        facesizes_dict = {}
        delta_time = None

        for i, bbox in enumerate(inputs["bboxes"].bboxes):
            if bbox.time not in facesizes_dict:
                facesizes_dict[bbox.time] = []
            facesizes_dict[bbox.time].append(bbox.w * bbox.h)
            delta_time = bbox.delta_time

        if parameters.get("reduction") == "max":
            facesizes = [np.max(x).tolist() for x in facesizes_dict.values()]
        else:  # parameters.get("reduction") == "mean":
            facesizes = [np.mean(x).tolist() for x in facesizes_dict.values()]

        # predict shot size based on facesizes
        predictions = self.model.predict_proba(np.asarray(facesizes).reshape(-1, 1))

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "probs": ListData(
                data=[
                    ScalarData(y=np.asarray(y), time=list(facesizes_dict.keys()), delta_time=delta_time)
                    for y in zip(*predictions)
                ],
                index=["p_ECU", "p_CU", "p_MS", "p_FS", "p_LS"],
            ),
            "facesizes": ScalarData(y=facesizes, time=list(facesizes_dict.keys()), delta_time=delta_time),
        }
