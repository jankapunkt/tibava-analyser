from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import BboxesData, ScalarData
from analyser.plugins import Plugin

import numpy as np

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"reduction": "max"}

requires = {
    "bboxes": BboxesData,
}

provides = {
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

    def call(self, inputs, parameters):
        facesizes_dict = {}
        delta_time = None

        for bbox in inputs["bboxes"].bboxes:
            if bbox.time not in facesizes_dict:
                facesizes_dict[bbox.time] = []
            facesizes_dict[bbox.time].append(bbox.w * bbox.h)
            delta_time = bbox.delta_time

        if parameters.get("reduction") == "max":
            facesizes = [np.max(x).tolist() for x in facesizes_dict.values()]
        else:  # parameters.get("reduction") == "mean":
            facesizes = [np.mean(x).tolist() for x in facesizes_dict.values()]

        return {"facesizes": ScalarData(y=facesizes, time=list(facesizes_dict.keys()), delta_time=delta_time)}
