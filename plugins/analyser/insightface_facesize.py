from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import BboxesData, ScalarData
from analyser.plugins import Plugin
import logging


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {}

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
        facesizes = []
        time = []

        print(inputs)
        for bbox in inputs["bboxes"].bboxes:
            print(bbox.x, bbox.y, bbox.w, bbox.h)
            facesizes.append(bbox.w * bbox.h)
            time.append(bbox.time)

        return {"facesizes": ScalarData(y=facesizes, time=time)}
