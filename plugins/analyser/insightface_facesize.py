from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import BboxesData, ScalarData
from analyser.plugins import Plugin


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
        delta_time = None

        for bbox in inputs["bboxes"].bboxes:
            facesizes.append(bbox.w * bbox.h)
            time.append(bbox.time)
            delta_time = bbox.delta_time

        return {"facesizes": ScalarData(y=facesizes, time=time, delta_time=delta_time)}
