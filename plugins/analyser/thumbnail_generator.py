from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import ImageData, VideoData, ShotsData
from analyser.plugins import Plugin

default_config = {
    "host": "localhost",
    "port": 6379,
    "model_name": "byol_wikipedia",
    "model_device": "gpu",
    "model_file": "/home/matthias/transnetv2.mar",
}


default_parameters = {"fps": 1.0, "max_resolution": 128}

requires = {
    "video": VideoData,
}

provides = {
    "images": ImageData,
}


@AnalyserPluginManager.export("thumbnail_generator")
class ThumbnailGenerator(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None, name=None):
        super().__init__(config, name)

    def call(self, inputs):
        pass
