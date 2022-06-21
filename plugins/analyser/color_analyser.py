from ctypes.wintypes import RGB
from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import VideoData, ScalarData, ListData, RGBData
from analyser.plugins import Plugin
from analyser.utils import VideoDecoder
import logging
from sklearn.cluster import KMeans
import numpy as np

default_config = {"data_dir": "/data/"}


default_parameters = {
    "k": 4,
    "fps": 5,
    "max_iter": 20,
    "max_resolution": 64,
}

requires = {
    "video": VideoData,
}

provides = {
    "colors": ListData,
}


@AnalyserPluginManager.export("color_analyser")
class ColorAnalyser(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters):
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=parameters.get("max_resolution"), fps=parameters.get("fps")
        )

        kcolors = []
        time = []
        for i, frame in enumerate(video_decoder):
            image = frame["frame"]
            image = image.reshape((image.shape[0] * image.shape[1], 3))
            cls = KMeans(n_clusters=parameters.get("k"), max_iter=parameters.get("max_iter"))
            cls.fit(image)
            kcolors.append(cls.cluster_centers_.tolist())
            time.append(i / parameters.get("fps"))

        return {
            "colors": ListData(data=[RGBData(colors=np.asarray(colors) / 255, time=time) for colors in zip(*kcolors)])
        }
