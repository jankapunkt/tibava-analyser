from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import VideoData, ScalarData, generate_id
from analyser.plugins import Plugin
from analyser.utils import VideoDecoder
import json
import logging
import numpy as np
from sklearn.cluster import KMeans

default_config = {"data_dir": "/data/"}


default_parameters = {
    "k": 8,
    "fps": 5,
    "max_iter": 20,
    "max_resolution": 128,
}

requires = {
    "video": VideoData,
}

provides = {
    "colors": ScalarData,
}


@AnalyserPluginManager.export("color_analyser")
class ColorAnalyser(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters):

        logging.info("color_analyser::start")

        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=parameters.get("max_resolution"), fps=parameters.get("fps")
        )
        # video_decoder.fps
        # job_id = generate_id()

        kcolors = []
        time = []
        for i, frame in enumerate(video_decoder):
            frame = frame["frame"]
            logging.info(f"color_analyser::frame_shape{frame.shape}")
            image = frame.reshape((frame.shape[0] * frame.shape[1], 3))
            cls = KMeans(n_clusters=parameters.get("k"), max_iter=parameters.get("max_iter"))
            cls.fit(image)
            kcolors.append(cls.cluster_centers_)
            time.append(i / parameters.get("fps"))

        logging.info(f"color_analyser::colors_shape{kcolors[0].shape}")

        return {"colors": ScalarData(y=kcolors, time=time)}
