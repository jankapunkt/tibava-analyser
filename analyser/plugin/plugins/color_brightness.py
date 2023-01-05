from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ScalarData, VideoData, ListData, RGBData
from analyser.utils import VideoDecoder
import numpy as np
import cv2

default_config = {"data_dir": "/data/"}


default_parameters = {
    "fps": 5,
    "normalize": True,
}

requires = {
    "video": VideoData,
}

provides = {
    "brightness": ScalarData,
}


@AnalyserPluginManager.export("color_brightness_analyser")
class ColorBrightnessAnalyser(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters, callbacks=None):
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=parameters.get("max_resolution"), fps=parameters.get("fps")
        )

        values = []
        time = []
        num_frames = video_decoder.duration() * video_decoder.fps()
        for i, frame in enumerate(video_decoder):
            self.update_callbacks(callbacks, progress=i / num_frames)
            image = frame["frame"]
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value = np.mean(hsv[:, :, 2])
            values.append(value)
            time.append(i / parameters.get("fps"))

        y = np.stack(values)

        if parameters.get("normalize"):
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        self.update_callbacks(callbacks, progress=1.0)
        return {
            "brightness": ScalarData(
                y=y,
                time=time,
                delta_time=1 / parameters.get("fps"),
            )
        }
