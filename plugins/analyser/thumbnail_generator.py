import os

import imageio

from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import ImageData, VideoData, ImagesData
from analyser.data import generate_id, create_data_path
from analyser.plugins import Plugin
from analyser.utils import VideoDecoder


default_config = {"data_dir": "/data"}


default_parameters = {"fps": 1.0, "max_dimension": 128}

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
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters):

        video_decoder = VideoDecoder(
            path=inputs["video"].path, fps=parameters.get("fps"), max_dimension=parameters.get("max_dimension")
        )

        images = []
        for frame in video_decoder:
            image_id = generate_id()
            output_path = create_data_path(self.config.get("data_dir"), image_id, "jpg")
            imageio.imwrite(output_path, frame.get("frame"))
            images.append(ImageData(id=image_id, ext="jpg", time=frame.get("time")))
        data = ImagesData(images=images)
        return {"images": data}
