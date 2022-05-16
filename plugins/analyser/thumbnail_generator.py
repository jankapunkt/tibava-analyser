import os

from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import ImageData, VideoData, ShotsData
from analyser.plugins import Plugin


default_config = {"data_dir": "/data"}


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
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters):
        print(parameters)

        output_data = ImageData(ext="mp3")

        output_data.path = os.path.join(self.config.get("data_dir"), f"{output_data.id}.mp3")

        video = ffmpeg.input(inputs["video"].path)
        audio = video.audio
        stream = ffmpeg.output(audio, output_data.path)
        ffmpeg.run(stream)

        return {"audio": output_data}
