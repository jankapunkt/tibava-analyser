from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import AudioData, ImagesData
from analyser.plugins import Plugin
import ffmpeg
import os

default_config = {"data_dir": "/data/"}


default_parameters = {}

requires = {
    "images": ImagesData,
}

provides = {
    # "images_embedding": ImagesEmbedding,
}


@AnalyserPluginManager.export("clip_embedding")
class VideoToAudio(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters):

        output_data = AudioData(ext="mp3", data_dir=self.config.get("data_dir"))

        video = ffmpeg.input(inputs["video"].path)
        audio = video.audio
        stream = ffmpeg.output(audio, output_data.path)
        ffmpeg.run(stream)

        return {"audio": output_data}
