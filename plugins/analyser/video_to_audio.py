from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import AudioData, VideoData
from analyser.plugins import Plugin
import ffmpeg
import os

default_config = {"data_dir": "/data/"}


default_parameters = {}

requires = {
    "video": VideoData,
}

provides = {
    "audio": AudioData,
}


@AnalyserPluginManager.export("video_to_audio")
class VideoToAudio(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs):

        output_data = AudioData(ext="mp3")
        output_data.path = os.path.join(self.config.get("data_dir"), f"{output_data.id}.mp3")

        video = ffmpeg.input(inputs["video"].path)
        audio = video.audio
        stream = ffmpeg.output(audio, output_data.path)
        ffmpeg.run(stream)

        return {"audio": output_data}
