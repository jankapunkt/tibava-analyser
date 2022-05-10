from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.plugins import Plugin, AudioData, VideoData
import ffmpeg

default_config = {}


default_parameters = {"fps": 1.0, "max_resolution": 128}

requires = {
    "video": VideoData,
}

provides = {
    "audio": AudioData,
}


@AnalyserPluginManager.export("video_to_audio")
class ThumbnailGenerator(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None, name=None):
        super().__init__(config, name)

    def call(self, inputs):

        video = None
        for key, data in inputs.items():
            print(key)

        video = ffmpeg.input(video_file)
        audio = video.audio
        stream = ffmpeg.output(audio, audio_file)
        ffmpeg.run(stream)

        pass
