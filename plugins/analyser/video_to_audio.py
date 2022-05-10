from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import AudioData, VideoData
from analyser.plugins import Plugin
import ffmpeg

default_config = {}


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
        print("AA")

    def call(self, inputs):

        print("BB")
        video = None
        for key, data in inputs.items():
            print(key)

        video = ffmpeg.input(video_file)
        audio = video.audio
        stream = ffmpeg.output(audio, audio_file)
        ffmpeg.run(stream)

        pass
