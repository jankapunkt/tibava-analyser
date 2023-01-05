from subprocess import call
from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import AudioData, VideoData
import ffmpeg
import os

default_config = {"data_dir": "/data/"}


default_parameters = {}

requires = {
    "video": VideoData,
}

provides = {
    "video": VideoData,
}


@AnalyserPluginManager.export("video_to_video")
class VideoToVideo(
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

        output_data = VideoData(ext="mp4", data_dir=self.config.get("data_dir"))

        video = ffmpeg.input(inputs["video"].path)

        stream = ffmpeg.output(
            video.video, video.audio, output_data.pat, preset="faster", ac=2, vcodec="libx264", acodec="aac"
        )

        ffmpeg.run(stream)

        self.update_callbacks(callbacks, progress=1.0)
        return {"audio": output_data}
