from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import AudioData, VideoData
import ffmpeg
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict

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
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["video"] as input_data, data_manager.create_data("AudioData") as output_data:
            output_data.ext = "wav"

            with input_data.open_video() as f_video, output_data.open_audio("w") as f_audio:

                process = (
                    ffmpeg.input("pipe:")
                    .audio.output("pipe:", format="wav")
                    .run_async(pipe_stdin=True, pipe_stdout=True)
                )
                outputs, _ = process.communicate(input=f_video.read())
                f_audio.write(outputs)

            return {"audio": output_data}
