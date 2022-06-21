from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import AudioData, ScalarData
from analyser.plugins import Plugin
import librosa
import numpy as np

default_config = {"data_dir": "/data/"}


default_parameters = {
    "sr": 8000,
    "max_samples": 50000,
    "normalize": True,
}

requires = {
    "audio": AudioData,
}

provides = {
    "amp": ScalarData,
}


@AnalyserPluginManager.export("audio_amp_analysis")
class AudioAmpAnalysis(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters):

        y, sr = librosa.load(inputs.get("audio").path, sr=parameters.get("sr"))
        if parameters.get("max_samples"):
            target_sr = sr / (len(y) / int(parameters.get("max_samples")))

            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        if parameters.get("normalize"):
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

        return {"amp": ScalarData(y=y, time=(np.arange(len(y)) / sr).tolist(), delta_time=1 / sr)}
