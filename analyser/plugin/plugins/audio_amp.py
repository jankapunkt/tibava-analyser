from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import AudioData, ScalarData
import librosa
import numpy as np
import logging

default_config = {"data_dir": "/data/"}


default_parameters = {
    "sr": 8000,
    "max_samples": 200000,
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

        y, sr = librosa.load(inputs.get("audio").path, sr=parameters.get("sr"))

        if parameters.get("max_samples"):
            target_sr = sr / (len(y) / int(parameters.get("max_samples")))
            try:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception as e:
                logging.warning("Resampling failed. Try numpy.")
                t = np.arange(y.shape[0]) / sr
                t_target = np.arange(int(y.shape[0] / sr * target_sr)) / target_sr

                y = np.interp(t_target, t, y)
                sr = target_sr

        if parameters.get("normalize"):
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

        return {"amp": ScalarData(y=y, time=(np.arange(len(y)) / sr).tolist(), delta_time=1 / sr)}
