from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import AudioData, ScalarData, HistData
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict
import librosa
import numpy as np

default_config = {"data_dir": "/data/"}


default_parameters = {"sr": 8000, "max_samples": 100000, "normalize": True, "n_fft": 256}

requires = {
    "audio": AudioData,
}

provides = {
    "freq": HistData,
}


@AnalyserPluginManager.export("audio_freq_analysis")
class AudioFreqAnalysis(
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

        y, sr = librosa.load(inputs.get("audio").path, sr=parameters.get("sr"))
        if parameters.get("max_samples"):
            target_sr = sr / (len(y) / int(parameters.get("max_samples")))

            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        S = np.abs(librosa.stft(y, n_fft=parameters.get("n_fft")))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        time = y.shape[0] / sr
        t_delta = time / S_db.shape[1]

        if parameters.get("normalize"):
            S_db = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))

        S_db = np.transpose(S_db)

        return {"freq": HistData(hist=S_db, time=(np.arange(S_db.shape[0]) * t_delta).tolist(), delta_time=1 / sr)}
