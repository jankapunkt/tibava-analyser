from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.data import BboxesData, ListData, ScalarData

import numpy as np
import pickle
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_file": "/models/naivebayes_facesize/naivebayes_facesize.pkl",
}

default_parameters = {"reduction": "max"}

requires = {
    "bboxes": BboxesData,
}

provides = {
    "probs": ListData,
    "facesizes": ScalarData,
}


@AnalyserPluginManager.export("insightface_facesize")
class InsightfaceFacesize(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # self.host = self.config["host"]
        # self.port = self.config["port"]
        self.model = None

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        if self.model is None:
            with open(self.config["model_file"], "rb") as pklfile:
                self.model = pickle.load(pklfile)

        with inputs["bboxes"] as input_data, data_manager.create_data(
            "ListData"
        ) as probs_data, data_manager.create_data("ScalarData") as facesizes_data:
            facesizes_dict = {}
            delta_time = None

            for i, bbox in enumerate(input_data.bboxes):
                if bbox.time not in facesizes_dict:
                    facesizes_dict[bbox.time] = []
                facesizes_dict[bbox.time].append(bbox.w * bbox.h)
                delta_time = bbox.delta_time

            if parameters.get("reduction") == "max":
                facesizes = [np.max(x).tolist() for x in facesizes_dict.values()]
            else:  # parameters.get("reduction") == "mean":
                facesizes = [np.mean(x).tolist() for x in facesizes_dict.values()]

            # predict shot size based on facesizes
            predictions = self.model.predict_proba(np.asarray(facesizes).reshape(-1, 1))

            self.update_callbacks(callbacks, progress=1.0)

            facesizes_data.y = facesizes
            facesizes_data.time = list(facesizes_dict.keys())
            facesizes_data.delta_time = delta_time

            index = ["p_ECU", "p_CU", "p_MS", "p_FS", "p_LS"]
            for i, y in zip(index, zip(*predictions)):
                with probs_data.create_data("ScalarData", index=i) as d:
                    d.y = np.asarray(y)
                    d.time = list(facesizes_dict.keys())
                    d.delta_time = delta_time

            return {
                "probs": probs_data,
                "facesizes": facesizes_data,
            }
