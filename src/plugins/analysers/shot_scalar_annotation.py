from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager

from analyser.data import Annotation, AnnotationData, ShotsData, ScalarData, ListData, generate_id

import numpy as np
import math


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "shot_type_classifier",
}

default_parameters = {}

requires = {
    "shots": ShotsData,
    "scalar": ScalarData,
}

provides = {
    "annotations": AnnotationData,
}


@AnalyserPluginManager.export("shot_scalar_annotator")
class ShotScalarAnnotator(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def call(self, inputs, parameters, callbacks=None):
        annotations = []

        y = np.asarray(inputs["scalar"].y)
        time = np.asarray(inputs["scalar"].time)
        for i, shot in enumerate(inputs["shots"].shots):
            shot_y_data = y[np.logical_and(time >= shot.start, time <= shot.end)]

            if len(shot_y_data) <= 0:
                continue

            y_mean = np.mean(shot_y_data)
            annotations.append(
                Annotation(start=shot.start, end=shot.end, labels=[str(y_mean)])
            )  # Maybe store max_mean_class_prob as well?
            self.update_callbacks(callbacks, progress=i / len(inputs["shots"].shots))

        self.update_callbacks(callbacks, progress=1.0)
        return {"annotations": AnnotationData(annotations=annotations)}
