from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ScalarData, ListData

import logging
import numpy as np


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"aggregation": "or"}

requires = {
    "timeline": ListData,
}

provides = {
    "aggregated_timeline": ListData,
}


@AnalyserPluginManager.export("aggregate_scalar_per_time")
class AggregateScalarPerTime(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def aggregate_probs(self, y_per_t: dict, aggregation: str = "or") -> list:
        def aggregate_mean(y: list) -> np.array:
            return np.mean(y, axis=0)

        def aggregate_prod(y: list) -> np.array:
            return np.prod(y, axis=0)

        def aggregate_or(y: list) -> np.array:
            return 1 - np.prod(1 - y, axis=0)

        def aggregate_and(y: list) -> np.array:
            return np.prod(y, axis=0)

        aggregation_f = {"mean": aggregate_mean, "prod": aggregate_prod, "or": aggregate_or, "and": aggregate_and}

        if aggregation not in aggregation_f:
            logging.error("Unknown aggregation method. Using <mean> instead.")
            aggregation = "mean"

        return [aggregation_f[aggregation](np.stack(y, axis=0)) for y in y_per_t.values()]

    def call(self, inputs: ListData, parameters: dict, callbacks=None) -> ListData:
        aggregated_y = []

        for i, data in enumerate(inputs["timeline"].data):

            y_per_t = {}
            for n in range(len(data.time)):
                if data.time[n] not in y_per_t:
                    y_per_t[data.time[n]] = []

                y_per_t[data.time[n]].append(data.y[n])

            aggregated_y.append(self.aggregate_probs(y_per_t, aggregation=parameters.get("aggregation")))
            self.update_callbacks(callbacks, progress=i / len(inputs["timeline"].data))

        self.update_callbacks(callbacks, progress=1.0)

        return {
            "aggregated_timeline": ListData(
                data=[
                    ScalarData(y=np.asarray(y), time=list(y_per_t.keys()), delta_time=data.delta_time)
                    for y in aggregated_y
                ],
                index=inputs["timeline"].index,
            )
        }
