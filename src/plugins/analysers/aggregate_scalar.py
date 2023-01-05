from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ScalarData, ListData

import logging
import numpy as np


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"aggregation": "prod"}

requires = {
    "timelines": ListData,
}

provides = {
    "probs": ScalarData,
}


@AnalyserPluginManager.export("aggregate_scalar")
class AggregateScalar(
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

    def aggregate_probs(self, probs, times, interp_time, aggregation="prod"):
        probs_interp = []
        for i in range(len(probs)):
            prob_interp = np.interp(x=interp_time, xp=times[i], fp=probs[i])
            probs_interp.append(prob_interp)

        probs_interp = np.stack(probs_interp, axis=0)

        if aggregation == "mean":
            return np.mean(probs_interp, axis=0)
        if aggregation == "prod":
            return np.prod(probs_interp, axis=0)
        if aggregation == "or":
            return 1 - np.prod(1 - probs_interp, axis=0)
        if aggregation == "and":
            return np.prod(probs_interp, axis=0)
        else:
            logging.error("Unknown aggregation method. Using <mean> instead.")

        return np.mean(probs_interp, axis=0)

    def call(self, inputs, parameters, callbacks=None):
        probs = []
        times = []
        longest_timeline = 0
        for data in inputs["timelines"].data:
            probs.append(data.y)
            times.append(data.time)

            if len(data.time) > longest_timeline:
                longest_timeline = len(data.time)
                interp_time = data.time
                interp_delta_time = data.delta_time

        aggregated_probs = self.aggregate_probs(probs, times, interp_time, aggregation=parameters.get("aggregation"))
        self.update_callbacks(callbacks, progress=1.0)
        return {"probs": ScalarData(y=aggregated_probs.squeeze(), time=interp_time, delta_time=interp_delta_time)}
