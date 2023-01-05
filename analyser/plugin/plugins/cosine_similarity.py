from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ScalarData, ImageEmbeddings

import logging
import numpy as np
from scipy.spatial.distance import cdist


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"aggregation": "max", "normalize": 0, "normalize_min_val": None, "normalize_max_val": None}

requires = {
    "query_features": ImageEmbeddings,
    "target_features": ImageEmbeddings,
}

provides = {
    "probs": ScalarData,
}


@AnalyserPluginManager.export("cosine_similarity")
class CosineSimilarity(
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

        query_features = inputs["query_features"].embeddings
        target_features = inputs["target_features"].embeddings

        unique_times = set()
        times = []
        tfs = []
        for tf in target_features:
            unique_times.add(tf.time)
            times.append(tf.time)
            tfs.append(tf.embedding)
            delta_time = tf.delta_time

        qfs = [qf.embedding for qf in query_features]

        tfs = np.asarray(tfs)
        qfs = np.asarray(qfs)

        cossim = 1 - cdist(tfs, qfs, "cosine")
        cossim = (cossim + 1) / 2

        # aggregation over available features at a given time t
        if parameters.get("aggregation") == "max":
            cossim = np.max(cossim, axis=-1)
        else:
            logging.error("Unknown aggregation method. Using max instead ...")
            cossim = np.max(cossim, axis=-1)

        # aggregation over time using max
        # NOTE: Its sufficient if a query feature vector match one vector at a specific time in the target video
        cossim_t = []
        for t in unique_times:
            cossim_t.append(np.max(cossim[np.asarray(times) == t]))

        unique_times = list(unique_times)
        cossim_t = np.squeeze(np.asarray(cossim_t))

        if parameters.get("normalize") > 0:
            if parameters.get("normalize_min_val") and parameters.get("normalize_max_val"):
                cossim_t = (cossim_t - parameters.get("normalize_min_val")) / (
                    parameters.get("normalize_max_val") - parameters.get("normalize_min_val")
                )
            else:
                cossim_t = (cossim_t - np.min(cossim_t)) / (np.max(cossim_t) - np.min(cossim_t))

        return {"probs": ScalarData(y=np.squeeze(cossim), time=list(unique_times), delta_time=delta_time)}
