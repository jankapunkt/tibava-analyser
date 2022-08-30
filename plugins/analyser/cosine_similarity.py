from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import ScalarData, ImageEmbeddings
from analyser.plugins import Plugin

import logging
import numpy as np
from scipy.spatial.distance import cdist


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"aggregation": "max"}

requires = {
    "query_features": ImageEmbeddings,
    "target_features": ImageEmbeddings,
}

provides = {
    "probs": ScalarData,
}


@AnalyserPluginManager.export("cosine_similarity")
class CosineSimilarity(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def call(self, inputs, parameters, callbacks=None):

        query_features = inputs["query_features"].embeddings
        target_features = inputs["target_features"].embeddings

        times = []
        tfs = []
        for tf in target_features:
            times.append(tf.time)
            tfs.append(tf.embedding)
            delta_time = tf.delta_time

        qfs = [qf.embedding for qf in query_features]

        tfs = np.asarray(tfs)
        qfs = np.asarray(qfs)

        cossim = 1 - cdist(tfs, qfs, "cosine")

        if parameters.get("aggregration") == "max":
            cossim = np.max(cossim, axis=-1)
        else:
            logging.error("Unknown aggregation method. Using max instead ...")
            cossim = np.max(cossim, axis=-1)

        print(np.squeeze(cossim))
        print(times)
        print(delta_time)
        return {"probs": ScalarData(y=np.squeeze(cossim), time=times, delta_time=delta_time)}
