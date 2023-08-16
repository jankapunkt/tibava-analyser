from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ScalarData, ImageEmbeddings

import logging
import numpy as np
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict


default_config = {
    "data_dir": "/data/",
}

default_parameters = {
    "aggregation": "max",
    "normalize": 0,
    "normalize_min_val": None,
    "normalize_max_val": None,
    "index": None,
    "cluster_id": None,
}

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
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        from scipy.spatial.distance import cdist

        with inputs["query_features"] as query_features_data, inputs[
            "target_features"
        ] as target_features_data, data_manager.create_data(
            "ScalarData"
        ) as output_data:
            if parameters.get("index") == None:
                query_features = query_features_data.embeddings
                qfs = [qf.embedding for qf in query_features]
            else:
                cluster_id = parameters.get("cluster_id")
                index = parameters.get("index").split(",")
                cluster_repr = [
                    c.embedding_repr
                    for c in query_features_data.clusters
                    if c.id == cluster_id
                ][0]
                cluster_embeddings = []
                for i in index:
                    cluster_embeddings.append(cluster_repr[int(i)])

                qfs = [np.mean(cluster_embeddings, axis=0)]

            target_features = target_features_data.embeddings

            unique_times = set()
            times = []
            tfs = []
            for tf in target_features:
                unique_times.add(tf.time)
                times.append(tf.time)
                tfs.append(tf.embedding)
                delta_time = tf.delta_time

            unique_times = sorted(unique_times)

            # qfs = [qf.embedding for qf in query_features]

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
                if parameters.get("normalize_min_val") and parameters.get(
                    "normalize_max_val"
                ):
                    cossim_t = (cossim_t - parameters.get("normalize_min_val")) / (
                        parameters.get("normalize_max_val")
                        - parameters.get("normalize_min_val")
                    )
                else:
                    cossim_t = (cossim_t - np.min(cossim_t)) / (
                        np.max(cossim_t) - np.min(cossim_t)
                    )

            output_data.y = np.squeeze(cossim_t)
            output_data.time = list(unique_times)
            output_data.delta_time = delta_time

            return {"probs": output_data}
