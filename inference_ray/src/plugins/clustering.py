import logging
import numpy as np
from typing import Callable, Optional, Dict

from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.data import DataManager, Data, ImageEmbeddings, ClusterData, Cluster


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {
    "cluster_threshold": 0.5,
}

requires = {
    "embeddings": ImageEmbeddings,
}

provides = {
    "cluster_data": ClusterData,
}


@AnalyserPluginManager.export("clustering")
class Clustering(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        from scipy.cluster.hierarchy import fclusterdata

        with inputs["embeddings"] as embeddings, data_manager.create_data(
            "ClusterData"
        ) as output_data:
            np_embeddings = np.asarray([em.embedding for em in embeddings.embeddings])

            metric = "cosine"
            result = fclusterdata(
                X=np_embeddings,
                t=parameters.get("cluster_threshold"),
                criterion="distance",
                metric=metric,
            )
            # result format: list of cluster ids [1 2 1 3]
            clusters = []
            for x in np.unique(result):
                ids = np.where(result == x)[0]
                clusters.append([embeddings.embeddings[id].id for id in ids])

            clusters = sorted(clusters, key=lambda x: len(x), reverse=True)

            output_data.clusters = [
                Cluster(object_refs=cluster) for cluster in clusters
            ]

            return {"cluster_data": output_data}
