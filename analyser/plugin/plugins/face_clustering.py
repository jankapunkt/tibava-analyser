from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ImageEmbeddings, ImageEmbedding, FaceClusterData, Cluster

import logging
import numpy as np
from scipy.spatial.distance import cdist
from analyser.data import DataManager, Data
from scipy.cluster.hierarchy import fclusterdata

from typing import Callable, Optional, Dict


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"min_threshold": None, "max_threshold": None, "cluster_threshold": 0.5}

requires = {
    "embeddings": ImageEmbeddings,
}

provides = {
    "face_cluster_data": FaceClusterData,
}


@AnalyserPluginManager.export("face_clustering")
class FaceClustering(
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
        with inputs["embeddings"] as face_embeddings,\
                inputs["faces"] as faces,\
                inputs["bboxes"] as bboxes, \
                inputs["kpss"] as kpss,\
                inputs["images"] as images,\
                data_manager.create_data("FaceClusterData") as output_data,\
                data_manager.create_data("ImageEmbeddings") as mean_embeddings:

            embeddings = [em.embedding for em in face_embeddings.embeddings]
            face_ids = [f.id for f in faces.faces]

            metric="cosine"
            result = fclusterdata(X=embeddings, t=parameters.get("cluster_threshold"), criterion="distance", metric=metric)
            # result format: list of cluster ids [1 2 1 3]
            
            clustered_embeddings = [[] for _ in np.unique(result)]
            output_data.clusters = [Cluster() for _ in np.unique(result)]
            
            
            for c in output_data.clusters:
                c.face_refs = []

            # sort face refs into clusters
            for id, cluster_id in enumerate(result):
                output_data.clusters[cluster_id-1].face_refs.append(face_ids[id])
                clustered_embeddings[cluster_id-1].append(embeddings[id])


            # compute mean embedding for each cluster
            for id, embedding_cluster in enumerate(clustered_embeddings):
                img_emb = ImageEmbedding(embedding=np.mean(embedding_cluster, axis=0).tolist())
                mean_embeddings.embeddings.append(img_emb)

            # sort clusters and embeddings together by cluster length
            zipped = sorted(zip(output_data.clusters, mean_embeddings.embeddings), key=lambda cluster: (len(cluster[0].face_refs)), reverse=True)
            unzipped_clusters, unzipped_embeddings = zip(*zipped)
            output_data.clusters = list(unzipped_clusters)
            mean_embeddings.embeddings = list(unzipped_embeddings)

            output_data.faces = faces
            output_data.bboxes = bboxes
            output_data.kpss = kpss
            output_data.images = images
         
            return {"face_cluster_data": output_data, "mean_embeddings": mean_embeddings}
