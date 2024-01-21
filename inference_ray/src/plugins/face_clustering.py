from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager

from analyser.data import ImageEmbeddings, FaceClusterData, Cluster, FacesData, ImagesData, KpssData, BboxesData

import logging
import numpy as np
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {
    "min_threshold": None,
    "max_threshold": None,
    "cluster_threshold": 0.5,
    "max_cluster": 2,
    "max_faces": 3,
}

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
    version="0.3",
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

        with inputs["embeddings"] as face_embeddings,\
            inputs["faces"] as faces,\
            inputs["bboxes"] as bboxes,\
            inputs["kpss"] as kpss,\
            inputs["images"] as images,\
            data_manager.create_data("FaceClusterData") as output_data:
            
            embeddings = np.asarray([em.embedding for em in face_embeddings.embeddings])
            face_ids = [f.id for f in faces.faces]

            metric = "cosine"
            result = fclusterdata(
                X=embeddings,
                t=parameters.get("cluster_threshold"),
                criterion="distance",
                metric=metric,
            )
            # result format: list of cluster ids [1 2 1 3]
            clusters = []
            for x in np.unique(result):
                logging.error(f"######################")
                
                ids = np.where(result == x)[0]

                cluster_size = len(ids)

                logging.error(f"{cluster_size} {ids}")
                ids_ids= np.linspace(0, cluster_size-1, min(cluster_size, parameters.get("max_faces")))
                logging.error(f"a {ids_ids}")
                ids_ids = [round(idx) for idx in ids_ids]
                logging.error(f"b {ids_ids}")
                
                logging.error(f"old {ids}")
                ids = ids[ids_ids]
                logging.error(f"new {ids}")
                cluster_embeddings = embeddings[ids]
                cluster_faces = [faces.faces[id] for id in ids]
                cluster_bboxes = [bboxes.bboxes[id] for id in ids]
                cluster_kpss = [kpss.kpss[id] for id in ids]
                cluster_images = [images.images[id] for id in ids]
                logging.error(cluster_faces)

                clusters.append({
                    "size": cluster_size,
                    "ids": ids,
                    "embeddings": cluster_embeddings,
                    "faces": cluster_faces,
                    "bboxes": cluster_bboxes,
                    "kpss": cluster_kpss,
                    "images": cluster_images,
                })

            clusters = sorted(clusters, key= lambda x: x["size"], reverse=True)[:parameters.get("max_cluster")]
            logging.error(f"cluster {len(clusters)}")
            logging.error(f"cluster {[x['size'] for x in clusters]}")
            

            # clustered_embeddings = [[] for _ in np.unique(result)]
            # output_data.clusters = [Cluster() for _ in np.unique(result)]

            # for c in output_data.clusters:
            #     c.object_refs = []

            # # sort face refs into clusters
            # for id, cluster_id in enumerate(result):
            #     output_data.clusters[cluster_id - 1].object_refs.append(face_ids[id])
            #     clustered_embeddings[cluster_id - 1].append(embeddings[id])

            # # compute mean embedding for each cluster
            # for id, embedding_cluster in enumerate(clustered_embeddings):
            #     converted_clusters = [x for x in embedding_cluster]
            #     output_data.clusters[id].embedding_repr = converted_clusters

            # # sort clusters and embeddings together by cluster length
            # output_data.clusters = sorted(
            #     output_data.clusters,
            #     key=lambda cluster: (len(cluster.object_refs)),
            #     reverse=True,
            # )

            output_data.clusters = [Cluster(embedding_repr=cluster["embeddings"], object_refs=[y.id for y in cluster["faces"]]) for cluster in clusters]

            output_data.faces =  FacesData(faces= [x for cluster in clusters for x in cluster["faces"]])
            output_data.bboxes =  BboxesData(bboxes=[x for cluster in clusters for x in cluster["bboxes"]])
            output_data.kpss =  KpssData(kpss=[x for cluster in clusters for x in cluster["kpss"]])
            output_data.images =  ImagesData(images=[x for cluster in clusters for x in cluster["images"]])


            return {"face_cluster_data": output_data}
