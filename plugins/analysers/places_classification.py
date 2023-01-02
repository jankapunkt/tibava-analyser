from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder, image_pad
from analyser.data import ListData, ScalarData, VideoData, ListData, ImageEmbedding, ImageEmbeddings, generate_id


from analyser.inference import InferenceServer

import csv
import numpy as np
from sklearn.preprocessing import normalize


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "places_classifier",
    "model_device": "cpu",
    "model_file": "/models/places_classification/resnet50_places365.pt",
    "classes_file": "/models/places_classification/categories_places365.txt",
    "hierarchy_file": "/models/places_classification/scene_hierarchy_places365.csv",
    "image_resolution": 224,
}

default_parameters = {"fps": 5}

requires = {
    "video": VideoData,
}

provides = {
    "embedding": ImageEmbeddings,
    "probs_places365": ListData,
    "probs_places16": ListData,
    "probs_places3": ListData,
}


@AnalyserPluginManager.export("places_classifier")
class PlacesClassifier(
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
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.image_resolution = self.config["image_resolution"]

        self.classes = self.read_classes(self.config["classes_file"], self.config["hierarchy_file"])
        self.hierarchy = self.read_hierarchy(self.config["hierarchy_file"])

        self.server = InferenceServer(
            model_file=self.model_file,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            device=self.model_device,
        )

    def read_classes(self, classes_file, hierarchy_file):
        classes = {"places365": [], "places16": [], "places3": []}

        with open(classes_file, "r") as csvfile:
            content = csv.reader(csvfile, delimiter=" ")
            for line in content:
                classes["places365"].append(line[0])

        with open(hierarchy_file, "r") as csvfile:
            content = csv.reader(csvfile, delimiter=",")
            next(content)  # skip explanation line
            hierarchy_labels = next(content)  # second row contains hierarchy labels

            classes["places3"] = hierarchy_labels[1:4]
            classes["places16"] = hierarchy_labels[4:]

        return classes

    def read_hierarchy(self, hierarchy_file):
        hierarchy_places3 = []
        hierarchy_places16 = []
        with open(hierarchy_file, "r") as csvfile:
            content = csv.reader(csvfile, delimiter=",")
            next(content)  # skip explanation line
            next(content)  # second row contains hierarchy labels

            for line in content:
                hierarchy_places3.append(line[1:4])
                hierarchy_places16.append(line[4:])

        # normalize label if it belongs to multiple categories
        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=np.float)
        hierarchy_places3 /= np.expand_dims(np.sum(hierarchy_places3, axis=1), axis=-1)

        hierarchy_places16 = np.asarray(hierarchy_places16, dtype=np.float)
        hierarchy_places16 /= np.expand_dims(np.sum(hierarchy_places16, axis=1), axis=-1)

        return {"places3": hierarchy_places3, "places16": hierarchy_places16}

    def call(self, inputs, parameters, callbacks=None):
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=self.image_resolution, fps=parameters.get("fps")
        )

        embeddings = []
        probs = {"places365": [], "places16": [], "places3": []}
        time = []
        num_frames = video_decoder.duration() * video_decoder.fps()
        for i, frame in enumerate(video_decoder):
            result = self.server({"data": image_pad(frame["frame"])}, ["embedding", "prob"])
            if result is not None:
                # store embeddings
                embeddings.append(
                    ImageEmbedding(
                        embedding=normalize(result["embedding"]),
                        image_id=generate_id(),
                        time=frame.get("time"),
                        delta_time=1 / parameters.get("fps"),
                    )
                )

                # store places365 probabilities
                prob = result["prob"]
                probs["places365"].append(np.squeeze(np.asarray(prob)))

                # store places16 probabilities
                probs["places16"].append(np.matmul(prob, self.hierarchy["places16"])[0])

                # store places3 probabilities
                probs["places3"].append(np.matmul(prob, self.hierarchy["places3"])[0])

                # store time
                time.append(i / parameters.get("fps"))

            self.update_callbacks(callbacks, progress=i / num_frames)

        probs_grpc = {}
        for level in probs.keys():
            probs_grpc[level] = ListData(
                data=[
                    ScalarData(y=np.asarray(y), time=time, delta_time=1 / parameters.get("fps"))
                    for y in zip(*probs[level])
                ],
                index=self.classes[level],
            )

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "embeddings": ImageEmbeddings(embeddings=embeddings),
            "probs_places365": probs_grpc["places365"],
            "probs_places16": probs_grpc["places16"],
            "probs_places3": probs_grpc["places3"],
        }
