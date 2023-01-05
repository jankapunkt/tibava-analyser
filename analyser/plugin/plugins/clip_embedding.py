import gzip
import os
import html
import ftfy
import regex as re
import numpy as np

import scipy
from scipy import spatial
from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import (
    VideoData,
    StringData,
    ScalarData,
    AnnotationData,
    ImageEmbedding,
    TextEmbedding,
    ImageEmbeddings,
    TextEmbeddings,
    generate_id,
)

from analyser.inference import InferenceServer
from analyser.utils import VideoDecoder
from analyser.utils.imageops import image_resize, image_crop, image_pad
from PIL import Image
from functools import lru_cache
from cv2 import cvtColor, COLOR_BGR2RGB
from typing import Union, List

from sklearn.preprocessing import normalize
import imageio


default_config = {
    "data_dir": "/data/",
}


img_embd_parameters = {
    "fps": 2,
    "crop_size": [224, 224],
}

text_embd_parameters = {
    "search_term": "",
}

prob_parameters = {
    "search_term": "",
}

anno_parameters = {
    "threshold": 0.5,
}

img_embd_requires = {
    "video": VideoData,
}

text_embd_requires = {}


img_embd_provides = {
    "embeddings": ImageEmbeddings,
}

text_embd_provides = {
    "embeddings": TextEmbeddings,
}


@AnalyserPluginManager.export("clip_image_embedding")
class ClipImageEmbedding(
    AnalyserPlugin,
    config=default_config,
    parameters=img_embd_parameters,
    version="0.1",
    requires=img_embd_requires,
    provides=img_embd_provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        inference_config = self.config.get("inference", None)
        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def preprocess(self, img, resize_size, crop_size):
        converted = image_resize(image_pad(img), size=crop_size)
        return converted

    def call(self, inputs, parameters, callbacks=None):
        preds = []
        video_decoder = VideoDecoder(path=inputs["video"].path, fps=parameters.get("fps"))
        num_frames = video_decoder.duration() * video_decoder.fps()
        for i, frame in enumerate(video_decoder):
            self.update_callbacks(callbacks, progress=i / num_frames)
            img_id = generate_id()
            img = frame.get("frame")
            img = self.preprocess(img, parameters.get("resize_size"), parameters.get("crop_size"))
            imageio.imwrite(os.path.join(self.config.get("data_dir"), f"test_{i}.jpg"), img)
            result = self.server({"data": np.expand_dims(img, axis=0)}, ["embedding"])
            preds.append(
                ImageEmbedding(
                    embedding=normalize(result["embedding"]),
                    image_id=img_id,
                    time=frame.get("time"),
                    delta_time=1 / parameters.get("fps"),
                )
            )

        self.update_callbacks(callbacks, progress=1.0)
        return {"embeddings": ImageEmbeddings(embeddings=preds)}


@AnalyserPluginManager.export("clip_text_embedding")
class ClipTextEmbedding(
    AnalyserPlugin,
    config=default_config,
    parameters=text_embd_parameters,
    version="0.1",
    requires=text_embd_requires,
    provides=text_embd_provides,
):
    def __init__(self, config=None):
        super().__init__(config)

        inference_config = self.config.get("inference", None)
        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def preprocess(self, text):
        # tokenize text

        tokenized = self.tokenizer.tokenize(text)
        return tokenized

    def call(self, inputs, parameters, callbacks=None):
        text_id = generate_id()
        # text = self.preprocess(parameters["search_term"])
        result = self.server({"data": parameters["search_term"]}, ["embedding"])

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "embeddings": TextEmbeddings(
                embeddings=[
                    TextEmbedding(text_id=text_id, text=parameters["search_term"], embedding=result["embedding"][0])
                ]
            )
        }


prob_requires = {
    "embeddings": ImageEmbeddings,
}

prob_provides = {
    "probs": ScalarData,
}


@AnalyserPluginManager.export("clip_probs")
class ClipProbs(
    AnalyserPlugin,
    config=default_config,
    parameters=prob_parameters,
    version="0.1",
    requires=prob_requires,
    provides=prob_provides,
):
    def __init__(self, config=None):
        super().__init__(config)

    def preprocess(self, text):
        # tokenize text
        tokenized = self.tokenizer.tokenize(text)
        return tokenized

    def call(self, inputs, parameters, callbacks=None):
        probs = []
        time = []
        delta_time = None
        embeddings = inputs["embeddings"]
        text = self.preprocess(parameters["search_term"])
        result = self.text_server({"data": text}, ["o"])

        text_embedding = normalize(result["o"])

        neg_text = self.preprocess("Not " + parameters["search_term"])
        neg_result = self.text_server({"data": neg_text}, ["o"])

        neg_text_embedding = normalize(neg_result["o"])

        text_embedding = np.concatenate([text_embedding, neg_text_embedding], axis=0)
        for embedding in embeddings.embeddings:

            result = 100 * text_embedding @ embedding.embedding.T

            prob = scipy.special.softmax(result, axis=0)

            # sim = 1 - spatial.distance.cosine(embedding.embedding, text_embedding)
            probs.append(prob[0, 0])
            time.append(embedding.time)
            delta_time = embedding.delta_time

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "probs": ScalarData(y=np.array(probs), time=time, delta_time=delta_time, name="image_text_similarities")
        }
