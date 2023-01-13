import os
import numpy as np

import scipy
from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import VideoData, ScalarData, ImageEmbedding, TextEmbedding, ImageEmbeddings, TextEmbeddings
from analyser.data import DataManager, Data

from typing import Callable, Optional, Dict

from analyser.inference import InferenceServer
from analyser.utils import VideoDecoder
from analyser.utils.imageops import image_resize, image_crop, image_pad

from sklearn.preprocessing import normalize
import imageio


default_config = {
    "data_dir": "/data/",
}


img_embd_parameters = {
    "fps": 2,
    "crop_size": [224, 224],
}


img_embd_requires = {
    "video": VideoData,
}

img_embd_provides = {
    "embeddings": ImageEmbeddings,
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
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        inference_config = self.config.get("inference", None)
        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def preprocess(self, img, resize_size, crop_size):
        converted = image_resize(image_pad(img), size=crop_size)
        return converted

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["video"] as input_data, data_manager.create_data("ImageEmbeddings") as output_data:
            with input_data.open_video("r") as f_video:
                video_decoder = VideoDecoder(f_video, fps=parameters.get("fps"), extension=f".{input_data.ext}")
                num_frames = video_decoder.duration() * video_decoder.fps()
                for i, frame in enumerate(video_decoder):
                    self.update_callbacks(callbacks, progress=i / num_frames)

                    img = frame.get("frame")
                    img = self.preprocess(img, parameters.get("resize_size"), parameters.get("crop_size"))
                    result = self.server({"data": np.expand_dims(img, axis=0)}, ["embedding"])
                    output_data.embeddings.append(
                        ImageEmbedding(
                            embedding=normalize(result["embedding"]),
                            time=frame.get("time"),
                            delta_time=1 / parameters.get("fps"),
                        )
                    )

                self.update_callbacks(callbacks, progress=1.0)
            return {"embeddings": output_data}


text_embd_parameters = {
    "search_term": "",
}

text_embd_requires = {}

text_embd_provides = {
    "embeddings": TextEmbeddings,
}


@AnalyserPluginManager.export("clip_text_embedding")
class ClipTextEmbedding(
    AnalyserPlugin,
    config=default_config,
    parameters=text_embd_parameters,
    version="0.1",
    requires=text_embd_requires,
    provides=text_embd_provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        inference_config = self.config.get("inference", None)
        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def preprocess(self, text):
        # tokenize text

        tokenized = self.tokenizer.tokenize(text)
        return tokenized

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with data_manager.create_data("TextEmbeddings") as output_data:
            # text = self.preprocess(parameters["search_term"])
            result = self.server({"data": parameters["search_term"]}, ["embedding"])
            output_data.embeddings.append(
                TextEmbedding(text=parameters["search_term"], embedding=result["embedding"][0])
            )
            self.update_callbacks(callbacks, progress=1.0)
            return {"embeddings": output_data}


prob_parameters = {
    "search_term": "",
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
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        inference_config = self.config.get("inference", None)
        self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["embeddings"] as input_data, data_manager.create_data("ScalarData") as output_data:
            probs = []
            time = []
            delta_time = None
            pos_text = parameters["search_term"]
            result = self.server({"data": pos_text}, ["embedding"])

            text_embedding = normalize(result["embedding"])

            neg_text = "Not " + parameters["search_term"]
            neg_result = self.server({"data": neg_text}, ["embedding"])

            neg_text_embedding = normalize(neg_result["embedding"])

            text_embedding = np.concatenate([text_embedding, neg_text_embedding], axis=0)
            for embedding in input_data.embeddings:

                result = 100 * text_embedding @ embedding.embedding.T

                prob = scipy.special.softmax(result, axis=0)

                # sim = 1 - spatial.distance.cosine(embedding.embedding, text_embedding)
                probs.append(prob[0, 0])
                time.append(embedding.time)
                delta_time = embedding.delta_time

            self.update_callbacks(callbacks, progress=1.0)
            output_data.y = np.array(probs)
            output_data.time = time
            output_data.delta_time = delta_time
            output_data.name = "image_text_similarities"
            return {"probs": output_data}
