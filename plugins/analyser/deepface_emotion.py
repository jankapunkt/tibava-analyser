from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import ListData, ScalarData, ImagesData, generate_id
from analyser.plugins import Plugin
import cv2
import imageio
import redisai as rai
import ml2rt
import numpy as np

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "deepface_emotion",
    "model_device": "cpu",
    "model_file": "/models/deepface_emotion/facial_expression_model.onnx",
    "grayscale": True,
    "target_size": (48, 48),
}

default_parameters = {"threshold": 0.5, "reduction": "max"}

requires = {
    "images": ImagesData,
}

provides = {
    "probs": ListData,
}


@AnalyserPluginManager.export("deepface_emotion")
class DeepfaceEmotion(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]

        self.grayscale = self.config["grayscale"]
        self.target_size = self.config["target_size"]

        self.con = rai.Client(host=self.host, port=self.port)

        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name,
            backend="onnx",
            device=self.model_device,
            data=model,
            inputs=["input"],
            outputs=["dense_2"],
            batch=16,
        )

    def preprocess(self, img_path):
        # read image
        img = imageio.imread(img_path)

        # post-processing
        if self.grayscale == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize image to expected shape
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = self.target_size[0] / img.shape[0]
            factor_1 = self.target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = self.target_size[0] - img.shape[0]
            diff_1 = self.target_size[1] - img.shape[1]
            if self.grayscale == False:
                # Put the base image in the middle of the padded image
                img = np.pad(
                    img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "constant"
                )
            else:
                img = np.pad(
                    img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), "constant"
                )

        if img.shape[0:2] != self.target_size:
            img = cv2.resize(img, self.target_size)

        # normalizing the image pixels
        img_pixels = np.asarray(img, np.float32)  # TODO same as: keras.preprocessing.image.img_to_array(img)?
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]

        if len(img_pixels.shape) == 3:  # RGB dimension missing
            img_pixels = np.expand_dims(img_pixels, axis=-1)

        return img_pixels

    def call(self, inputs, parameters):
        predictions_dict = {}
        # time = []
        delta_time = None
        for entry in inputs["images"].images:
            job_id = generate_id()
            image = self.preprocess(entry.path)

            self.con.tensorset(f"data_{job_id}", image)

            # modelname, input, output
            _ = self.con.modelrun(self.model_name, f"data_{job_id}", f"prob_{job_id}")
            prediction = self.con.tensorget(f"prob_{job_id}")[0]

            if entry.time not in predictions_dict:
                predictions_dict[entry.time] = []

            predictions_dict[entry.time].append(prediction)
            delta_time = entry.delta_time

        if parameters.get("reduction") == "max":
            predictions = [np.max(np.stack(x, axis=0), axis=0).tolist() for x in predictions_dict.values()]
        else:  # parameters.get("reduction") == "mean":
            predictions = [np.mean(np.stack(x, axis=0), axis=0).tolist() for x in predictions_dict.values()]

        probs = ListData(
            data=[
                ScalarData(y=y, time=list(predictions_dict.keys()), delta_time=delta_time) for y in zip(*predictions)
            ],
            index=["p_angry", "p_disgust", "p_fear", "p_happy", "p_sad", "p_surprise", "p_neutral"],
        )

        return {"probs": probs}
