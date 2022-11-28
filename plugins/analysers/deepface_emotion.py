from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ListData, ScalarData, ImagesData, generate_id
from analyser.utils import InferenceServer, Backend, Device
import cv2
import imageio
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

requires = {"images": ImagesData, "faces": ListData}

provides = {
    "probs": ListData,
}


@AnalyserPluginManager.export("deepface_emotion")
class DeepfaceEmotion(
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

        self.grayscale = self.config["grayscale"]
        self.target_size = self.config["target_size"]

        self.server = InferenceServer(
            model_file=self.model_file,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            backend=Backend.ONNX,
            device=self.model_device,
            inputs=["input"],
            outputs=["dense_2"],
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

    def call(self, inputs, parameters, callbacks=None):
        time = []
        ref_ids = []
        predictions = []

        faceid_lut = {}
        for face in inputs["faces"].faces:
            faceid_lut[face.img_id] = face.id

        for i, entry in enumerate(inputs["images"].images):

            self.update_callbacks(callbacks, progress=i / len(inputs["images"].images))
            image = self.preprocess(entry.path)

            result = self.server({"data": image}, ["prob"])
            prediction = result.get(f"prob")[0] if result else None
            face_id = faceid_lut[entry.id] if entry.id in faceid_lut else None

            time.append(entry.time)
            ref_ids.append(face_id)
            predictions.append(prediction.tolist())
            delta_time = entry.delta_time  # same for all examples

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "probs": ListData(
                data=[
                    ScalarData(y=np.asarray(y), time=time, delta_time=delta_time, ref_id=ref_ids)
                    for y in zip(*predictions)
                ],
                index=["p_angry", "p_disgust", "p_fear", "p_happy", "p_sad", "p_surprise", "p_neutral"],
            )
        }
