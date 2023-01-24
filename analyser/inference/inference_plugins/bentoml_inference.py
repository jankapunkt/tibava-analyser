from ..inference import Device, Backend, generate_id, InferenceServer
import logging

import time
from typing import AnyStr, Union, List, Dict

import numpy as np
import requests
import json


@InferenceServer.export("bentoml")
class BentoMLInferenceServer:
    def __init__(self, config: Dict = None) -> None:
        if config is None:
            logging.error("No config provided for bentoml interface")
            return
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 3000)
        self.service = config.get("service", 3000)

    def __call__(self, inputs: Dict, outputs: List):
        transformer_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}", flush=True)
                v = v.tolist()
            transformer_inputs[k] = v

        data = json.dumps(transformer_inputs)
        # print(data)
        # print(f"http://{self.host}:{self.port}/{self.service}")
        raw_output = requests.post(
            f"http://{self.host}:{self.port}/{self.service}",
            headers={"content-type": "application/json"},
            data=data,
        ).json()

        # print(f"{raw_output}", flush=True)
        output_dict = {}
        for x in outputs:
            if x not in raw_output:
                logging.error(f"Unknown output field {x}")
                return None
            try:
                output_dict[x] = np.asarray(raw_output[x])
            except:
                output_dict[x] = raw_output[x]

        # print(f"{output_dict}", flush=True)
        return output_dict
