from ..inference import Device, Backend, generate_id, InferenceServer
import logging

import base64
import time
from typing import AnyStr, Union, List, Dict
from numpy.typing import NDArray

import numpy as np

def numpy_to_dict(nd_array:NDArray):
    return {
        "data": base64.b64encode(nd_array).decode("utf-8"),
        "dtype": str(nd_array.dtype),
        "shape": nd_array.shape,
    }

def dict_to_numpy(data:Dict):
    if "data" not in data or "dtype" not in data or "shape" not in data:
        return 
    try:

        return np.frombuffer(base64.decodebytes(data["data"].encode()), dtype=data["dtype"]).reshape(data["shape"])
        # return np.frombuffer(base64.decodebytes(data["data"]).encode(), dtype=data["dtype"], shape=data["shape"])
    except Exception as e:
        logging.error(f"[BentoMLInferenceServer] dict_to_numpy {e}")


try:
    import requests

    import json
    import re

    @InferenceServer.export("bentoml")
    class BentoMLInferenceServer:
        def __init__(self, config: Dict = None) -> None:
            if config is None:
                logging.error("No config provided for bentoml interface")
                return
            self.host = config.get("host", "localhost")
            self.port = config.get("port", 3000)
            self.service = config.get("service")

        def __call__(self, inputs: Dict, outputs: List):
            start_time = time.time()
            transformer_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, np.ndarray):
                    logging.debug(f"{k}: {v.shape}", flush=True)
                    # v = json.dumps(v.tolist())
                    v = numpy_to_dict(v)
                # else:
                #     v = str(v)
                transformer_inputs[k] = v

            data = json.dumps(transformer_inputs)
            
            raw_output = requests.post(
                f"http://{self.host}:{self.port}/{self.service}",
                headers={"content-type": "application/json"},
                data=data,
            ).json()
            
            start_time = time.time()
            
            output_dict = {}
            for x in outputs:
                if x not in raw_output:
                    logging.error("Unknown output field {x}")
                    return None
                v = dict_to_numpy(raw_output[x])
                if v is not None:
                    output_dict[x] = v
                else:
                    output_dict[x] = raw_output[x]

            for k, v in output_dict.items():
                logging.debug(f"{k} {v.shape}")

            logging.debug(f"DECODER {time.time() - start_time}")
            return output_dict

except:
    logging.warning("BentoML not available")
