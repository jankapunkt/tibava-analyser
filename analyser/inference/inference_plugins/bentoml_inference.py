from ..inference import Device, Backend, generate_id, InferenceServer
import logging

import time
from typing import AnyStr, Union, List, Dict

import numpy as np

try:
    import requests

    from requests_toolbelt.multipart.encoder import MultipartEncoder
    from requests_toolbelt.multipart.decoder import MultipartDecoder
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
                    v = json.dumps(v.tolist())
                else:
                    v = str(v)
                transformer_inputs[k] = v
            m = MultipartEncoder(fields=transformer_inputs)
            # logging.debug(m)
            logging.debug(f"ENCODER {time.time() - start_time}")
            raw_output = requests.post(
                f"http://{self.host}:{self.port}/{self.service}", data=m, headers={"Content-Type": m.content_type}
            )

            start_time = time.time()
            multipart_data = MultipartDecoder.from_response(raw_output)

            output_dict = {}
            for part in multipart_data.parts:
                m = re.match(r'.*?name="(.*?)".*?', part.headers.get(b"Content-Disposition", "").decode("utf-8"))
                if m:
                    output_dict[m.group(1)] = np.asarray(json.loads(part.text))

            for k, v in output_dict.items():
                logging.debug(f"{k} {v.shape}")

            logging.debug(f"DECODER {time.time() - start_time}")
            return output_dict

except:
    logging.warning("BentoML not available")
