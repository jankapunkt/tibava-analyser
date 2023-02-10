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
                    print(f"{k}: {v.shape}", flush=True)
                    v = json.dumps(v.tolist())
                else:
                    v = str(v)
                transformer_inputs[k] = v
                print(k)
            m = MultipartEncoder(fields=transformer_inputs)
            # print(m)
            print(f"ENCODER {time.time() - start_time}")
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
                print(f"{k} {v.shape}")

            print(f"DECODER {time.time() - start_time}")
            return output_dict

except:
    logging.warning("BentoML not available")


# try:
#     import grpc
#     import asyncio
#     from bentoml.grpc.utils import import_generated_stubs
#     from bentoml.io import NumpyNdarray, Multipart

#     pb, services = import_generated_stubs()

#     @InferenceServer.export("bentoml_grpc")
#     class BentoMLGRPCInferenceServer:
#         def __init__(self, config: Dict = None) -> None:
#             if config is None:
#                 logging.error("No config provided for bentoml interface")
#                 return
#             self.host = config.get("host", "localhost")
#             self.port = config.get("port", 3000)
#             self.service = config.get("service")

#         def __call__(self, inputs: Dict, outputs: List):
#             with grpc.insecure_channel(
#                 f"{self.host}:{self.port}",
#                 options=[
#                     ("grpc.max_send_message_length", 50 * 1024 * 1024),
#                     ("grpc.max_receive_message_length", 50 * 1024 * 1024),
#                 ],
#             ) as channel:
#                 stub = services.BentoServiceStub(channel)
#                 multipart = pb.Multipart()
#                 for k, v in inputs.items():
#                     if isinstance(v, np.ndarray):
#                         #     print(f"{k}: {v.shape}", flush=True)
#                         if v.dtype == np.uint8:
#                             v = v.astype(np.uint32)
#                         print(k)
#                         multipart.fields[k].CopyFrom(pb.Part(ndarray=asyncio.run(NumpyNdarray().to_proto(v))))
#                     # transformer_inputs[k] = v
#                 # multipart = asyncio.run(Multipart(**{k: NumpyNdarray for k in inputs.keys()}).to_proto(transformer_inputs))
#                 # print(multipart, flush=True)
#                 # data = json.dumps(transformer_inputs)
#                 # NumpyNdarray(dtype=np.float16, shape=[2, 2])
#                 print("###### 1", flush=True)
#                 print(channel, flush=True)
#                 print(stub, flush=True)
#                 response = stub.Call(pb.Request(api_name=self.service, multipart=multipart))
#                 print("###### 2", flush=True)
#                 print(response, flush=True)
#                 # print(data)
#                 # print(f"http://{self.host}:{self.port}/{self.service}")
#                 # raw_output = requests.post(
#                 #     f"http://{self.host}:{self.port}/{self.service}",
#                 #     headers={"content-type": "application/json"},
#                 #     data=data,
#                 # ).json()

#                 # print(f"{raw_output}", flush=True)
#                 output_dict = {}
#                 for x in outputs:
#                     if x not in raw_output:
#                         logging.error(f"Unknown output field {x}")
#                         return None
#                     try:
#                         output_dict[x] = np.asarray(raw_output[x])
#                     except:
#                         output_dict[x] = raw_output[x]

#                 # print(f"{output_dict}", flush=True)
#                 return output_dict

# except:
#     logging.warning("BentoML GRPC not available")
