from ..inference import Device, Backend, generate_id, InferenceServer
import logging

import time
from typing import AnyStr, Union, List, Dict

import numpy as np


@InferenceServer.export("redisai")
class RedisAIInferenceServer:

    backend_lut = {
        Backend.PYTORCH: "TORCH",
        Backend.TENSORFLOW: "TF",
        Backend.ONNX: "ONNX",
    }

    device_lut = {
        Device.CPU: "cpu",
        Device.GPU: "gpu",
    }

    def __init__(
        self,
        model_file: str,
        model_name: str = None,
        backend: Union[AnyStr, Backend] = Backend.PYTORCH,
        device: Union[AnyStr, Device] = Device.CPU,
        timeout: float = 60,
        host: str = None,
        port: int = 6379,
        inputs: Union[AnyStr, List[AnyStr]] = None,
        outputs: Union[AnyStr, List[AnyStr]] = None,
        batch: int = 16,
    ):
        try:
            import redisai as rai
            import ml2rt
        except:
            logging.error("Could not import redisai python interface")
            return None

        self.model_name = model_name
        self.model_file = model_file
        self.host = host
        self.port = port
        self.batch = batch

        self.con = None
        start_time = time.time()
        while time.time() < start_time + timeout:
            try:
                self.con = rai.Client(host=self.host, port=self.port)
                break
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                logging.warning(e)
                time.sleep(0.1)

        assert self.con is not None, "Unable to connect to redisai"

        model = ml2rt.load_model(self.model_file)

        if isinstance(backend, str):
            if backend.lower() == "torch":
                backend = Backend.PYTORCH
            elif backend.lower() == "tf":
                backend = Backend.TENSORFLOW
            elif backend.lower() == "onnx":
                backend = Backend.ONNX

        if isinstance(device, str):
            if device.lower() == "cpu":
                device = Device.CPU
            elif device.lower() == "gpu":
                device = Device.GPU

        assert backend in self.backend_lut, "Backend is unknown"
        assert device in self.device_lut, "Device is unknown"
        # print(f"inputs: {inputs}")
        # print(f"outputs: {outputs}")

        start_time = time.time()
        model_uploaded = False
        while time.time() < start_time + timeout:
            try:
                self.con.modelset(
                    self.model_name,
                    backend=self.backend_lut[backend],
                    device=self.device_lut[device],
                    data=model,
                    batch=self.batch,
                    inputs=inputs,
                    outputs=outputs,
                )
                model_uploaded = True
                break
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                logging.warning(e)
                time.sleep(0.1)

        assert model_uploaded, "Unable to upload model to redisai"

    def __call__(self, inputs: Dict, outputs: List):

        job_id = generate_id()
        input_keys = []
        for k, v in inputs.items():
            self.con.tensorset(f"{k}_{job_id}", v)
            input_keys.append(f"{k}_{job_id}")

        output_keys = []
        for o in outputs:
            output_keys.append(f"{o}_{job_id}")

        # print(f"run inputs: {input_keys}")
        # print(f"run outputs: {output_keys}")
        ok = self.con.modelrun(self.model_name, input_keys, output_keys)

        for k in input_keys:
            self.con.delete(k)
        if ok != "OK":
            return None

        output_dict = {}
        for o in outputs:
            output_dict[o] = self.con.tensorget(f"{o}_{job_id}")
            self.con.delete(f"{o}_{job_id}")
        return output_dict
