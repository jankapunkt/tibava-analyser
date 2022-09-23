import logging
import os
import sys
import re
import argparse
import time
import uuid
import json

import yaml
import copy
import traceback
from concurrent import futures
import multiprocessing as mp


from google.protobuf.json_format import MessageToJson
from analyser.plugins.plugin import ProgressCallback

import analyser_pb2, analyser_pb2_grpc
import grpc

from analyser.plugins.manager import AnalyserPluginManager
from google.protobuf.json_format import MessageToJson, MessageToDict, ParseDict

from analyser.data import DataManager

# class RunPlugin:
#     def __init__(self, config=None):
#         if config is not None:
#             self.init_worker(config)

#     @classmethod
#     def init(cls, config):
#         print("[RunPlugin] init")

#         manager = AnalyserPluginManager(configs=config.get("image_text", []))
#         manager.find()

#         setattr(cls, "manager", manager)


def run_plugin(args):
    try:
        plugin_manager = globals().get("plugin_manager")
        data_manager = globals().get("data_manager")
        params = args.get("params")
        shared = args.get("shared")
        shared["progress"] = 0.0
        shared["status"] = analyser_pb2.GetPluginStatusResponse.RUNNING
        plugin_inputs = {}
        if "inputs" in params:
            for data_in in params.get("inputs"):
                data = data_manager.load(data_in.get("id"))
                plugin_inputs[data_in.get("name")] = data

        plugin_parameters = {}
        if "parameters" in params:
            for parameter in params.get("parameters"):
                if parameter.get("type") == "FLOAT_TYPE":
                    plugin_parameters[parameter.get("name")] = float(parameter.get("value"))
                if parameter.get("type") == "INT_TYPE":
                    plugin_parameters[parameter.get("name")] = int(parameter.get("value"))
                if parameter.get("type") == "STRING_TYPE":
                    plugin_parameters[parameter.get("name")] = str(parameter.get("value"))
                # data = data_manager.load(data_in.get("name"))
                # plugin_inputs[data_in.get("name")] = data
        callbacks = [ProgressCallback(shared)]
        results = plugin_manager(
            plugin=params.get("plugin"), inputs=plugin_inputs, parameters=plugin_parameters, callbacks=callbacks
        )
        if results is None:
            logging.error(f"Analyser: {params.get('plugin')} without results")
            return []

        result_map = []
        for key, data in results.items():
            data_manager.save(data)
            result_map.append({"name": key, "id": data.id})

        return result_map
    except Exception as e:
        logging.error(f"Analyser: {repr(e)}")
        exc_type, exc_value, exc_traceback = sys.exc_info()

        traceback.print_exception(
            exc_type,
            exc_value,
            exc_traceback,
            limit=2,
            file=sys.stdout,
        )


def init_plugins(config):
    data_dict = {}

    manager = AnalyserPluginManager(configs=config.get("plugins", []))
    manager.find()
    data_dict["plugin_manager"] = manager

    data_manager = DataManager(data_dir=config.get("data_dir", ""))
    data_dict["data_manager"] = data_manager
    return data_dict


def init_process(config):
    globals().update(init_plugins(config))


class Commune(analyser_pb2_grpc.AnalyserServicer):
    def __init__(self, config):
        self.config = config
        self.managers = init_plugins(config)
        self.process_pool = futures.ProcessPoolExecutor(
            max_workers=self.config.get("num_worker", 1), initializer=init_process, initargs=(config,)
        )
        self.shared_manager = mp.Manager()
        self.futures = []

        # self.max_results = config.get("Analyser", {}).get("max_results", 100)

    def list_plugins(self, request, context):
        reply = analyser_pb2.ListPluginsReply()

        for _, plugin_class in self.managers["plugin_manager"].plugins().items():
            reply.plugins.extend([plugin_class.serialize_class()])

        return reply

    def upload_data(self, request_iterator, context):
        # try:
        data = self.managers["data_manager"].load_from_stream(request_iterator)

        return analyser_pb2.UploadDataResponse(success=True, id=data.id)

        # except Exception as e:
        #     logging.error(f"copy_video: {repr(e)}")
        #     logging.error(traceback.format_exc())
        #     # context.set_code(grpc.StatusCode.UNAVAILABLE)
        #     # context.set_details(f"Error transferring video with id {req.video_id}")

        return analyser_pb2.UploadDataResponse(success=False)

    def run_plugin(self, request, context):

        if request.plugin not in self.managers["plugin_manager"].plugins():
            return analyser_pb2.RunPluginResponse(success=False)

        job_id = uuid.uuid4().hex
        variable = {
            "params": MessageToDict(request),
            "config": self.config,
            "future": None,
            "id": job_id,
        }
        process_args = copy.deepcopy(variable)

        d = self.shared_manager.dict()
        d["progress"] = 0.0
        d["status"] = analyser_pb2.GetPluginStatusResponse.WAITING
        variable["shared"] = d
        process_args["shared"] = d
        future = self.process_pool.submit(run_plugin, process_args)
        variable["future"] = future
        self.futures.append(variable)

        return analyser_pb2.RunPluginResponse(success=True, id=job_id)

    def get_plugin_status(self, request, context):
        futures_lut = {x["id"]: i for i, x in enumerate(self.futures)}
        response = analyser_pb2.GetPluginStatusResponse()
        if request.id in futures_lut:
            job_data = self.futures[futures_lut[request.id]]
            done = job_data["future"].done()

            status = job_data["shared"].get("status", analyser_pb2.GetPluginStatusResponse.UNKNOWN)
            response.status = status

            progress = job_data["shared"].get("progress", 0.0)
            response.progress = progress
            if not done:
                return response

            # try:
            results = job_data["future"].result()

            if results is None:
                response.status = analyser_pb2.GetPluginStatusResponse.ERROR
                return response
            for k in results:
                output = response.outputs.add()
                output.name = k["name"]
                output.id = k["id"]

            # except Exception as e:
            #     logging.error(f"Analyser: {repr(e)}")
            #     logging.error(traceback.format_exc())
            #     logging.error(traceback.print_stack())

            #     response.status = analyser_pb2.GetPluginStatusResponse.ERROR
            #     return response

            response.status = analyser_pb2.GetPluginStatusResponse.DONE
            return response
        response.status = analyser_pb2.GetPluginStatusResponse.UNKNOWN

        return response

    def download_data(self, request, context):
        try:
            data = self.managers["data_manager"].load(request.id)

            for x in self.managers["data_manager"].dump_to_stream(data):
                yield analyser_pb2.DownloadDataResponse(type=x["type"], data_encoded=x["data_encoded"])

        except Exception as e:
            logging.error(f"download_data: {repr(e)}")
            logging.error(traceback.format_exc())
            return analyser_pb2.DownloadDataResponse()
            # context.set_code(grpc.StatusCode.UNAVAILABLE)
            # context.set_details(f"Error transferring video with id {req.video_id}")


class Server:
    def __init__(self, config):
        self.config = config

        self.commune = Commune(config)

        pool = futures.ThreadPoolExecutor(max_workers=10)

        self.server = grpc.server(
            pool,
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        analyser_pb2_grpc.add_AnalyserServicer_to_server(
            self.commune,
            self.server,
        )

        grpc_config = config.get("grpc", {})
        port = grpc_config.get("port", 50051)
        self.server.add_insecure_port(f"[::]:{port}")

    def run(self):
        logging.info("[Server] starting")
        self.server.start()
        logging.info("[Server] ready")

        try:
            while True:
                num_jobs = len(self.commune.futures)
                num_jobs_done = len([x for x in self.commune.futures if x["future"].done()])
                # print(num_jobs)
                # print(num_jobs_done)
                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)


def read_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument("-c", "--config", help="config path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    if args.config is not None:
        config = read_config(args.config)
    else:
        config = {}

    time.sleep(1)  # TODO
    server = Server(config)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
