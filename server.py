import logging
import os
import sys
import re
import argparse
import time
import uuid
import json
from more_itertools import first
import yaml
import copy
import traceback
from concurrent import futures

from google.protobuf.json_format import MessageToJson

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


def plugin_run(args):

    try:
        plugin_manager = globals().get("plugin_manager")
        data_manager = globals().get("data_manager")
        params = args.get("params")
        plugin_inputs = []
        for data_in in params.get("inputs"):
            data = data_manager.load(data_in.get('id'))
            print(data)
        return
        plugin_manager(plugin=params.get("plugin"), inputs=params.get("inputs"))
        print(plugin_manager)
        print(args)
    except Exception as e:
        logging.error(f"Indexer: {repr(e)}")
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

    manager = AnalyserPluginManager(configs=config.get("image_text", []))
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
        self.process_pool = futures.ProcessPoolExecutor(max_workers=1, initializer=init_process, initargs=(config,))
        self.futures = []

        # self.max_results = config.get("indexer", {}).get("max_results", 100)

    def list_plugins(self, request, context):
        reply = analyser_pb2.ListPluginsReply()

        for _, plugin_class in self.managers["plugin_manager"].plugins().items():
            print(plugin_class.serialize_class())
            reply.plugins.extend([plugin_class.serialize_class()])

        return reply

    def upload_data(self, request_iterator, context):
        try:
            data = self.managers["data_manager"].load_from_stream(request_iterator)
            print(data)
            # save data from request input stream
            # datastream = iter(request_iterator)
            # firstpkg = next(datastream)
            # data = data_from_proto(firstpkg, data_dir=self.config)
            # data.load_from_stream(datastream)

            # data.add_data_from_proto(firstpkg)
            # for x in datastream:
            # data.add_data_from_proto(data)

            return analyser_pb2.UploadDataResponse(success=True, id=data.id)

        except Exception as e:
            logging.error(f"copy_video: {repr(e)}")
            logging.error(traceback.format_exc())
            # context.set_code(grpc.StatusCode.UNAVAILABLE)
            # context.set_details(f"Error transferring video with id {req.video_id}")

        return analyser_pb2.UploadDataResponse(success=False)

    def run_plugin(self, request, context):

        logging.info("[Commune] run")
        reply = analyser_pb2.RunPluginResponse()
        if request.plugin not in self.managers["plugin_manager"].plugins():
            return reply

        job_id = uuid.uuid4().hex
        variable = {
            "params": MessageToDict(request),
            "config": self.config,
            "future": None,
            "id": job_id,
        }

        future = self.process_pool.submit(plugin_run, copy.deepcopy(variable))
        variable["future"] = future
        self.futures.append(variable)

        print(self.managers["plugin_manager"].plugins())
        logging.info(MessageToJson(request))

        return reply


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
                print(num_jobs)
                print(num_jobs_done)
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
    server = Server(config)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
