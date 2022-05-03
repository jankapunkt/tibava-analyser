import logging
import os
import sys
import re
import argparse
import time
import json
from concurrent import futures

import analyser_pb2, analyser_pb2_grpc
import grpc

from analyser.plugins.manager import AnalyserPluginManager


def init_plugins(config):
    data_dict = {}

    manager = AnalyserPluginManager(configs=config.get("image_text", []))
    manager.find()

    data_dict["manager"] = manager

    return data_dict


def init_process(config):
    globals().update(init_plugins(config))


class Commune(analyser_pb2_grpc.AnalyserServicer):
    def __init__(self, config):
        self.config = config
        self.managers = init_plugins(config)
        print(self.managers["manager"].plugins())
        self.process_pool = futures.ProcessPoolExecutor(max_workers=1, initializer=init_process, initargs=(config,))
        # self.indexing_process_pool = futures.ProcessPoolExecutor(
        #     max_workers=8, initializer=IndexingJob().init_worker, initargs=(config,)
        # )
        self.futures = []

        # self.max_results = config.get("indexer", {}).get("max_results", 100)

    def list_plugins(self, request, context):
        reply = analyser_pb2.ListPluginsReply()

        for _, plugin_class in self.managers["manager"].plugins().items():
            print(plugin_class.serialize_class())
            reply.plugins.extend([plugin_class.serialize_class()])

        return reply

    def copy_data(self, request_iterator, context):
        try:
            # save data from request input stream
            datastream = iter(request_iterator)
            firstpkg = next(datastream)
            type = firstpkg.type

            with open(os.path.join(self.config.get("data_folder"), video_id + ".mp4"), "wb") as videofile:
                videofile.write(firstpkg.video_encoded)  # write first package
                for data in datastream:
                    videofile.write(data.video_encoded)

            return shotdetection_pb2.VideoResponse(success=True)

        except Exception as e:
            logging.error(f"copy_video: {repr(e)}")
            logging.error(traceback.format_exc())
            # context.set_code(grpc.StatusCode.UNAVAILABLE)
            # context.set_details(f"Error transferring video with id {req.video_id}")

        return shotdetection_pb2.VideoResponse(success=False)


class Server:
    def __init__(self, config):
        self.config = config
        self.commune = Commune(config)

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
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
        self.server.start()
        logging.info("[Server] Ready")

        try:
            while True:
                num_jobs = len(self.commune.futures)
                num_jobs_done = len([x for x in self.commune.futures if x["future"].done()])

                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)


def read_config(path):
    with open(path, "r") as f:
        return json.load(f)
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
