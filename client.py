import os
import sys
import re
import argparse
import logging

import grpc
import analyser_pb2, analyser_pb2_grpc


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=50051, type=int)
    parser.add_argument("-t", "--task", choices=["list_plugins"])
    args = parser.parse_args()
    return args


class AnalyserClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def list_plugins(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = analyser_pb2_grpc.AnalyserStub(channel)
        response = stub.list_plugins(analyser_pb2.ListPluginsRequest())

        result = {}

        for plugin in response.plugins:
            print(plugin)

        return result

    def copy_data_to_grpc_server(self, data_type, path):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = analyser_pb2_grpc.AnalyserStub(channel)

        def generateRequests(video_id, file_object, chunk_size=1024):
            """Lazy function (generator) to read a file piece by piece.
            Default chunk size: 1k"""
            with open(file_object, "rb") as videobytestream:
                while True:
                    data = videobytestream.read(chunk_size)
                    if not data:
                        break
                    yield analyser_pb2.DataRequest(type=data_type, data_encoded=data)

        if cache_result:
            logging.info(f"Video with id {video_id} already stored ...")
            return True
        else:
            logging.info(f"Transferring video with id {video_id} to shotdetection server ...")
            response = stub.copy_video(generateRequests(video_id, video_path))

            if response.success:
                r.set(f"videofile_{video_id}", msgpack.packb({"stored": True}))
                return True

            logging.error("Error while copying video ...")
            return False


def main():
    args = parse_args()

    level = logging.ERROR
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)
    client = AnalyserClient(args.host, args.port)

    if args.task == "list_plugins":
        available_plugins = client.list_plugins()
        print(available_plugins)

    return 0


if __name__ == "__main__":
    sys.exit(main())
