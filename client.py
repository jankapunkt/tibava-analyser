import os
import sys
import re
import argparse
import logging
import mimetypes

import grpc
import analyser_pb2, analyser_pb2_grpc

from google.protobuf.json_format import MessageToJson


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=50051, type=int)
    parser.add_argument("-t", "--task", choices=["list_plugins", "copy_data"])
    parser.add_argument("--path")
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
            print(MessageToJson(plugin))

        return result

    def copy_data(self, path):
        mimetype = mimetypes.guess_type(path)
        if re.match(r"video/*", mimetype[0]):
            data_type = analyser_pb2.VIDEO_DATA

        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = analyser_pb2_grpc.AnalyserStub(channel)

        def generateRequests(file_object, chunk_size=1024):
            """Lazy function (generator) to read a file piece by piece.
            Default chunk size: 1k"""
            with open(file_object, "rb") as bytestream:
                while True:
                    data = bytestream.read(chunk_size)
                    if not data:
                        break
                    yield analyser_pb2.DataRequest(type=data_type, data_encoded=data)

        response = stub.copy_data(generateRequests(path))

        if response.success:
            return response.id

        logging.error("Error while copying data ...")
        return None


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

    if args.task == "copy_data":
        available_plugins = client.copy_data(args.path)
        print(available_plugins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
