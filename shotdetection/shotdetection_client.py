import argparse
import grpc
import logging
import os
import sys
import yaml

import shotdetection_pb2
import shotdetection_pb2_grpc

# read config
_CUR_PATH = os.path.dirname(__file__)
with open(os.path.join(_CUR_PATH, "config.yml")) as f:
    _CFG = yaml.load(f, Loader=yaml.FullLoader)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--info", action="store_true", help="info output")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    return args


def generateRequests(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k"""
    with open(file_object, "rb") as videobytestream:
        while True:
            data = videobytestream.read(chunk_size)
            if not data:
                break
            yield shotdetection_pb2.VideoRequest(video_encoded=data)


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.ERROR
    if args.info:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # open grpc channel
    channel = grpc.insecure_channel("[::]:" + str(_CFG["grpc_port"]))
    stub = shotdetection_pb2_grpc.ShotDetectorStub(channel)

    # test video streaming
    response = stub.copy_video(generateRequests(args.video))
    print(response)

    # test shot detection
    response = stub.get_shots(shotdetection_pb2.ShotRequest(video_id=response.video_id))
    print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())