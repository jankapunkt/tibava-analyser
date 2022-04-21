import argparse
import grpc
import imageio
import logging
import os
import sys
import yaml

import facedetection_pb2
import facedetection_pb2_grpc

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
    stub = facedetection_pb2_grpc.FaceDetectorStub(channel)

    # test face detection (store video first using shotdetection service)
    response = stub.detect_faces(video_id=response.video_id)
    print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())