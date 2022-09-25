import os
import sys
import re
import argparse
import logging
import time

import grpc
import json

from analyser.client import AnalyserClient
from analyser import analyser_pb2


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--video_path", help="path to input video .mp4")
    parser.add_argument("--host", default="localhost", help="host to analyser server")
    parser.add_argument("--port", type=int, default=50051, help="port to analyser server")

    parser.add_argument("--existing_path", help="path to some existing outputs")
    parser.add_argument("--output_path", help="path to output folder")
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    input_files = []
    if os.path.isdir(args.video_path):
        for root, dirs, files in os.walk(args.video_path):
            for f in files:
                file_path = os.path.join(root, f)
                input_files.append(file_path)
    else:
        input_files.append(args.video_path)
            


    existing_shot_data = {}
    if args.existing_path:
        with open(args.existing_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "shot":
                    existing_shot_data[d.get("video_id")] = d
    
    client = AnalyserClient(args.host, args.port)
    
    with open(os.path.join(args.output_path, 'prediction.jsonl'), 'w') as f:
        for input_file in input_files:
            if input_file in existing_shot_data:
                logging.info(f"skip {input_file}")
                continue
            logging.info(f"Start uploading")
            video_id = client.upload_file(input_file)
            logging.info(f"Upload done: {video_id}")


            job_id = client.run_plugin("transnet_shotdetection", [{"id": video_id, "name": "video"}], [])
            logging.info(f"Job video_to_audio started: {job_id}")

            result = client.get_plugin_results(job_id=job_id)
            if result is None:
                logging.error("Job is crashing")
                return

            shots_id = None
            for output in result.outputs:
                if output.name == "shots":
                    shots_id = output.id

            client.download_data(shots_id, args.output_path)
            if shots_id is not None:
                f.write(json.dumps({"video_id":input_file, "type":"shot", "shots_id": shots_id})+'\n')


    return 0


if __name__ == "__main__":
    sys.exit(main())
