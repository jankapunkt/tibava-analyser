import os
import sys
import re
import argparse
import logging
import time

import grpc

from analyser.client import AnalyserClient
from analyser import analyser_pb2


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--input_path", help="verbose output")
    parser.add_argument("--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    client = AnalyserClient("localhost", 50051)
    logging.info(f"Start uploading")
    data_id = client.upload_data(args.input_path)
    logging.info(f"Upload done: {data_id}")

    job_id = client.run_plugin("video_to_audio", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job video_to_audio started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return
    # while True:
    #     result = client.get_plugin_status(job_id)
    #     if result.status == analyser_pb2.GetPluginStatusResponse.RUNNING:
    #         time.sleep(0.5)
    #         continue
    #     elif result.status == analyser_pb2.GetPluginStatusResponse.ERROR:
    #         logging.error("Job is crashing")
    #         return
    #     elif result.status == analyser_pb2.GetPluginStatusResponse.DONE:
    #         break

    #     break

    audio_id = None
    for output in result.outputs:
        if output.name == "audio":
            audio_id = output.id

    logging.info(f"Job video_to_audio done: {audio_id}")

    job_id = client.run_plugin("audio_amp_analysis", [{"id": audio_id, "name": "audio"}], [])
    logging.info(f"Job audio_amp started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    amp_id = None
    for output in result.outputs:
        if output.name == "amp":
            amp_id = output.id

    logging.info(f"Job audio_amp done: {amp_id}")

    logging.info(client.download_data(amp_id, args.output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
