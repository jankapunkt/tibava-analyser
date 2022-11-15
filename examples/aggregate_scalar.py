import sys
import argparse
import numpy as np
import logging

from analyser.client import AnalyserClient
from analyser.data import ListData, ScalarData


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    # parser.add_argument("--input_path", default="/media/test.mp4", help="path to input video .mp4")
    parser.add_argument("--fps", nargs="+", type=int, default=[2, 5], help="fps of input scalar series")
    parser.add_argument("--duration", default=30, help="duration [s] of the signal")
    parser.add_argument("--output_path", default="/media", help="path to output folder")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    client = AnalyserClient("localhost", 50051)

    # Create random time series with given fps
    timelines = []
    for fps in args.fps:
        time = np.linspace(0, args.duration, int(args.duration * fps))[:, np.newaxis]
        y = np.random.rand(int(args.duration * fps))
        timelines.append(ScalarData(y=y.squeeze(), time=time.squeeze().tolist(), delta_time=1 / fps))

    timelines = ListData(
        data=[x for x in timelines],
        index=[f"timeline{i}" for i in range(len(timelines))],
    )

    # Upload data
    logging.info(f"Start uploading")
    data_id = client.upload_data(timelines)
    logging.info(f"Upload done: {data_id}")

    # Run scalar aggregation
    job_id = client.run_plugin("aggregate_scalar", [{"id": data_id, "name": "timelines"}], [])
    logging.info(f"Job aggregate_scalar started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    logging.info(result)

    output_id = None
    for output in result.outputs:
        if output.name == "probs":
            output_id = output.id

    logging.info(output_id)
    logging.info(client.download_data(output_id, args.output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
