import sys
import argparse
import logging

from analyser.client import AnalyserClient


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--input_path", default="/media/test.mp4", help="path to input video .mp4")
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
    logging.info(f"Start uploading")
    data_id = client.upload_file(args.input_path)
    logging.info(f"Upload done: {data_id}")

    job_id = client.run_plugin("color_brightness_analyser", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job color_brightness_analyser started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    logging.info(result)

    brightness_id = None
    for output in result.outputs:
        if output.name == "brightness":
            brightness_id = output.id

    job_id = client.run_plugin("transnet_shotdetection", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job shot_detection started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    shots_id = None
    for output in result.outputs:
        if output.name == "shots":
            shots_id = output.id

    job_id = client.run_plugin(
        "shot_scalar_annotator", [{"id": shots_id, "name": "shots"}, {"id": brightness_id, "name": "scalar"}], []
    )

    logging.info(f"Job shot_scalar_annotator started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    annotations_id = None
    for output in result.outputs:
        if output.name == "annotations":
            annotations_id = output.id
    data = client.download_data(annotations_id, args.output_path)
    with data:
        logging.info(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
