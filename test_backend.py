import argparse
import logging
import os
import requests
import sys
from time import sleep

_BACKEND_URL = "http://127.0.0.1:5000/"
_FACE_DETECTION_URL = "http://127.0.0.1:5002/"


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")

    parser.add_argument("--video_path", type=str, required=True, help="path to video file")

    parser.add_argument("--test_face", action="store_true", help="test face detection module")
    parser.add_argument("--test_shot", action="store_true", help="test shot detection module")

    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    title = os.path.splitext(os.path.basename(args.video_path))[0]

    logging.info("Ping backend")
    logging.info(requests.get(_BACKEND_URL + "ping").json())

    # get meta information and corresponding video_id for redis cache
    logging.info("Get meta information")
    response = requests.get(_BACKEND_URL + "read_meta", {"title": title, "path": args.video_path}).json()
    video_id = response["video_id"]
    logging.info(response)

    # test shot detection service
    if args.test_shot:
        response = requests.post(_BACKEND_URL + "detect_shots", {"video_id": video_id, "path": args.video_path})
        logging.info(response)
        response = response.json()
        job_id = response["job_id"]

        shots = []
        logging.info("Detect shots in video ...")
        while True:
            response = requests.get(_BACKEND_URL + "detect_shots", {"job_id": job_id})
            response = response.json()
            logging.debug(response)

            if "status" in response and response["status"] == "SUCCESS":
                logging.info("JOB DONE!")
                shots = response["shots"]
                break
            elif "status" in response and response["status"] == "PENDING":
                sleep(0.5)
            else:
                logging.error("Something went wrong")
                break

        logging.info(shots)

        # convert shots to shoebox format
        logging.info("Converting shots to shoebox format")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": shots,
                "format": "shoebox",
                "ELANType_key": "CUTS",
                "ELANBegin_key": "start_frame",
                "ELANEnd_key": "end_frame",
            },
        )

        logging.info(response.json())

    if args.test_face:
        logging.info("Ping face detection")
        logging.info(requests.get(_FACE_DETECTION_URL + "ping").json())

        logging.info("Test face detection")
        response = requests.put(
            _BACKEND_URL + "detect_faces/" + str(args.video_id),
            {"title": title, "path": args.video_path, "max_frames": 100},
        )

        logging.info(response)
        logging.info(response.json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
