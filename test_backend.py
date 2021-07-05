import argparse
import logging
import os
import requests
import sys
from time import sleep

_BACKEND_URL = "http://127.0.0.1:5000/"


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
    fps = response["metadata"]["fps"]
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
            response = requests.get(_BACKEND_URL + "detect_shots", {"job_id": job_id, "fps": fps})
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

        # create html for shot keyframes
        with open(os.path.join(os.path.dirname(args.video_path), str(video_id) + "_keyframes.html"), "w") as html_file:
            for shot in response["shots"]:
                for keyframe in shot["keyframes"]:
                    html_file.write(
                        f"""
<div>
    <p>Shot {shot["shot_id"]}</p>
    <img src="data:image/jpg;base64,{keyframe}">
</div>

"""
                    )

        # logging.info(shots)

        # convert shots to csv format
        logging.info("Converting shots to csv format")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": shots,
                "format": "csv",
                "task": "Cuts",
                "keys_to_store": ["shot_id", "start_frame", "end_frame"],
            },
        )

        logging.info(response.json())

        # convert shots to jsonl format
        logging.info("Converting shots to jsonl format")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": shots,
                "format": "jsonl",
                "task": "Cuts",
                # "keys_to_store": ["shot_id", "start_frame", "end_frame", "start_time", "end_time"],
            },
        )

        logging.info(response.json())

        # convert shots to shoebox format
        logging.info("Converting shots to shoebox format")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": shots,
                "format": "shoebox",
                "task": "Cuts",
                "ELANBegin_key": "start_time",
                "ELANEnd_key": "end_time",
            },
        )

        logging.info(response.json())

    if args.test_face:
        response = requests.post(_BACKEND_URL + "detect_faces", {"video_id": video_id, "path": args.video_path})
        logging.info(response)
        response = response.json()
        job_id = response["job_id"]

        faces = []
        logging.info("Detect faces in video ...")

        while True:
            response = requests.get(_BACKEND_URL + "detect_faces", {"job_id": job_id})
            response = response.json()
            logging.debug(response)

            if "status" in response and response["status"] == "SUCCESS":
                logging.info("JOB DONE!")
                faces = response["faces"]
                break
            elif "status" in response and response["status"] == "PENDING":
                sleep(0.5)
            else:
                logging.error("Something went wrong")
                break

        # convert shots to csv format
        logging.info("Converting faces to csv format")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": faces,
                "format": "csv",
                "task": "Faces",
                "keys_to_store": ["face_id", "frame_idx", "bbox_xywh", "bbox_area"],
            },
        )

        logging.info(response.json())

        # convert shots to jsonl format
        logging.info("Converting faces to jsonl format")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": faces,
                "format": "jsonl",
                "task": "Faces",
            },
        )

        logging.info(response.json())

        logging.info("Perform face clustering ...")
        face_clusters = []
        response = requests.post(_BACKEND_URL + "cluster_faces", {"video_id": video_id, "path": args.video_path})
        logging.info(response)
        response = response.json()
        job_id = response["job_id"]

        while True:
            response = requests.get(_BACKEND_URL + "cluster_faces", {"job_id": job_id})
            response = response.json()
            logging.debug(response)

            if "status" in response and response["status"] == "SUCCESS":
                logging.info("JOB DONE!")
                face_clusters = response["face_clusters"]
                break
            elif "status" in response and response["status"] == "PENDING":
                sleep(0.5)
            else:
                logging.error("Something went wrong")
                break

        logging.info(face_clusters)

    return 0


if __name__ == "__main__":
    sys.exit(main())
