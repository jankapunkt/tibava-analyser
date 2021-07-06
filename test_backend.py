import argparse
import json
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


def post_job(route, args):
    response = requests.post(_BACKEND_URL + route, json=args)
    logging.info(response)
    response = response.json()
    return response["job_id"]


def get_response(route, args):
    while True:
        response = requests.get(_BACKEND_URL + route, args)
        response = response.json()
        logging.debug(response)

        if "status" in response and response["status"] == "SUCCESS":
            logging.info("JOB DONE!")
            return response
        elif "status" in response and response["status"] == "PENDING":
            sleep(0.5)
        else:
            logging.error("Something went wrong")
            break

    return None


def generate_thumbnails(output_file, data):
    with open(output_file, "w") as html_file:
        for entry in data:
            html_file.write(
                f"""
<div>
    <p>Facecluster {entry["id"]}</p>
    <img src="data:image/jpg;base64,{entry["img"]}">
</div>

"""
            )
    return


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    title = os.path.splitext(os.path.basename(args.video_path))[0]

    logging.info("Ping backend ...")
    logging.info(requests.get(_BACKEND_URL + "ping").json())

    # get meta information and corresponding video_id for redis cache
    logging.info("Get meta information ...")
    response = requests.get(_BACKEND_URL + "read_meta", {"title": title, "path": args.video_path}).json()
    video_id = response["video_id"]
    fps = response["metadata"]["fps"]
    logging.info(response)

    # test shot detection service
    if args.test_shot:
        logging.info("Detect shots in video ...")
        job_id = post_job(route="detect_shots", args={"video_id": video_id, "path": args.video_path})
        response = get_response(route="detect_shots", args={"job_id": job_id, "fps": fps})

        shots = []
        if response:
            shots = response["shots"]

        # convert shots to csv format
        logging.info("Converting shots to csv format ...")
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
        logging.info("Converting shots to jsonl format ...")
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
        logging.info("Converting shots to shoebox format ...")
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

        # generate thumbnails
        logging.info("Generating thumbnails for shots ...")
        thumbnail_frames = []
        for shot in shots:
            middle_frame = shot["start_frame"] + (shot["end_frame"] - shot["start_frame"]) // 2

            for frame in [shot["start_frame"], middle_frame, shot["end_frame"]]:
                thumbnail_frames.append(
                    {
                        "id": shot["shot_id"],
                        "idx": frame,
                        "bbox_xywh": None,
                    }
                )

        job_id = post_job(
            route="get_thumbnails", args={"video_id": video_id, "path": args.video_path, "frames": thumbnail_frames}
        )
        response = get_response(route="get_thumbnails", args={"job_id": job_id})

        if response:
            output_file = os.path.join(os.path.dirname(args.video_path), str(video_id) + "_Cuts.html")
            generate_thumbnails(output_file=output_file, data=response["thumbnails"])

    if args.test_face:
        # FACE DETECTION
        logging.info("Detect faces in video ...")
        job_id = post_job(route="detect_faces", args={"video_id": video_id, "path": args.video_path})
        response = get_response(route="detect_faces", args={"job_id": job_id})

        faces = []
        if response:
            faces = response["faces"]

        # convert shots to jsonl format
        logging.info("Converting faces to jsonl format ...")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": faces,
                "format": "jsonl",
                "task": "FaceDetection",
            },
        )
        logging.info(response.json())

        # FACE CLUSTERING
        logging.info("Perform face clustering ...")
        job_id = post_job(route="cluster_faces", args={"video_id": video_id, "path": args.video_path})
        response = get_response(route="cluster_faces", args={"job_id": job_id})

        face_clusters = []
        if response:
            face_clusters = response["face_clusters"]

        logging.info("Converting face clusters to jsonl format ...")
        response = requests.post(
            _BACKEND_URL + "export_data",
            json={
                "video_id": video_id,
                "input_data": face_clusters,
                "format": "jsonl",
                "task": "FaceClustering",
            },
        )
        logging.info(response.json())

        # create thumbnails for face clusters
        logging.info("Generating thumbnails for face clusters ...")
        thumbnail_frames = []
        face_dict = {}
        for face in faces:
            face_dict[face["face_id"]] = face

        for cluster in face_clusters:
            logging.info(cluster)

            face_ids = [
                cluster["face_ids"][0],
                cluster["face_ids"][cluster["occurrences"] // 2],
                cluster["face_ids"][-1],
            ]

            for face_id in face_ids:
                thumbnail_frames.append(
                    {
                        "id": cluster["cluster_id"],
                        "idx": face_dict[face_id]["frame_idx"],
                        "bbox_xywh": face_dict[face_id]["bbox_xywh"],
                    }
                )
        logging.debug(thumbnail_frames)

        job_id = post_job(
            route="get_thumbnails", args={"video_id": video_id, "path": args.video_path, "frames": thumbnail_frames}
        )
        response = get_response(route="get_thumbnails", args={"job_id": job_id})
        if response:
            output_file = os.path.join(os.path.dirname(args.video_path), str(video_id) + "_FaceClustering.html")
            generate_thumbnails(output_file=output_file, data=response["thumbnails"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
