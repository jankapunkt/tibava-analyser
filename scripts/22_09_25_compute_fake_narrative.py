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


def compute_shot_segments(client, video_id, output_path):

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

    data = client.download_data(shots_id, output_path)

    return data.id


def compute_face_emotions(client, video_id, output_path):

    # insightface_detection
    job_id = client.run_plugin("insightface_video_detector", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job insightface_video_detector started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get facial images from facedetection
    images_id = None
    for output in result.outputs:
        if output.name == "images":
            images_id = output.id

    # deepface_emotion
    job_id = client.run_plugin("deepface_emotion", [{"id": images_id, "name": "images"}], [])
    logging.info(f"Job deepface_emotion started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get emotions
    output_id = None
    for output in result.outputs:
        if output.name == "probs":
            output_id = output.id

    data = client.download_data(output_id, output_path)

    return data.id


def compute_face_emotions_annotation(client, shots_id, face_emotions_id, output_path):
    job_id = client.run_plugin(
        "shot_annotator", [{"id": shots_id, "name": "shots"}, {"id": face_emotions_id, "name": "probs"}], []
    )
    if job_id is None:
        return

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        return

    annotation_id = None
    for output in result.outputs:
        if output.name == "annotations":
            annotation_id = output.id
    if annotation_id is None:
        return

    result_annotations = client.download_data(annotation_id, output_path)
    return result_annotations.id


def compute_shot_sizes(client, video_id, output_path):

    job_id = client.run_plugin("shot_type_classifier", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job shot_type_classifier started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    logging.info(result)

    output_id = None
    for output in result.outputs:
        if output.name == "probs":
            output_id = output.id

    data = client.download_data(output_id, output_path)

    return data.id


def compute_shot_sizes_annotation(client, shots_id, shot_size_id, output_path):
    job_id = client.run_plugin(
        "shot_annotator", [{"id": shots_id, "name": "shots"}, {"id": shot_size_id, "name": "probs"}], []
    )
    if job_id is None:
        return

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        return

    annotation_id = None
    for output in result.outputs:
        if output.name == "annotations":
            annotation_id = output.id
    if annotation_id is None:
        return

    result_annotations = client.download_data(annotation_id, output_path)
    return result_annotations.id


def compute_places(client, video_id, output_path):
    """
    Run place classification
    """
    job_id = client.run_plugin("places_classifier", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job places_classifier started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    logging.info(result)

    places365_id = None
    places16_id = None
    places3_id = None

    for output in result.outputs:
        if output.name == "probs_places365":
            places365_id = output.id
        if output.name == "probs_places16":
            places16_id = output.id
        if output.name == "probs_places3":
            places3_id = output.id

    places365_data = client.download_data(places365_id, output_path)
    places16_data = client.download_data(places16_id, output_path)
    places3_data = client.download_data(places3_id, output_path)

    return places365_data.id, places16_data.id, places3_data.id


def compute_places_annotation(client, shots_id, places365_id, places16_id, places3_id, output_path):
    results = []
    for places_id in [places365_id, places16_id, places3_id]:
        job_id = client.run_plugin(
            "shot_annotator", [{"id": shots_id, "name": "shots"}, {"id": places_id, "name": "probs"}], []
        )
        if job_id is None:
            return

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            return

        annotation_id = None
        for output in result.outputs:
            if output.name == "annotations":
                annotation_id = output.id
        if annotation_id is None:
            return
        result_annotations = client.download_data(annotation_id, output_path)
        results.append(result_annotations.id)
    return results


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
    existing_face_emotions_data = {}
    existing_shot_size_data = {}
    existing_places_data = {}

    existing_face_emotions_annotaion_data = {}
    existing_shot_size_annotaion_data = {}
    existing_places_annotaion_data = {}

    existing = []
    if args.existing_path:
        with open(args.existing_path, "r") as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "shot":
                    existing_shot_data[d.get("video_id")] = d
                if d.get("type") == "face_emotions":
                    existing_face_emotions_data[d.get("video_id")] = d
                if d.get("type") == "shot_size":
                    existing_shot_size_data[d.get("video_id")] = d
                if d.get("type") == "places":
                    existing_places_data[d.get("video_id")] = d

                if d.get("type") == "face_emotions_annotaion":
                    existing_face_emotions_annotaion_data[d.get("video_id")] = d
                if d.get("type") == "shot_size_annotaion":
                    existing_shot_size_annotaion_data[d.get("video_id")] = d
                if d.get("type") == "places_annotaion":
                    existing_places_annotaion_data[d.get("video_id")] = d

                existing.append(line)

    client = AnalyserClient(args.host, args.port)

    with open(os.path.join(args.output_path, "prediction.jsonl"), "w") as f:
        for line in existing:
            f.write(line)

        for input_file in input_files:

            logging.info(f"Start uploading")
            video_id = client.upload_file(input_file)
            logging.info(f"Upload done: {video_id}")

            if input_file not in existing_shot_data:
                shots_id = compute_shot_segments(client, video_id, output_path=args.output_path)
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "shot",
                            "shots_id": shots_id,
                        }
                    )
                    + "\n"
                )
            else:
                shots_id = existing_shot_data[input_file]["shots_id"]

            if input_file not in existing_face_emotions_data:
                face_emotions_id = compute_face_emotions(client, video_id, output_path=args.output_path)
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "face_emotions",
                            "face_emotions_id": face_emotions_id,
                        }
                    )
                    + "\n"
                )
            else:
                face_emotions_id = existing_face_emotions_data[input_file]["face_emotions_id"]

            if input_file not in existing_shot_size_data:
                shot_size_id = compute_shot_sizes(client, video_id, output_path=args.output_path)
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "shot_size",
                            "shot_size_id": shot_size_id,
                        }
                    )
                    + "\n"
                )
            else:
                shot_size_id = existing_shot_size_data[input_file]["shot_size_id"]

            if input_file not in existing_places_data:
                places365_id, places16_id, places3_id = compute_places(client, video_id, output_path=args.output_path)
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "places",
                            "places365_id": places365_id,
                            "places16_id": places16_id,
                            "places3_id": places3_id,
                        }
                    )
                    + "\n"
                )

            else:
                places365_id = existing_places_data[input_file]["places365_id"]
                places16_id = existing_places_data[input_file]["places16_id"]
                places3_id = existing_places_data[input_file]["places3_id"]

            if input_file not in existing_face_emotions_annotaion_data:
                face_emotions_annotation_id = compute_face_emotions_annotation(
                    client, shots_id, face_emotions_id, output_path=args.output_path
                )
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "face_emotions_annotaion",
                            "face_emotions_annotation_id": face_emotions_annotation_id,
                        }
                    )
                    + "\n"
                )
            else:
                face_emotions_annotation_id = existing_face_emotions_annotaion_data[input_file][
                    "face_emotions_annotation_id"
                ]

            if input_file not in existing_shot_size_annotaion_data:
                shot_size_annotation_id = compute_shot_sizes_annotation(
                    client, shots_id, shot_size_id, output_path=args.output_path
                )
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "shot_size_annotaion",
                            "shot_size_annotation_id": shot_size_annotation_id,
                        }
                    )
                    + "\n"
                )
            else:
                shot_size_annotation_id = existing_shot_size_annotaion_data[input_file]["shot_size_annotation_id"]

            if input_file not in existing_places_annotaion_data:
                places365_annotation_id, places16_annotation_id, places3_annotation_id = compute_places_annotation(
                    client, shots_id, places365_id, places16_id, places3_id, output_path=args.output_path
                )
                f.write(
                    json.dumps(
                        {
                            "video_id": input_file,
                            "type": "places_annotaion",
                            "places365_annotation_id": places365_annotation_id,
                            "places16_annotation_id": places16_annotation_id,
                            "places3_annotation_id": places3_annotation_id,
                        }
                    )
                    + "\n"
                )
            else:
                places365_annotation_id = existing_places_annotaion_data[input_file]["places365_annotation_id"]
                places16_annotation_id = existing_places_annotaion_data[input_file]["places16_annotation_id"]
                places3_annotation_id = existing_places_annotaion_data[input_file]["places3_annotation_id"]

    return 0


if __name__ == "__main__":
    sys.exit(main())
