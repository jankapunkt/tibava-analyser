from celery import Celery
import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse, abort
import grpc
import hashlib
import imageio
import json
import logging
import msgpack
import os
import redis
import requests
import sys
import traceback
import yaml

# own imports
_CUR_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(_CUR_PATH, "facedetection"))
sys.path.append(os.path.join(_CUR_PATH, "shotdetection"))
from utils import export_to_csv, export_to_jsonl, export_to_shoebox

import facedetection_pb2, facedetection_pb2_grpc
import shotdetection_pb2, shotdetection_pb2_grpc


# read config
with open(os.path.join(_CUR_PATH, "config.yml")) as f:
    _CFG = yaml.load(f, Loader=yaml.FullLoader)

# instantiate the appmsgpack 1.0.2
app = Flask(__name__)
app.config.from_object(__name__)
app.config["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
api = Api(app)


def make_celery(app):
    celery = Celery(
        app.import_name, backend=app.config["CELERY_RESULT_BACKEND"], broker=app.config["CELERY_BROKER_URL"]
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


celery = make_celery(app)

# enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# init logging
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# arguments for getting meta information from a video file
metaargs = reqparse.RequestParser()
metaargs.add_argument("title", type=str, required=True, help="title of the video")
metaargs.add_argument("path", type=str, required=True, help="path to video file")
metaargs.add_argument("max_frames", type=int, required=False, help="maximum number of video frames to process")

# arguments for videos
videoargs = reqparse.RequestParser()
videoargs.add_argument("video_id", type=str, required=True, help="id of the video")
videoargs.add_argument("path", type=str, required=True, help="path to the video")
videoargs.add_argument("max_frames", type=int, required=False, help="maximum number of video frames to process")

# arguments for jobs
jobargs = reqparse.RequestParser()
jobargs.add_argument("job_id", type=str, required=True, help="id of the celery job")
jobargs.add_argument("fps", type=float, required=False, help="fps for time conversion")

# arguments for data conversion
converterargs = reqparse.RequestParser()
converterargs.add_argument("video_id", type=str, required=True, help="video id")
converterargs.add_argument("input_data", type=list, required=True, location="json", help="input data")
converterargs.add_argument("task", type=str, required=True, help="name of the task")
converterargs.add_argument(
    "format", type=str, required=True, choices=["csv", "jsonl", "shoebox"], help="format to convert to"
)

converterargs.add_argument("keys_to_store", type=list, required=False, location="json", help="keys to store")
converterargs.add_argument("ELANBegin_key", type=str, required=False, help="ELANBegin key for shoebox")
converterargs.add_argument("ELANEnd_key", type=str, required=False, help="ELANEnd key for shoebox")


# sanity check route
class Ping(Resource):
    def get(self):
        return jsonify("pong!")


api.add_resource(Ping, "/ping")


class MetaReader(Resource):

    # extracts metainformation (fps, codec, etc) from video
    def get(self):
        args = metaargs.parse_args()
        vid_reader = imageio.get_reader(args["path"])
        metadata = vid_reader.get_meta_data()

        # supplement metadata
        metadata["path"] = args["path"]
        metadata["title"] = args["title"]

        # create unique hash based on metadata
        meta_hash = hashlib.sha256(msgpack.packb(metadata)).hexdigest()

        # return metainformation and meta_hash as video_id
        return jsonify({"video_id": meta_hash, "metadata": metadata})


api.add_resource(MetaReader, "/read_meta")


class DataConverter(Resource):

    # converts data list (entries are dicts) to specified file format
    def post(self):
        args = converterargs.parse_args()
        output_file = None

        if len(args.input_data) < 1:
            fname = f"{args.video_id}_{args.task}.{args.format}"
            with open(os.path.join(_CFG["media_folder"], fname), "w") as f:
                f.write("")
            return jsonify({"status": "SUCCESS", "output_file": os.path.join(_CFG["media_folder"], fname)})

        input_entry = args.input_data[0]
        if not isinstance(input_entry, dict):
            logging.error("First entry in input data is not of type dict!")
            return jsonify({"status": "ERROR", "output_file": None})

        input_keys = input_entry.keys()

        if args.format == "csv":
            output_file = export_to_csv(
                video_id=args.video_id,
                input_data=args.input_data,
                media_folder=_CFG["media_folder"],
                task=args.task,
                keys=args.keys_to_store,
            )

        elif args.format == "jsonl":
            output_file = export_to_jsonl(
                video_id=args.video_id,
                input_data=args.input_data,
                media_folder=_CFG["media_folder"],
                task=args.task,
            )

        elif args.format == "shoebox":

            if args.ELANBegin_key in input_keys and args.ELANEnd_key in input_keys:
                output_file = export_to_shoebox(
                    video_id=args.video_id,
                    input_data=args.input_data,
                    media_folder=_CFG["media_folder"],
                    task=args.task,
                    ELANBegin_key=args.ELANBegin_key,
                    ELANEnd_key=args.ELANEnd_key,
                )
            else:
                logging.error("Key(s) for ELANBegin or ELANEnd not provided or not in input")
                return jsonify({"status": "ERROR", "output_file": None})
        else:
            logging.error("Unknown conversion format!")
            return jsonify({"status": "ERROR", "output_file": None})

        if output_file is not None and os.path.exists(output_file):
            return jsonify({"status": "SUCCESS", "output_file": output_file})
        else:
            return jsonify({"status": "ERROR", "output_file": None})


api.add_resource(DataConverter, "/export_data")


class ShotDetection(Resource):

    # posts job to detect cuts/shots in videos
    def post(self):
        args = videoargs.parse_args()

        # assign task
        task = shot_detection_task.apply_async((args,))
        return jsonify({"status": "PENDING", "job_id": task.id})

    # gets result from posted jobs for cut/shot detection
    def get(self):
        args = jobargs.parse_args()
        try:
            task = shot_detection_task.AsyncResult(args.job_id)
        except Exception as e:
            logging.warning(e)
            logging.warning(traceback.format_exc())
            return jsonify({"status": "ERROR", "msg": "job is unknown"})

        if task.info is None:
            return jsonify({"status": "PENDING"})

        if task.state == "SUCCESS":
            status = task.info.get("status")
            video_id = task.info.get("video_id")
            shots = task.info.get("shots")

            if args.fps:
                for shot in shots:

                    shot["start_time"] = (
                        (datetime.datetime.min + datetime.timedelta(seconds=shot["start_frame"] / args.fps))
                        .time()
                        .isoformat(timespec="milliseconds")
                    )

                    shot["end_time"] = (
                        (datetime.datetime.min + datetime.timedelta(seconds=shot["end_frame"] / args.fps))
                        .time()
                        .isoformat(timespec="milliseconds")
                    )

            return jsonify({"status": status, "video_id": video_id, "shots": shots})

        elif task.state == "PENDING":
            return jsonify({"status": "PENDING", "msg": task.info.get("msg"), "code": task.info.get("code")})

        return jsonify(
            {
                "status": "ERROR",
                "msg": "job crashed",
            }
        )


@celery.task(bind=True)
def shot_detection_task(self, args):
    """
    Celery task for shot detection in videos
    """

    # copy video to server
    copy_status = copy_video_to_grpc_server(video_id=args["video_id"], video_path=args["path"])
    if not copy_status:
        return {"status": "ERROR", "shots": []}

    # open grpc channel
    channel = grpc.insecure_channel(f"[::]:{_CFG['shotdetection']['port']}")
    stub = shotdetection_pb2_grpc.ShotDetectorStub(channel)

    # check if shots are already extracted
    r = redis.Redis()
    cache_result = r.get(f"shots_{args['video_id']}")

    if cache_result:  # load results from cache and return
        logging.info(f"Loading shot detection results for {args['video_id']} from cache ...")
        cache_unpacked = msgpack.unpackb(cache_result)
        return {"status": "SUCCESS", "shots": cache_unpacked["shots"]}

    # calculate shot results if no cached result
    logging.info(f"Calculate shot detection results for {args['video_id']} ...")
    self.update_state(
        state="PENDING",
        meta={"msg": "Detect cuts in video ..."},
    )
    try:
        response = stub.get_shots(shotdetection_pb2.ShotRequest(video_id=args["video_id"]))

        if not response.success:
            logging.error(f"Error while detecting shots ...")
            return {"status": "ERROR", "shots": []}

        shots = []
        for shot in response.shots:
            shots.append({"shot_id": shot.shot_id, "start_frame": shot.start_frame, "end_frame": shot.end_frame})

        # write to cache and return
        logging.info(f"Store shot detection results for {args['video_id']} in cache ...")
        r.set(f"shots_{args['video_id']}", msgpack.packb({"shots": shots}))

        return {"status": "SUCCESS", "shots": shots}

    except Exception as e:
        logging.error(f"Error while detecting shots: {repr(e)}")
        logging.error(traceback.format_exc())
        return {"status": "ERROR", "shots": []}


def copy_video_to_grpc_server(video_id, video_path):
    channel = grpc.insecure_channel(f"[::]:{_CFG['shotdetection']['port']}")
    stub = shotdetection_pb2_grpc.ShotDetectorStub(channel)

    # check if video already exists on target server
    r = redis.Redis()
    cache_result = r.get(f"videofile_{video_id}")

    def generateRequests(video_id, file_object, chunk_size=1024):
        """Lazy function (generator) to read a file piece by piece.
        Default chunk size: 1k"""
        with open(file_object, "rb") as videobytestream:
            while True:
                data = videobytestream.read(chunk_size)
                if not data:
                    break
                yield shotdetection_pb2.VideoRequest(video_id=video_id, video_encoded=data)

    if cache_result:
        logging.info(f"Video with id {video_id} already stored ...")
        return True
    else:
        logging.info(f"Transferring video with id {video_id} to shotdetection server ...")
        response = stub.copy_video(generateRequests(video_id, video_path))

        if response.success:
            r.set(f"videofile_{video_id}", msgpack.packb({"stored": True}))
            return True

        logging.error("Error while copying video ...")
        return False


api.add_resource(ShotDetection, "/detect_shots")


# route to run face detection on videos
class FaceDetection(Resource):

    # posts job to detect faces in videos
    def post(self):
        args = videoargs.parse_args()

        # assign task
        task = face_detection_task.apply_async((args,))
        return jsonify({"status": "PENDING", "job_id": task.id})

    # gets result from posted jobs for face detection
    def get(self):
        args = jobargs.parse_args()
        try:
            task = face_detection_task.AsyncResult(args.job_id)
        except Exception as e:
            logging.warning(e)
            logging.warning(traceback.format_exc())
            return jsonify({"status": "ERROR", "msg": "job is unknown"})

        if task.info is None:
            return jsonify({"status": "PENDING"})

        if task.state == "SUCCESS":
            return jsonify(
                {
                    "status": task.info.get("status"),
                    "video_id": task.info.get("video_id"),
                    "faces": task.info.get("faces"),
                    "max_num_faces": task.info.get("max_num_faces"),
                }
            )

        elif task.state == "PENDING":
            return jsonify({"status": "PENDING", "msg": task.info.get("msg"), "code": task.info.get("code")})

        return jsonify(
            {
                "status": "ERROR",
                "msg": "job crashed",
            }
        )


@celery.task(bind=True)
def face_detection_task(self, args):
    """
    Celery task for face detection in videos
    """

    # copy video to server
    copy_status = copy_video_to_grpc_server(video_id=args["video_id"], video_path=args["path"])
    if not copy_status:
        return {"status": "ERROR", "faces": []}

    # open grpc channel
    channel = grpc.insecure_channel(f"[::]:{_CFG['facedetection']['port']}")
    stub = facedetection_pb2_grpc.FaceDetectorStub(channel)

    # check if faces are already extracted
    r = redis.Redis()
    cache_result = r.get(f"faces_{args['video_id']}")

    if cache_result:  # load results from cache and return
        logging.info(f"Loading face detection results for {args['video_id']} from cache ...")
        cache_unpacked = msgpack.unpackb(cache_result)
        return {"status": "SUCCESS", "faces": cache_unpacked["faces"], "max_num_faces": cache_unpacked["max_num_faces"]}

    # calculate face detection results if no cached result
    logging.info(f"Calculate face detection results for {args['video_id']} ...")
    self.update_state(
        state="PENDING",
        meta={"msg": "Detect faces in video ..."},
    )
    try:
        response = stub.detect_faces(facedetection_pb2.FaceRequest(video_id=args["video_id"]))

        if not response.success:
            logging.error(f"Error while detecting faces ...")
            return {"status": "ERROR", "faces": []}

        # convert faces to list
        faces = []
        for face in response.faces:
            faces.append(
                {
                    "frame_idx": face.frame_idx,
                    "bbox_xywh": (face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    "bbox_area": face.bbox_area,
                }
            )

        # write to cache and return
        logging.info(f"Store face detection results for {args['video_id']} in cache ...")
        r.set(f"faces_{args['video_id']}", msgpack.packb({"faces": faces, "max_num_faces": response.max_num_faces}))

        return {"status": "SUCCESS", "faces": faces, "max_num_faces": response.max_num_faces}

    except Exception as e:
        logging.error(f"Error while detecting faces: {repr(e)}")
        logging.error(traceback.format_exc())
        return {"status": "ERROR", "faces": []}

    # TODO


api.add_resource(FaceDetection, "/detect_faces")

if __name__ == "__main__":
    app.run(debug=False)
