from celery import Celery
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
sys.path.append(os.path.join(_CUR_PATH, "shotdetection"))
from utils import export_to_shoebox
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

# arguments for data conversion
converterargs = reqparse.RequestParser()
converterargs.add_argument("video_id", type=str, required=True, help="video id")
converterargs.add_argument("input_data", type=list, required=True, location="json", help="input data")
converterargs.add_argument("ELANType_key", type=str, required=True, help="dictkey providing the data")
converterargs.add_argument("ELANBegin_key", type=str, required=True, help="dictkey providing the data")
converterargs.add_argument("ELANEnd_key", type=str, required=True, help="dictkey providing the data")
converterargs.add_argument("format", type=str, required=True, choices=["shoebox"], help="format to convert to")


# sanity check route
class Ping(Resource):
    def get(self):
        return jsonify("pong!")


api.add_resource(Ping, "/ping")


class MetaReader(Resource):

    # calculates and stores face detection results
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
    def post(self):
        args = converterargs.parse_args()
        output_file = None

        if args.format == "shoebox":
            output_file = export_to_shoebox(
                video_id=args.video_id,
                input_data=args.input_data,
                media_folder=_CFG["media_folder"],
                ELANType_key=args.ELANType_key,
                ELANBegin_key=args.ELANBegin_key,
                ELANEnd_key=args.ELANEnd_key,
            )

        if output_file is not None and os.path.exists(output_file):
            return jsonify({"status": "SUCCESS", "output_file": output_file})
        else:
            return jsonify({"status": "ERROR", "output_file": None})


api.add_resource(DataConverter, "/export_data")


# route to run shot detection on videos
class ShotDetection(Resource):

    # calculates and stores face detection results
    def post(self):
        args = videoargs.parse_args()

        # assign task
        task = shot_detection_task.apply_async((args,))
        return jsonify({"status": "PENDING", "job_id": task.id})

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
    channel = grpc.insecure_channel(f"[::]:{_CFG['shotdetection']['port']}")
    stub = shotdetection_pb2_grpc.ShotDetectorStub(channel)

    # check if video already exists on target server
    r = redis.Redis()
    cache_result = r.get(f"videofile_{args['video_id']}")

    def generateRequests(video_id, file_object, chunk_size=1024):
        """Lazy function (generator) to read a file piece by piece.
        Default chunk size: 1k"""
        with open(file_object, "rb") as videobytestream:
            while True:
                data = videobytestream.read(chunk_size)
                if not data:
                    break
                yield shotdetection_pb2.VideoRequest(video_id=video_id, video_encoded=data)

    if not cache_result:
        logging.info(f"Transferring video with id {args['video_id']} to shotdetection server ...")
        self.update_state(
            state="PENDING",
            meta={"msg": "Copy video to shotdetection server ..."},
        )
        response = stub.copy_video(generateRequests(args["video_id"], args["path"]))

        if response.success:
            r.set(f"videofile_{args['video_id']}", msgpack.packb({"stored": True}))
        else:
            logging.error("Error while copying video ...")
            return {"status": "ERROR", "shots": []}

    else:
        logging.info(f"Video with id {args['video_id']} already stored ...")

    # check if shots are already extracted
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


api.add_resource(ShotDetection, "/detect_shots")


# route to run face detection on videos
# class FaceDetection(Resource):

#     # calculates and stores face detection results
#     def put(self, video_id):
#         args = vidargs.parse_args()
#         outfile = os.path.join("media", str(video_id) + "_faces.json")

#         # TODO load result from proper database
#         # TODO assign unique ids to videos
#         if os.path.exists(outfile):
#             with open(outfile, "r") as jsonfile:
#                 results = json.load(jsonfile)
#                 return jsonify(results)

#         # get results from submodule
#         response = requests.put(f"http://facedetection:5002/detect_faces/{video_id}", args)
#         results = response.json()

#         with open(outfile, "w") as jsonfile:
#             json.dump(results, jsonfile)

#         return jsonify(results)


# api.add_resource(FaceDetection, "/detect_faces/<int:video_id>")

if __name__ == "__main__":
    app.run(debug=False)
