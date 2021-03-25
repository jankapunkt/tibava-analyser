from celery import Celery
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse, abort
import grpc
import imageio
import json
import logging
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

# instantiate the app
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

# arguments for videos
vidargs = reqparse.RequestParser()
vidargs.add_argument("title", type=str, required=True, help="title of the video")
vidargs.add_argument("path", type=str, required=True, help="path to video file")
vidargs.add_argument("max_frames", type=int, required=False, help="maximum number of video frames to process")

# arguments for data conversion
converterargs = reqparse.RequestParser()
converterargs.add_argument("input", type=dict, required=True, help="input dictionary")
converterargs.add_argument("dictkey", type=str, required=True, help="dictkey providing the data")
converterargs.add_argument("format", type=str, required=True, choices=["shoebox"], help="format to convert to")


# sanity check route
class Ping(Resource):
    def get(self):
        return jsonify("pong!")


api.add_resource(Ping, "/ping")


class MetaReader(Resource):

    # calculates and stores face detection results
    def get(self, video_id):
        args = vidargs.parse_args()
        vid_reader = imageio.get_reader(args["path"])
        metadata = vid_reader.get_meta_data()
        return jsonify({"id": video_id, "title": args["title"], "path": args["path"], "metadata": metadata})


api.add_resource(MetaReader, "/read_meta/<int:video_id>")


class DataConverter(Resource):
    def get(self, video_id):
        args = converterargs.parse_args()

        output_file = None
        if args.format == "shoebox":
            output_file = export_to_shoebox(video_id=video_id, input_dict=args.input, dictkey=args.dictkey)

        if output_file is not None and os.path.exists(output_file):
            return jsonify({"status": "SUCCESS", "output_file": output_file})
        else:
            return jsonify({"status": "ERROR", "output_file": None})


api.add_resource(DataConverter, "/export_data/<int:video_id>")


# route to run shot detection on videos
class ShotDetection(Resource):

    # calculates and stores face detection results
    def post(self, video_id):
        args = vidargs.parse_args()

        # assign task
        task = shot_detection_task.apply_async((args,))
        return jsonify({"status": "PENDING", "job_id": task.id})

    def get(self, job_id):
        try:
            task = shot_detection_task.AsyncResult(job_id)
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

            # TODO convert shots to python dict
            logging.info(shots)
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

    def generateRequests(file_object, chunk_size=1024):
        """Lazy function (generator) to read a file piece by piece.
        Default chunk size: 1k"""
        with open(file_object, "rb") as videobytestream:
            while True:
                data = videobytestream.read(chunk_size)
                if not data:
                    break
                yield shotdetection_pb2.VideoRequest(video_encoded=data)

    self.update_state(
        state="PENDING",
        meta={"msg": "Copy video to shotdetection server ..."},
    )

    response = stub.copy_video(generateRequests(args["path"]))
    video_id = response.video_id

    self.update_state(
        state="PENDING",
        meta={"msg": "Detect cuts in video ..."},
    )

    response = stub.get_shots(shotdetection_pb2.ShotRequest(video_id=video_id))

    shots = []
    for shot in response.shots:
        shots.append({"shot_id": shot.shot_id, "start_frame": shot.start_frame, "end_frame": shot.end_frame})

    return {"status": "SUCCESS", "shots": shots, "video_id": video_id}


api.add_resource(ShotDetection, "/detect_shots/<int:video_id>", "/detect_shots/<string:job_id>")


# route to run face detection on videos
class FaceDetection(Resource):

    # calculates and stores face detection results
    def put(self, video_id):
        args = vidargs.parse_args()
        outfile = os.path.join("media", str(video_id) + "_faces.json")

        # TODO load result from proper database
        # TODO assign unique ids to videos
        if os.path.exists(outfile):
            with open(outfile, "r") as jsonfile:
                results = json.load(jsonfile)
                return jsonify(results)

        # get results from submodule
        response = requests.put(f"http://facedetection:5002/detect_faces/{video_id}", args)
        results = response.json()

        with open(outfile, "w") as jsonfile:
            json.dump(results, jsonfile)

        return jsonify(results)


api.add_resource(FaceDetection, "/detect_faces/<int:video_id>")

if __name__ == "__main__":
    app.run(debug=False)
