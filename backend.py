from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse, abort
import json
import logging
import os
import requests

# instantiate the app
app = Flask(__name__)

CORS(app)
app.config.from_object(__name__)
api = Api(app)

# init logging
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


# sanity check route
class Ping(Resource):
    def get(self):
        return jsonify("pong!")


api.add_resource(Ping, "/ping")

# route to run face detection on videos
vidargs = reqparse.RequestParser()
vidargs.add_argument("title", type=str, required=True, help="title of the video")
vidargs.add_argument("path", type=str, required=True, help="path to video file")
vidargs.add_argument("max_frames", type=int, required=False, help="maximum number of video frames to process")


class FaceDetection(Resource):

    # calculates and stores face detection results
    def put(self, video_id):
        args = vidargs.parse_args()

        # TODO load result from database
        # TODO assign unique ids to videos
        if os.path.exists(os.path.join("media", str(video_id) + ".json")):
            with open(os.path.join("media", str(video_id) + ".json"), "r") as jsonfile:
                results = json.load(jsonfile)
                return jsonify(results)

        # get results from submodule
        response = requests.put(f"http://facedetection:5002/detect_faces/{video_id}", args)
        results = response.json()

        with open(os.path.join("media", str(video_id) + ".json"), "w") as jsonfile:
            json.dump(results, jsonfile)

        return jsonify(results)


api.add_resource(FaceDetection, "/detect_faces/<int:video_id>")

if __name__ == "__main__":
    app.run(debug=False)
