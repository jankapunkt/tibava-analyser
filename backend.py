from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse, abort
import logging
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
        print(args)
        # TODO load result from database if video already processed

        # get results from submodule
        return requests.put(f"http://facedetection:5002/detect_faces/{video_id}", args)


api.add_resource(FaceDetection, "/detect_faces/<int:video_id>")

if __name__ == "__main__":
    app.run(debug=False)
