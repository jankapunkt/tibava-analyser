# import cv2
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse, abort
import imageio
import logging
import numpy as np
import os
import sys
import pickle 

from flask_cors import CORS

from facedetector.dlib_facedetector import DLIBFaceDetector

# instantiate the app
app = Flask(__name__)

CORS(app)
app.config.from_object(__name__)
api = Api(app)

# init logging
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# init face detector
FaceDetector = DLIBFaceDetector()
video_analysis_filename = "video_analysis_dict_main.pkl"
videos = pickle.load(open(video_analysis_filename,"rb"))

# sanity check route
class Ping(Resource):
    def get(self):
        return jsonify("pong!")


api.add_resource(Ping, "/ping")

# route to gather news image and text
vidargs = reqparse.RequestParser()
vidargs.add_argument("title", type=str, required=True, help="title of the video")
vidargs.add_argument("path", type=str, required=True, help="path to video file")
vidargs.add_argument("max_frames", type=int, required=False, help="maximum number of video frames to process")


class FaceDetection(Resource):

    # gets pre-computed face detection results
    def get(self, video_id):
        if video_id not in videos:
            abort(404, message=f"Video with id {video_id} does not exist ...")

        return jsonify(video_id)

    # calculates and stores face detection results
#    @cross_origin(origin='*',headers=['Content-Type','Authorization'])
    def put(self, video_id):
        args = vidargs.parse_args()
        if(video_id in videos):
            return videos[video_id]
        faces, fps, max_n_faces = _detect_faces(video_path=args["path"], max_frames=args["max_frames"])
        videos[video_id] = {"id": video_id, "title": args["title"], "path": args["path"], "face_detection": faces, "fps": fps, "max_n_faces":max_n_faces}
        pickle.dump(videos,open(video_analysis_filename,"wb"))
        return videos[video_id]


def _detect_faces(video_path, max_frames):
    faces = {}

    # loop through video frames and get faces
    vid_reader = imageio.get_reader(video_path)
    fps = vid_reader.get_meta_data()['fps']
    max_n_faces = 0
    for frame_idx, frame_img in enumerate(vid_reader):
        faces[frame_idx] = FaceDetector.detect_faces(frame_img)
        n_faces = len(faces[frame_idx])
        if(n_faces > max_n_faces):
            max_n_faces = n_faces
        if max_frames and frame_idx >= (max_frames - 1):
            break

    return faces, fps, max_n_faces


api.add_resource(FaceDetection, "/detect_faces/<int:video_id>")

if __name__ == "__main__":
    app.run(debug=False)
