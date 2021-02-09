import dlib
import logging


class DLIBFaceDetector():
    def __init__(self, model="HOG"):
        if model == "HOG":
            self.facedetector = dlib.get_frontal_face_detector()
        elif model == "DNN":
            self.facedetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
        else:
            logging.warning("Unknown face detector model! Using HOG detector instead.")
            self.facedetector

    def detect_faces(self, img):
        faces = []
        for rect in self.facedetector(img, 0):
            x = max(0, rect.left())
            y = max(0, rect.top())
            w = rect.right() - x
            h = rect.bottom() - y

            faces.append((x, y, w, h))

        return faces


"""
Test class
"""
import argparse
import imageio
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--info', action='store_true', help='info output')
    parser.add_argument('-vv', '--debug', action='store_true', help='debug output')
    parser.add_argument('--video', type=str, required=True, help='path to video file')
    parser.add_argument('--frames', type=int, required=False, help='stop after specified amount of frames')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.ERROR
    if args.info:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)

    FaceDetector = DLIBFaceDetector()

    faces = {}
    vid_reader = imageio.get_reader(args.video)
    for frame_idx, frame_img in enumerate(vid_reader):
        logging.debug(f"Frame Index: {frame_idx}")
        faces[frame_idx] = FaceDetector.detect_faces(frame_img)

        if len(faces[frame_idx]) > 0:
            logging.info(faces[frame_idx])

        if args.frames and frame_idx > args.frames:
            break

    logging.info(faces)
    return faces


if __name__ == '__main__':
    sys.exit(main())
