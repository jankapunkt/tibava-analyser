import json
import sys
import numpy as np
import cv2
import torch

from scipy.cluster.hierarchy import fclusterdata
from mxnet.runtime import Features
from RetinaFace.retinaface import RetinaFace
from arcface_torch.backbones import get_model


class RetinaFaceDetector:
    def __init__(self, retina_model='R50', arcface_model='r100',
                 gpuid=-1, thresh=0.8, network='net3'):
        models = './model/'
        retina_model = models + retina_model
        self.facedetector = RetinaFace(retina_model, 0, gpuid, network)
        self.thresh = thresh

        # prepare arcface-model
        self.arcface = get_model(arcface_model, fp16=False)
        # check if torch can use GPU or not
        device_name = 'cpu'
        if gpuid >= 0:
            device_name = 'cuda:' + str(gpuid)
        self.arcface.load_state_dict(torch.load(models + '/backbone.pth', map_location=device_name))
        self.arcface.eval()

    @torch.no_grad()
    def calculate_face_embd(self, img):
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return self.arcface(img).numpy()

    def detect_faces(self, img, do_flip=False, scales=[1024, 1980]):
        stored_faces = []
        # calculate scales
        shape = img.shape
        size_min, size_max = np.min(shape[0:2]), np.max(shape[0:2])
        scale = float(scales[0]) / float(size_min)
        # prevent bigger than maximum size
        if np.round(scale * size_max) > scales[1]:
            scale = float(scales[1]) / float(size_max)

        # get result of frame (faces = bounding box, landmarks = faceparts)
        faces, _ = self.facedetector.detect(img, threshold=self.thresh, scales=[scale], do_flip=do_flip)

        # process all found faces
        for face in faces:
            # get correct positions (box[0] = x1, box[1] = y1, box[2] = x2, box[3] = y2)
            box = face.astype(np.int)
            # calculate bbox
            x = max(0, box[0])
            y = max(0, box[1])
            w = box[2] - x
            h = box[3] - y
            area = (w * h) / (shape[0] * shape[1])
            # get box for embedding and resize to 112x112
            box_img = cv2.resize(img[y:y + h, x:x + w], (112, 112), interpolation=cv2.INTER_AREA)
            stored_faces.append(
                {
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(w),
                    "bbox_h": int(h),
                    "bbox_area": int(area),
                    "embedding": self.calculate_face_embd(box_img).tolist()[0]
                }
            )

        return stored_faces

    def draw_boxes(self, img, faces, face_embeddings, clusters):
        box_img = img.copy()
        red = (255, 0, 0)
        for i, box in enumerate(faces):
            idx = face_embeddings.index(box["embedding"])
            cv2.rectangle(box_img, (box["bbox_x"], box["bbox_y"]), (box["bbox_x"] + box["bbox_w"],
                                                                    box["bbox_y"] + box["bbox_h"]), red, 2)
            cv2.putText(box_img, "".join(["person", str(clusters[idx])]), (box["bbox_x"], box["bbox_y"] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, red, 2)

        return box_img

    def cluster_faces(self, face_embeddings, cluster_threshold=0.4, metric="cosine"):
        return fclusterdata(X=face_embeddings, t=cluster_threshold, criterion="distance", metric=metric)


import argparse
import logging
import imageio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--info", action="store_true", help="info output")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    parser.add_argument("--video", type=str, required=True, help="path to video file")
    parser.add_argument("--frames", type=int, required=False, help="stop after specified amount of frames")
    args = parser.parse_args()

    return args


# for local testing
def main():
    args = parse_args()

    level = logging.ERROR
    if args.info:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # check if mxnet can use CUDA and set gpuid
    gpuid = -1
    features = Features()
    if features.is_enabled('CUDA'):
        gpuid = 0

    detector = RetinaFaceDetector(gpuid=gpuid)
    detect = detector.detect_faces
    faces = {}

    # create video reader and writer to store edited video
    vid_reader = imageio.get_reader(args.video)
    fps = vid_reader.get_meta_data()["fps"]
    vid_writer = imageio.get_writer('result.mp4', fps=fps)
    # iterate through all frames
    for frame_idx, frame_img in enumerate(vid_reader):
        logging.debug(f"Frame Index: {frame_idx}")
        faces[frame_idx] = detect(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
        if len(faces[frame_idx]) > 0:
            logging.info(faces[frame_idx])

        if args.frames and frame_idx > args.frames:
            break

    logging.info(faces)

    # store results as json
    f = open("faces.json", "w")
    f.write(json.dumps(faces))
    f.close()

    # get all face-embeddings
    face_embeddings = []
    append = face_embeddings.append
    for faces_idx in faces.values():
        for face in faces_idx:
            append(face["embedding"])

    # get clusters
    clusters = detector.cluster_faces(face_embeddings)
    draw = detector.draw_boxes
    # draw boxes and personids on frames
    for frame_idx, frame_img in enumerate(vid_reader):
        vid_writer.append_data(draw(frame_img, faces[frame_idx], face_embeddings, clusters))
    vid_writer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
