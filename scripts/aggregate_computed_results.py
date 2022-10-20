from tqdm import tqdm
import numpy as np
import pickle
import shutil
import os
from pathlib import Path
from datetime import datetime

COLOR_PLUGIN_NAME = "colors"
COLOR_BRIGHTNESS_PLUGIN_NAME = "brightness"
FRQ_PLUGIN_NAME = "audio_freq"
AMP_PLUGIN_NAME = "audio_amp"
RMS_PLUGIN_NAME = "audio_rms"
PLACES_PLUGIN_NAME = "places"
SHOT_TYPE_PLUGIN_NAME = "shot_types"
SHOT_DETECTION_PLUGIN_NAME = "shots"
SHOT_DENSITY_PLUGIN_NAME = "shot_density"
CLIP_PLUGIN_NAME = "clip_embeddings"
FACE_DETECTION_PLUGIN_NAME = "insightface_detection"
FACE_EMOTION_PLUGIN_NAME = "deepface_emotion"
FACE_SIZES_PLUGIN_NAME = "insightface_facesize"
FACE_EMBEDDING_PLUGIN_NAME = "insightface_embeddings"

def pickle_data(data, outpath, plugin_name):
    os.makedirs(outpath, exist_ok=True)
    fname = str(outpath / plugin_name) + ".pkl"
    try:
        with open(fname, "wb+") as f:
            pickle.dump(data, f)
    except Exception as e:
        print("Did not write to", fname, e)

def load_pickle(path):
    with open(str(path) + ".pkl", "rb") as f:
        return pickle.load(f)

def _new_dict_by_meta(old_dict):
    new = {}
    new["id"] = old_dict["id"]
    new["last_access"] = datetime.fromtimestamp(old_dict["last_access"])
    new["video_path"] = str(old_dict["video_path"])
    return new

def convert_face_information(inpath, outpath):
    """
    output:
        bboxes
            time - Nx1
            positions - Nx4 xywh
            scores - Nx1
            delta_time - float
        kpss
            position - Nx5x2
            time - Nx1
            delta_time - float
        emotions
            index - 1x7
            time - Nx1
            delta_time - float
            y - Nx7
        embeddings
            time - Nx1
            delta_time - float
            embeddings - Nx512
        size
            shot_probs Nx5
            facesize - float
            index - 1x5
            time - Nx1
            delta_time - float

    """
    face_data = load_pickle(inpath / FACE_DETECTION_PLUGIN_NAME)
    emotion_data = load_pickle(inpath / FACE_EMOTION_PLUGIN_NAME)
    embedding_data = load_pickle(inpath / FACE_EMBEDDING_PLUGIN_NAME)
    size_data = load_pickle(inpath / FACE_SIZES_PLUGIN_NAME)

    new_face_data = _new_dict_by_meta(face_data)
    new_face_data["bboxes"] = {} # position, time, delta_time, scores
    new_face_data["kpss"] = {} # position, time, delta_time
    new_face_data["emotions"] = {} # data, index
    new_face_data["embeddings"] = {} # time, embeddings
    new_face_data["size"] = {} # shot_probs, time, delta_time, index, facesize

    # FACE DETECTION DATA

    new_face_data["bboxes"]["delta_time"] = face_data["bboxes"][0]["delta_time"]
    times = []
    positions = []
    scores = []
    for item in face_data["bboxes"]:
        times.append(item["time"])
        positions.append([item["x"], item["y"], item["w"], item["h"]])
        scores.append(item["det_score"])

    new_face_data["bboxes"]["time"] = np.stack(times)
    new_face_data["bboxes"]["positions"] = np.stack(positions)
    new_face_data["bboxes"]["scores"] = np.stack(scores)

    times = []
    positions = []
    for item in face_data["kpss"]:
        times.append(item["time"])
        positions.append([item["x"], item["y"]])

    new_face_data["kpss"]["time"] = np.stack(times)
    new_face_data["kpss"]["positions"] = np.stack(positions)
    # results in Nx5x2
    new_face_data["kpss"]["positions"] = np.swapaxes(new_face_data["kpss"]["positions"], 1,2)
   
   # FACE EMOTION DATA

    new_face_data["emotions"]["index"] = emotion_data["index"]
    new_face_data["emotions"]["delta_time"] = emotion_data["data"][0]["delta_time"]

    times = []
    ys = []
    for item in emotion_data["data"]:
        ys.append(item["y"])
        times.append(item["time"])

    new_face_data["emotions"]["time"] = times[0]
    new_face_data["emotions"]["y"] = np.stack(ys).T


    # FACE EMBEDDING DATA

    new_face_data["embeddings"]["delta_time"] = embedding_data["image_embeddings"][0]["delta_time"]
    embeddings = []
    times = []
    for item in embedding_data["image_embeddings"]:
        embeddings.append(item["embedding"])
        times.append(item["time"])

    new_face_data["embeddings"]["embeddings"] = np.stack(embeddings)
    new_face_data["embeddings"]["time"] = np.stack(times)

    # FACE SIZES AND PREDICT SHOTS DATA

    new_face_data["size"]["index"] = size_data["index"]
    new_face_data["size"]["delta_time"] = size_data["delta_time"]
    new_face_data["size"]["time"] = np.array(size_data["time"])

    probs = []
    for item in size_data["probs"]:
        probs.append(item["y"])
    new_face_data["size"]["shot_probs"] = np.stack(probs).T
    new_face_data["size"]["facesize"]  = size_data["y"]

    pickle_data(new_face_data, outpath, FACE_DETECTION_PLUGIN_NAME)

def convert_clip(inpath, outpath):
    """
    delta_time - float
    time - 1xN
    embeddings - Nx512
    """
    clip_data = load_pickle(inpath / CLIP_PLUGIN_NAME)
    new_clip_data = _new_dict_by_meta(clip_data)
    new_clip_data["delta_time"] = clip_data["image_embeddings"][0]["delta_time"]
    embeddings = []
    times = []
    for item in clip_data["image_embeddings"]:
        embeddings.append(item["embedding"])
        times.append(item["time"])

    new_clip_data["embeddings"] = np.stack(embeddings).squeeze(1)
    new_clip_data["time"] = np.stack(times)
    
    pickle_data(new_clip_data, outpath, CLIP_PLUGIN_NAME)

def convert_shots(inpath, outpath):
    """
    segments - Sx2 (start end)
    density - 1xN
    delta_time - float
    time - 1xN
    """
    # shots segments and shot density
    shots_data = load_pickle(inpath / SHOT_DETECTION_PLUGIN_NAME)
    new_shots_data = _new_dict_by_meta(shots_data)
    density_data = load_pickle(inpath / SHOT_DENSITY_PLUGIN_NAME)

    segments = []
    for item in shots_data["shots"]:
        segments.append([item["start"], item["end"]])

    new_shots_data["segments"] = np.stack(segments)
    new_shots_data["density"] = np.array(density_data["y"])
    new_shots_data["delta_time"] = np.array(density_data["delta_time"])
    new_shots_data["time"] = np.array(density_data["time"])
    pickle_data(new_shots_data, outpath, SHOT_DETECTION_PLUGIN_NAME)

def convert_shot_type(inpath, outpath):
    """
    out dict:
        time : 1xN
        delta_time : float
        index : 1x5
        y : Nx5
    """
    shot_type_data = load_pickle(inpath / SHOT_TYPE_PLUGIN_NAME)
    new_shot_type_data = _new_dict_by_meta(shot_type_data)
    new_shot_type_data["index"] = shot_type_data["index"]
    new_shot_type_data["time"] = shot_type_data["data"][0]["time"]
    ys = []
    for i in range(len(shot_type_data["index"])):
        ys.append(shot_type_data["data"][i]["y"])

    new_shot_type_data["y"] = np.stack(ys).T

    pickle_data(new_shot_type_data, outpath, SHOT_TYPE_PLUGIN_NAME)
    
def convert_places(inpath, outpath):
    """
    places_C has C labels
    places_C data has C list entries: a dict with y
    y should be 1xN, N being the number of frames

    out dict:
        embeddings : Nxd
        time : 1xN
        delta_time : float
        places3 : { y : Nx3, index: 1x3}
        places16 : { y : Nx16, index: 1x16}
        places365 : { y : Nx365, index: 1x365}
    """
    places_data = load_pickle(inpath / PLACES_PLUGIN_NAME)
    new_places_data = _new_dict_by_meta(places_data)
    # should be the same across places3,16,265
    new_places_data["delta_time"] = places_data["places365_data"][0]["delta_time"]
    embs = []
    times = []
    for item in places_data["image_embeddings"]:
        embs.append(item["embedding"])
        times.append(item["time"])

    new_places_data["embeddings"] = np.stack(embs).squeeze(1)
    new_places_data["time"] = np.stack(times)

    for c in (3,16,365):
        ys = []
        for i in range(c):
            ys.append(places_data[f"places{c}_data"][i]["y"])
            
        assert len(new_places_data["time"]) == len(ys[0]), f"{len(new_places_data['time'])} {len(ys[0])}"

        new_places_data[f"places{c}"] = {
            "y" : np.stack(ys).T,
            "index" : places_data[f"places{c}_index"]
        }

    pickle_data(new_places_data, outpath, PLACES_PLUGIN_NAME)

def convert_audio(inpath, outpath):
    # audio does not need to be changed
    try:
        shutil.copy(
            str(inpath / FRQ_PLUGIN_NAME) + ".pkl",
            str(outpath / FRQ_PLUGIN_NAME) + ".pkl",
        )
    except Exception as e:
        print("Did not write to", FRQ_PLUGIN_NAME, e)
    
    try:
        shutil.copy(
            str(inpath / AMP_PLUGIN_NAME) + ".pkl",
            str(outpath / AMP_PLUGIN_NAME) + ".pkl"  
        )
    except Exception as e:
        print("Did not write to", AMP_PLUGIN_NAME, e)

    try:
        shutil.copy(
            str(inpath / RMS_PLUGIN_NAME) + ".pkl",
            str(outpath / RMS_PLUGIN_NAME) + ".pkl"  
        )
    except Exception as e:
        print("Did not write to", RMS_PLUGIN_NAME, e)


def convert_color(inpath, outpath):
    color_data = load_pickle(inpath / COLOR_PLUGIN_NAME)
    new_color_data = _new_dict_by_meta(color_data)
    new_color_data["colors"] = color_data["data"][0]["colors"]
    new_color_data["delta_time"] = color_data["data"][0]["delta_time"]
    new_color_data["time"] = color_data["data"][0]["time"]
    pickle_data(new_color_data, outpath, COLOR_PLUGIN_NAME)

    # nothing to change with this plugin
    shutil.copy(
        str(inpath / COLOR_BRIGHTNESS_PLUGIN_NAME) + ".pkl",
        str(outpath / COLOR_BRIGHTNESS_PLUGIN_NAME) + ".pkl"  
    )


def main():
    base_path = Path("/nfs/home/rhotertj/datasets/fake_narratives/")
    media = ["BildTV", "compact", "tagesschau"]
    for medium in media:
        video_ids = os.listdir(base_path / "raw" / medium)
        for video_id in tqdm(video_ids):
            src = base_path/ "raw" / medium / video_id
            target = base_path/ "converted" / medium / video_id
            convert_color(src, target)
            convert_audio(src, target)
            convert_places(src, target)
            convert_shots(src, target)
            convert_shot_type(src, target)
            convert_clip(src, target)
            convert_face_information(src, target)



if __name__ == "__main__":
    main()