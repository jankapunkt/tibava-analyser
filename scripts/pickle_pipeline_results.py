import argparse
import logging
import numpy as np
import os.path
import pickle
import yaml

from analyser.data import DataManager


def audio_amp_analysis_pkl(outputs: dict) -> dict:
    """
    Does not need to be mapped
    output -> dict
    """
    logging.debug(outputs)
    return outputs["amp"]


def audio_freq_analysis_pkl(outputs: dict) -> dict:
    """
    Does not need to be mapped
    output -> dict
    """
    logging.debug(outputs)
    return outputs["freq"]


def audio_rms_analysis_pkl(outputs: dict) -> dict:
    """
    Does not need to be mapped
    output -> dict
    """
    logging.debug(outputs)
    return outputs["rms"]


def brightness_analysis_pkl(outputs: dict) -> dict:
    """
    Does not need to be mapped
    output -> dict
    """
    logging.debug(outputs)
    return outputs["brightness"]


def camera_size_classification_pkl(outputs: dict) -> dict:
    """
    output -> dict:
        id: str
        last_access: float
        y: tx5 (float)
        time: [t] (float)
        delta_time: float
        index: [5] (str)
    """
    logging.debug(outputs)
    camera_sizes = []
    for entry in outputs["camera_size_probs"]["data"]:  # iter over camera sizes
        time = entry["time"]
        delta_time = entry["delta_time"]
        camera_sizes.append(entry["y"])

    camera_sizes = np.stack(camera_sizes, axis=-1)

    return {
        "id": outputs["camera_size_probs"]["id"],
        "last_access": outputs["camera_size_probs"]["last_access"],
        "y": camera_sizes,
        "time": time,
        "delta_time": delta_time,
        "index": outputs["camera_size_probs"]["index"],
    }


def clip_image_embeddings_pkl(outputs: dict) -> dict:
    """
    output -> dict:
        id: str
        last_access: float
        embeddings: tx512 (float)
        time: [t] (float)
        delta_time: float
    """
    logging.debug(outputs)
    time = []
    embeddings = []
    for emb in outputs["embeddings"]["image_embeddings"]:
        delta_time = emb["delta_time"]
        time.append(emb["time"])
        embeddings.append(np.squeeze(emb["embedding"]))

    return {
        "id": outputs["embeddings"]["id"],
        "last_access": outputs["embeddings"]["last_access"],
        "embeddings": np.stack(embeddings),
        "time": time,
        "delta_time": delta_time,
    }


def color_analysis_pkl(outputs: dict) -> dict:
    """
    output -> dict:
        id: str
        last_access: float
        colors: kxtx3 (float)
        time: [t] (float)
        delta_time: float
        index: [k] (str) - Number of dominant colors
    """
    logging.debug(outputs)
    colors = []
    for entry in outputs["colors"]["data"]:  # iter over camera sizes
        time = entry["time"]
        delta_time = entry["delta_time"]
        colors.append(entry["colors"])

    colors = np.stack(colors, axis=0)
    print(colors.shape)

    return {
        "id": outputs["colors"]["id"],
        "last_access": outputs["colors"]["last_access"],
        "colors": colors,
        "time": time,
        "delta_time": delta_time,
        "index": outputs["colors"]["index"],
    }


def face_analysis_pkl(outputs: dict) -> list:
    """
    output -> list [faces]:
        id - str
        time - float
        delta_time - float
        bbox
            x - float
            y - float
            w - float
            h - float
            score - float
        kpss - 5x2 (float)
        embedding 512x1 (float)
        emotion - 7x1 (float)
        age - float
        gender - 2x1 (float)
    """
    logging.debug(outputs)
    faces = {}
    for face in outputs["faces"]["faces"]:
        faces[face["id"]] = {"id": face["id"]}

    for bbox in outputs["bboxes"]["bboxes"]:
        faces[bbox["ref_id"]]["time"] = bbox["time"]
        faces[bbox["ref_id"]]["delta_time"] = bbox["delta_time"]

        faces[bbox["ref_id"]]["bbox"] = {
            "x": bbox["x"],
            "y": bbox["y"],
            "w": bbox["w"],
            "h": bbox["h"],
            "det_score": bbox["det_score"],
        }

    for kps in outputs["kpss"]["kpss"]:
        faces[kps["ref_id"]]["kps"] = np.stack((kps["x"], kps["y"]), axis=-1)

    for emb in outputs["facialfeatures"]["image_embeddings"]:
        faces[emb["ref_id"]]["embedding"] = emb["embedding"]

    for key in ["emotions", "ages", "genders"]:
        for entry in outputs[key]["data"]:  # iter over emotion
            for i in range(len(entry["ref_id"])):  # iter over faces
                ref_id = entry["ref_id"][i]
                prob = entry["y"][i]

                if key[:-1] not in faces[ref_id]:
                    faces[ref_id][key[:-1]] = []

                faces[ref_id][key[:-1]].append(prob)

        for face in faces.values():
            face[key[:-1]] = np.asarray(face[key[:-1]])

    return [face for face in faces.values()]


def face_to_camera_size_pkl(outputs: dict) -> dict:
    """
    output -> dict:
        facessizes: dict
            id: str
            last_access: float
            y: tx5 (float)
            time: [t] (float)
            delta_time: float
        camerasize_probs: dict
            id: str
            last_access: float
            y: tx5 (float)
            time: [t] (float)
            delta_time: float
            index: [5] (str)
    """
    logging.debug(outputs)

    camera_sizes = []
    for entry in outputs["camerasize_probs"]["data"]:  # iter over camera sizes
        time = entry["time"]
        delta_time = entry["delta_time"]
        camera_sizes.append(entry["y"])

    camera_sizes = np.stack(camera_sizes, axis=-1)

    return {
        "facessizes": outputs["facessizes"],
        "camerasize_probs": {
            "id": outputs["camerasize_probs"]["id"],
            "last_access": outputs["camerasize_probs"]["last_access"],
            "y": camera_sizes,
            "time": time,
            "delta_time": delta_time,
            "index": outputs["camerasize_probs"]["index"],
        },
    }


def place_classification_pkl(outputs: dict) -> dict:
    """
    output -> dict:
        place_embeddings
            id: str
            last_access: float
            embeddings: tx2048 (float)
            time: [t] (float)
            delta_time: float
        probs_places365
            id: str
            last_access: float
            y: tx365 (float)
            time: [t] (float)
            delta_time: float
            index: [365] (str)
        probs_places16
            id: str
            last_access: float
            y: tx16 (float)
            time: [t] (float)
            delta_time: float
            index: [16] (str)
        probs_places3
            id: str
            last_access: float
            y: tx3 (float)
            time: [t] (float)
            delta_time: float
            index: [3] (str)
    """
    time = []
    embeddings = []

    output_dict = {}
    for emb in outputs["place_embeddings"]["image_embeddings"]:
        delta_time = emb["delta_time"]
        time.append(emb["time"])
        embeddings.append(np.squeeze(emb["embedding"]))

    output_dict["place_embeddings"] = {
        "id": outputs["place_embeddings"]["id"],
        "last_access": outputs["place_embeddings"]["last_access"],
        "embeddings": np.stack(embeddings),
        "time": time,
        "delta_time": delta_time,
    }

    for key in ["probs_places365", "probs_places16", "probs_places3"]:
        probs = []
        for entry in outputs[key]["data"]:  # iter over camera sizes
            probs.append(entry["y"])

        probs = np.stack(probs, axis=-1)

        output_dict[key] = {
            "id": outputs[key]["id"],
            "last_access": outputs[key]["last_access"],
            "y": probs,
            "time": time,
            "delta_time": delta_time,
            "index": outputs[key]["index"],
        }

    return output_dict


def shot_density_pkl(outputs: dict) -> dict:
    """
    output -> dict:
    """
    logging.debug(outputs)
    return outputs["shot_density"]


def transnet_shotdetection_pkl(outputs: dict) -> dict:
    """
    output -> dict:
    """
    logging.debug(outputs)
    return outputs["shots"]


def write_pkl(pipeline: dict, outputs: dict, output_path: str):
    writer_f = {
        "audio_amp_analysis": audio_amp_analysis_pkl,
        "audio_freq_analysis": audio_freq_analysis_pkl,
        "audio_rms_analysis": audio_rms_analysis_pkl,
        "brightness_analysis": brightness_analysis_pkl,
        "camera_size_classification": camera_size_classification_pkl,
        "clip_image_embeddings": clip_image_embeddings_pkl,
        "color_analysis": color_analysis_pkl,
        "face_analysis": face_analysis_pkl,
        "face_to_camera_size": face_to_camera_size_pkl,
        "place_classification": place_classification_pkl,
        "shot_density": shot_density_pkl,
        "transnet_shotdetection": transnet_shotdetection_pkl,
    }

    if pipeline["pipeline"] in writer_f:
        pkl_output = writer_f[pipeline["pipeline"]](outputs)

        output_file = os.path.join(
            output_path, os.path.splitext(pipeline["video_file"])[0], pipeline["pipeline"] + ".pkl"
        )

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        with open(output_file, "wb") as f:
            pickle.dump({**pipeline, **{"output_data": pkl_output}}, f)

        print(f"File written to: {output_file}")
        return

    logging.error(f"Unknown pipeline {pipeline['pipeline']}")
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Store pipeline results in pickled dictionaries.")

    parser.add_argument(
        "-p",
        "--pipeline_results",
        type=str,
        nargs="+",
        help="(List of) .yml files containing outputs ids to the pipeline results",
    )
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder for .pkl files")
    parser.add_argument("-m", "--media_path", type=str, default="/media", help="Path to the media folder")

    parser.add_argument("--host", default="localhost", help="host name")
    parser.add_argument("--port", default=50051, help="port number")
    parser.add_argument("--debug", action="store_true", help="debug output")

    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # start client and get available plugins
    dm = DataManager(data_dir=args.media_path)

    for pipeline_file in args.pipeline_results:
        # read results
        with open(pipeline_file, "r") as f:
            pipeline = yaml.safe_load(f)

        outputs = {}
        for output in pipeline["outputs"]:
            for output_name, output_id in output.items():
                data = dm.load(output_id)
                outputs[output_name] = data.to_dict()

        write_pkl(pipeline=pipeline, outputs=outputs, output_path=args.output_path)


if "__main__" == __name__:
    main()
