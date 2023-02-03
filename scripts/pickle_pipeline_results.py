import argparse
import logging
import numpy as np
import os.path
import pickle
import yaml

from analyser.data import DataManager


def print_data_info(key, val):
    str = f"{key}, {type(val)}"
    if isinstance(val, list):
        str += f", length {len(val)}"
    if isinstance(val, np.ndarray):
        str += f", shape {val.shape}"

    logging.debug(str)


def audio_amp_analysis_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline audio_amp_analysis

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            ref_id (str): hash id of the reference data package in TIB-AV-A
            y (np.ndarray): 1 amp value for t entries in time (shape t,)
            time (list): time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict == Output dict")
    for key, val in outputs["amp"].items():
        print_data_info(key, val)

    return outputs["amp"]


def audio_freq_analysis_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline audio_freq_analysis

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            hist (np.ndarray): n frequency histogram values for t enties in time (n x t)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict == Output dict")
    for key, val in outputs["freq"].items():
        print_data_info(key, val)

    return outputs["freq"]


def audio_rms_analysis_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline audio_rms_analysis

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            y (np.ndarray): 1 rms value for t entries in time (shape t,)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict == Output dict")
    for key, val in outputs["rms"].items():
        print_data_info(key, val)

    return outputs["rms"]


def brightness_analysis_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline brightness_analysis

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            y (np.ndarray): 1 brightness value for t entries in time (shape t,)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict == Output dict")
    for key, val in outputs["brightness"].items():
        print_data_info(key, val)

    return outputs["brightness"]


def camera_size_classification_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline camera_size_classification

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            y (np.ndarray): 5 camera size probabilities for t entries in time (shape t, 5)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
            index (list): class names for 5 camera sizes (length 5)
    """
    logging.debug("#### Input dict")
    for key, val in outputs["camera_size_probs"].items():
        print_data_info(key, val)

    camera_sizes = []
    for entry in outputs["camera_size_probs"]["data"]:  # iter over camera sizes
        time = entry["time"]
        delta_time = entry["delta_time"]
        camera_sizes.append(entry["y"])

    camera_sizes = np.stack(camera_sizes, axis=-1)

    output_dict = {
        "id": outputs["camera_size_probs"]["id"],
        "y": camera_sizes,
        "time": time,
        "delta_time": delta_time,
        "index": outputs["camera_size_probs"]["index"],
    }

    logging.debug("#### Output dict")
    for key, val in output_dict.items():
        print_data_info(key, val)

    return output_dict


def clip_image_embeddings_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline clip_image_embeddings

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            embeddings (np.ndarray): 512-dimension clip feature vector for t entries in time (shape t, 512)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict")
    for key, val in outputs["embeddings"].items():
        print_data_info(key, val)

    time = []
    embeddings = []
    for emb in outputs["embeddings"]["embeddings"]:
        delta_time = emb["delta_time"]
        time.append(emb["time"])
        embeddings.append(np.squeeze(emb["embedding"]))

    output_dict = {
        "id": outputs["embeddings"]["id"],
        "embeddings": np.stack(embeddings),
        "time": time,
        "delta_time": delta_time,
    }

    logging.debug("#### Output dict")
    for key, val in output_dict.items():
        print_data_info(key, val)

    return output_dict


def xclip_video_embeddings_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline xclip_image_embeddings

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            embeddings (np.ndarray): 512-dimension clip feature vector for t entries in time (shape t, 512)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict for video embeddings")
    for key, val in outputs["video_embeddings"].items():
        print_data_info(key, val)

    logging.debug("#### Input dict for image embeddings")
    for key, val in outputs["image_embeddings"].items():
        print_data_info(key, val)

    time = []
    video_embeddings = []
    for emb in outputs["video_embeddings"]["embeddings"]:
        delta_time = emb["delta_time"]
        time.append(emb["time"])
        video_embeddings.append(np.squeeze(emb["embedding"]))

    image_embeddings = []
    for emb in outputs["image_embeddings"]["embeddings"]:
        image_embeddings.append(np.squeeze(emb["embedding"]))

    output_dict = {
        "video_embeddings_id": outputs["video_embeddings"]["id"],
        "video_embeddings": np.stack(video_embeddings),
        "image_embeddings_id": outputs["image_embeddings"]["id"],
        "image_embeddings": np.stack(image_embeddings),
        "time": time,
        "delta_time": delta_time,
    }

    logging.debug("#### Output dict")
    for key, val in output_dict.items():
        print_data_info(key, val)

    return output_dict


def color_analysis_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline color_analysis

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            colors (np.ndarray): 3 color values (RGB) for t entries in time and k color clusters (shape k, t, 3)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
            index (list): index of k color cluster created with k-means (length k)
    """
    logging.debug("#### Input dict")
    for key, val in outputs["colors"].items():
        print_data_info(key, val)

    colors = []
    for entry in outputs["colors"]["data"]:  # iter over camera sizes
        time = entry["time"]
        delta_time = entry["delta_time"]
        colors.append(entry["colors"])

    colors = np.stack(colors, axis=0)

    output_dict = {
        "id": outputs["colors"]["id"],
        "colors": colors,
        "time": time,
        "delta_time": delta_time,
        "index": outputs["colors"]["index"],
    }

    logging.debug("#### Output dict")
    for key, val in output_dict.items():
        print_data_info(key, val)

    return output_dict


def face_analysis_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline face_analysis

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            faces (dict): dictionary containing a list of faces
                id (str): hash id of the data package in TIB-AV-A
                time (list): t time values (length t)
                delta_time (float): time duration for which a certain value is created (equals 1 / fps)
                bbox (dict): dictionary containing the bounding box for the face
                    x (float): x-coordinate of the face normalized by the image width
                    y (float): y-coordinate of the face normalized by the image height
                    w (float): width of the face normalized by the image width
                    h (float): height of the face normalized by the image height
                    det_score (float): likelihood of the bounding box beeing a face
                kps (np.ndarray): 5 keypoints with x, y location in the image (shape 5, 2)
                embedding (np.ndarray): 512-dimensional facial feature vector (shape 512,)
                emotion (np.ndarray): probability p for 7 facial emotions (shape 7,)
                    ("p_angry", "p_disgust", "p_fear", "p_happy", "p_sad", "p_surprise", "p_neutral")
                age (np.ndarray): estimated age / 100 (shape 1,)
                gender (np.ndarray): probabilities for female and male (shape 2,)
    """
    logging.debug("#### Input dict")
    for dkey in ["faces", "bboxes", "kpss", "facialfeatures", "emotions"]:  # , "ages", "genders"]:
        for key, val in outputs[dkey].items():
            print_data_info(key, val)

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

    for emb in outputs["facialfeatures"]["embeddings"]:
        faces[emb["ref_id"]]["embedding"] = emb["embedding"]

    for key in ["emotions"]:  # , "ages", "genders"]:
        for entry in outputs[key]["data"]:  # iter over emotion
            for i in range(len(entry["ref_id"])):  # iter over faces
                ref_id = entry["ref_id"][i]
                prob = entry["y"][i]

                if key[:-1] not in faces[ref_id]:
                    faces[ref_id][key[:-1]] = []

                faces[ref_id][key[:-1]].append(prob)

        for face in faces.values():
            face[key[:-1]] = np.asarray(face[key[:-1]])

    logging.debug("#### Output dict")
    output_dict = {"faces": [face for face in faces.values()]}
    for key, val in output_dict["faces"][0].items():
        print_data_info(key, val)

    return [output_dict]


def face_to_camera_size_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline face_to_camera_size

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl

            camerasize_probs (dict): probabilities for camera shot sizes
                id (str): hash id of the data package in TIB-AV-A
                y (np.ndarray): 5 camera size probabilities for t entries in time (shape t, 5)
                time (list): t time values (length t)
                delta_time (float): time duration for which a certain value is created (equals 1 / fps)
                index (list): class names for 5 camera sizes (length 5)

            facessizes (dict): size of faces found in the video
                id (str): hash id of the data package in TIB-AV-A
                ref_id (str): hash id of the reference data package in TIB-AV-A
                y (list): size of the largest face for t entries in time (length t)
                time (list): t time values (length t)
                delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict")
    for dkey in ["camerasize_probs", "facessizes"]:
        for key, val in outputs[dkey].items():
            print_data_info(key, val)

    camera_sizes = []
    for entry in outputs["camerasize_probs"]["data"]:  # iter over camera sizes
        time = entry["time"]
        delta_time = entry["delta_time"]
        camera_sizes.append(entry["y"])

    camera_sizes = np.stack(camera_sizes, axis=-1)

    output_dict = {
        "camerasize_probs": {
            "id": outputs["camerasize_probs"]["id"],
            "y": camera_sizes,
            "time": time,
            "delta_time": delta_time,
            "index": outputs["camerasize_probs"]["index"],
        },
        "facessizes": outputs["facessizes"],
    }

    logging.debug("#### Output dict")
    for dkey in ["camerasize_probs", "facessizes"]:
        for key, val in output_dict[dkey].items():
            print_data_info(key, val)

    return output_dict


def place_classification_pkl(outputs: dict) -> dict:
    """
    output -> dict:
        place_embeddings
            id: str
            embeddings: tx2048 (float)
            time: [t] (float)
            delta_time: float
        probs_places365
            id: str
            y: tx365 (float)
            time: [t] (float)
            delta_time: float
            index: [365] (str)
        probs_places16
            id: str
            y: tx16 (float)
            time: [t] (float)
            delta_time: float
            index: [16] (str)
        probs_places3
            id: str
            y: tx3 (float)
            time: [t] (float)
            delta_time: float
            index: [3] (str)
    """
    time = []
    embeddings = []

    output_dict = {}
    for emb in outputs["place_embeddings"]["embeddings"]:
        delta_time = emb["delta_time"]
        time.append(emb["time"])
        embeddings.append(np.squeeze(emb["embedding"]))

    output_dict["place_embeddings"] = {
        "id": outputs["place_embeddings"]["id"],
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
            "y": probs,
            "time": time,
            "delta_time": delta_time,
            "index": outputs[key]["index"],
        }

    return output_dict


def shot_density_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline shot_density

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            ref_id (str): hash id of the reference data package in TIB-AV-A
            y (np.ndarray): 1 brightness value for t entries in time (shape t,)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
    """
    logging.debug("#### Input dict == Output dict")
    for key, val in outputs["shot_density"].items():
        print_data_info(key, val)
    return outputs["shot_density"]


def transnet_shotdetection_pkl(outputs: dict) -> dict:
    """Converts outputs from the TIB-AV-A pipeline transnet_shotdetection

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            id (str): hash id of the data package in TIB-AV-A
            shots (list): list of n shots containing a dictionary with (length n)
                start (float): start time of the shot
                end (float): end time of the shot
    """
    logging.debug("#### Input dict == Output dict")
    for key, val in outputs["shots"].items():
        print_data_info(key, val)

    for key, val in outputs["shots"]["shots"][0].items():
        print_data_info(key, val)

    return outputs["shots"]


def write_pkl(pipeline: dict, outputs: dict, output_path: str):
    writer_f = {
        "audio_amp_analysis": audio_amp_analysis_pkl,
        "audio_freq_analysis": audio_freq_analysis_pkl,
        "audio_rms_analysis": audio_rms_analysis_pkl,
        "brightness_analysis": brightness_analysis_pkl,
        "camera_size_classification": camera_size_classification_pkl,
        "clip_image_embeddings": clip_image_embeddings_pkl,
        "xclip_video_embeddings": xclip_video_embeddings_pkl,
        "color_analysis": color_analysis_pkl,
        "face_analysis": face_analysis_pkl,
        "face_to_camera_size": face_to_camera_size_pkl,
        "place_classification": place_classification_pkl,
        "shot_density": shot_density_pkl,
        "transnet_shotdetection": transnet_shotdetection_pkl,
    }

    if pipeline["pipeline"] in writer_f:
        logging.info(f"Writing results for pipeline {pipeline['pipeline']}")
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
        valid_pipeline = True
        for output in pipeline["outputs"]:
            for output_name, output_id in output.items():
                try:
                    with dm.load(output_id) as data:
                        outputs[output_name] = data.to_dict()
                except Exception as e:
                    logging.error(f"Data package with id {output_id} does not exist")
                    valid_pipeline = False
                    continue

        if valid_pipeline:
            write_pkl(pipeline=pipeline, outputs=outputs, output_path=args.output_path)
        else:
            logging.error(f"Cannot write pkl file for pipeline {pipeline_file}")


if "__main__" == __name__:
    main()
