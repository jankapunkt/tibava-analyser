from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import os
import sys
sys.path.append(".")
import argparse
import logging


from analyser.client import AnalyserClient


def parse_args():
    parser = argparse.ArgumentParser(description="Convert predictions from .msg files into pickled dictionaries.")

    parser.add_argument("-v", "--video", help="Path to video.")
    parser.add_argument("-o", "--out_path", help="Directory to store pickled results.")
    args = parser.parse_args()
    return args


def compute_face_detection(client, video_id, output_path):
    job_id = client.run_plugin("insightface_video_detector", [{"id": video_id, "name": "video"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    bboxes_id = None
    for output in result.outputs:
        if output.name == "bboxes":
            bboxes_id = output.id

    data_boxes = client.download_data(bboxes_id, output_path)
    data = data_boxes.to_dict()
    data["bboxes"] = data.pop("bboxes")

    kpss_id = None
    for output in result.outputs:
        if output.name == "kpss":
            kpss_id = output.id

    data_kpss = client.download_data(kpss_id, output_path)
    data_kpss = data_kpss.to_dict()
    data["kpss"] = data_kpss["kpss"]

    images_id = None
    for output in result.outputs:
        if output.name == "images":
            images_id = output.id

    data_images = client.download_data(images_id, output_path)
    data_images = data_images.to_dict()
    data["images"] = data_images["images"]

    return data, "insightface_detection", {"kpss_id": kpss_id, "bboxes_id" : bboxes_id, "images_id" : images_id}

def compute_face_sizes(client, video_id, output_path, bboxes_id=None, images_id=None, kpss_id=None):    
    if bboxes_id == None:
        # run detection
        job_id = client.run_plugin("insightface_video_detector", [{"id": video_id, "name": "video"}], [])

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            logging.error("Job is crashing")
            return

        bboxes_id = None
        for output in result.outputs:
            if output.name == "bboxes":
                bboxes_id = output.id

    job_id = client.run_plugin("insightface_facesize", [{"id": bboxes_id, "name": "bboxes"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get shot size prediction
    output_id_probs = None
    output_id_facesizes = None
    for output in result.outputs:
        if output.name == "probs":
            output_id_probs = output.id
        if output.name == "facesizes":
            output_id_facesizes = output.id

    data_probs = client.download_data(output_id_probs, output_path).to_dict() # list data
    data_sizes = client.download_data(output_id_facesizes, output_path) # scalar data

    data = data_sizes.to_dict()
    data["probs"] = data_probs["data"]
    data["index"] = data_probs["index"]

    return data, "insightface_facesize", {}

def compute_facial_features(client, video_id, output_path, bboxes_id=None, images_id=None, kpss_id=None):
    if kpss_id == None:
        # run detection
        job_id = client.run_plugin("insightface_video_detector", [{"id": video_id, "name": "video"}], [])

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            logging.error("Job is crashing")
            return

        kpss_id = None
        for output in result.outputs:
            if output.name == "kpss":
                kpss_id = output.id

    job_id = client.run_plugin("insightface_video_feature_extractor", [{"id": video_id, "name": "video"}, {"id" : kpss_id, "name" : "kpss"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    features_id = None
    for output in result.outputs:
        if output.name == "features":
            features_id = output.id
    data = client.download_data(features_id, output_path)
    return data.to_dict(), "insightface_embeddings", {}

def compute_emotions(client, video_id, output_path, bboxes_id=None, images_id=None, kpss_id=None):
    if images_id == None:
        # run detection
        job_id = client.run_plugin("insightface_video_detector", [{"id": video_id, "name": "video"}], [])

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            logging.error("Job is crashing")
            return

        images_id = None
        for output in result.outputs:
            if output.name == "images":
                images_id = output.id

    # deepface_emotion
    job_id = client.run_plugin("deepface_emotion", [{"id": images_id, "name": "images"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get emotions
    output_id = None
    for output in result.outputs:
        if output.name == "probs":
            output_id = output.id

    data = client.download_data(output_id, output_path)

    return data.to_dict(), "deepface_emotion", {}

def compute_clip_embeddings(client, video_id, output_path):
    job_id = client.run_plugin("clip_image_embedding", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job clip_image_embedding started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    embd_id = None
    for output in result.outputs:
        if output.name == "embeddings":
            embd_id = output.id
            break
    img_data = client.download_data(embd_id, output_path)
    return img_data.to_dict(), "clip_embeddings", {}

def compute_shot_segments(client, video_id, output_path):

    job_id = client.run_plugin("transnet_shotdetection", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job video_to_audio started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    shots_id = None
    
    for output in result.outputs:
        if output.name == "shots":
            shots_id = output.id

    data = client.download_data(shots_id, output_path, save_meta=False)

    return data.to_dict(), output.name, {"shots_id": shots_id}

def compute_shot_density(client, video_id, output_path, shots_id=None):
    
    if shots_id == None:
        job_id = client.run_plugin("transnet_shotdetection", [{"id": video_id, "name": "video"}], [])

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            logging.error("Job is crashing")
            return

        shots_id = None
        
        for output in result.outputs:
            if output.name == "shots":
                shots_id = output.id
    
    job_id = client.run_plugin("shot_density", [{"id": shots_id, "name": "shots"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        return

    shot_density_id = None
    for output in result.outputs:
        if output.name == "shot_density":
            shot_density_id = output.id

    data = client.download_data(shot_density_id, output_path)
    return data.to_dict(), output.name, {}

def compute_shot_type(client, video_id, output_path):
    job_id = client.run_plugin("shot_type_classifier", [{"id": video_id, "name": "video"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    output_id = None
    for output in result.outputs:
        if output.name == "probs":
            output_id = output.id

    data = client.download_data(output_id, output_path)
    return data.to_dict(), "shot_types", {}

def convert_v2a(client, video_id):
    job_id = client.run_plugin("video_to_audio", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job video_to_audio started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    audio_id = None
    for output in result.outputs:
        if output.name == "audio":
            audio_id = output.id
    return audio_id

def compute_freq_analysis(client, audio_id, output_path):
    job_id = client.run_plugin("audio_rms_analysis", [{"id": audio_id, "name": "audio"}], [])
    logging.info(f"Job audio_rms started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    rms_id = None
    for output in result.outputs:
        if output.name == "rms":
            rms_id = output.id

    data = client.download_data(rms_id, output_path)
    return data.to_dict(), "audio_freq", {}

def compute_amp_analysis(client, audio_id, output_path):

    job_id = client.run_plugin("audio_amp_analysis", [{"id": audio_id, "name": "audio"}], [])
    logging.info(f"Job video_to_audio started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    amp_id = None
    
    for output in result.outputs:
        if output.name == "amp":
            amp_id = output.id

    data = client.download_data(amp_id, output_path, save_meta=False)

    return data.to_dict(), "audio_amp", {}

def compute_rms_analysis(client, audio_id, output_path):
    job_id = client.run_plugin("audio_rms_analysis", [{"id": audio_id, "name": "audio"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    rms_id = None
    for output in result.outputs:
        if output.name == "rms":
            rms_id = output.id

    data = client.download_data(rms_id, output_path)
    return data.to_dict(), "audio_rms", {}

def compute_color_brightness_analyser(client, video_id, output_path):
    job_id = client.run_plugin("color_brightness_analyser", [{"id": video_id, "name": "video"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    output_id = None
    for output in result.outputs:
        if output.name == "brightness":
            output_id = output.id

    data = client.download_data(output_id, output_path)
    return data.to_dict(), output.name, {}

def compute_color_analyser(client, video_id, output_path):
    
    job_id = client.run_plugin("color_analyser", [{"id": video_id, "name": "video"}], [])
    logging.info(f"Job color_analyser started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    output_id = None
    for output in result.outputs:
        if output.name == "colors":
            output_id = output.id

    data = client.download_data(output_id, output_path)
    return data.to_dict(), output.name, {}

def compute_places_analyser(client, video_id, output_path):
    job_id = client.run_plugin("places_classifier", [{"id": video_id, "name": "video"}], [])

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    embeddings_id = None
    places365_id = None
    places16_id = None
    places3_id = None

    for output in result.outputs:
        if output.name == "embeddings":
            embeddings_id = output.id
        if output.name == "probs_places365":
            places365_id = output.id
        if output.name == "probs_places16":
            places16_id = output.id
        if output.name == "probs_places3":
            places3_id = output.id

    data_emb = client.download_data(embeddings_id, output_path)
    data = data_emb.to_dict()
    data["image_embeddings"] = data.pop("image_embeddings")

    data365 = client.download_data(places365_id, output_path)
    data365 = data365.to_dict()
    data["places365_data"] = data365["data"]
    data["places365_index"] = data365["index"]


    data16 = client.download_data(places16_id, output_path)
    data16 = data16.to_dict()
    data["places16_data"] = data16["data"]
    data["places16_index"] = data16["index"]


    data3 = client.download_data(places3_id, output_path)
    data3 = data3.to_dict()
    data["places3_data"] = data3["data"]
    data["places3_index"] = data3["index"]

    return data, "places", {}


    
def analyse_video(client, video_path, out_path):
    """Runs all plugins and computes their preconditions if necessary for a given video and an established client.
    Results will be saved as a pickled dictionary. 

    Args:
        client (AnalyserClient): A client that can communicate with the backend.
        video_path (Path): The path of the video to be analysed.
        out_path (Path): The destination for results.
    """
    print("Uploading file...")
    video_id = client.upload_file(video_path)
    print("Done! ID:" ,video_id)
    print("Prepare audio...")
    audio_id = convert_v2a(client, video_id)
    print("Done!")

    try:
        os.makedirs(out_path / video_id)
    except:
        pass
    
    requires_audio = [compute_amp_analysis, compute_rms_analysis, compute_freq_analysis]

    requires_shots = [compute_shot_density]

    requires_faces = [compute_facial_features,
        compute_face_sizes,
        compute_emotions]

    plugin_funcs = [
        compute_amp_analysis,
        compute_freq_analysis,
        compute_rms_analysis,
        compute_color_analyser,
        compute_color_brightness_analyser,
        compute_clip_embeddings,
        compute_places_analyser,
        compute_shot_segments,
        compute_shot_density,
        compute_shot_type,
        compute_face_detection,
        compute_facial_features,
        compute_face_sizes,
        compute_emotions,
    ]

    print("Running plugins...")
    func_iter = tqdm(plugin_funcs)
    # cache will save ids for previously computed features
    cache = {}
    for plugin in func_iter:

        func_iter.set_description(plugin.__name__)
        if plugin in requires_audio:
            data, plugin_type, new_cache = plugin(client, audio_id, "/tmp")
        elif plugin in requires_shots:
            data, plugin_type, new_cache = plugin(client, audio_id, "/tmp", shots_id=cache.get("shots_id"))
        elif plugin in requires_faces:
            data, plugin_type, new_cache = plugin(client, video_id, "/tmp", bboxes_id=cache.get("bboxes_id"), images_id=cache.get("images_id"), kpss_id=cache.get("kpss_id"))
        else:
            data, plugin_type, new_cache = plugin(client, video_id, "/tmp")

        cache.update(new_cache)

        data["video_path"] = video_path

        pickle_data(data, video_id, plugin_type, out_path)
    
    
def pickle_data(out_dict, video_id, plugin_type, out_path):
    filename =  out_path / video_id / f"{plugin_type}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(out_dict, f)

def main():
    args = parse_args()
    out_path = Path(args.out_path)
    video_path = Path(args.video)
    client = AnalyserClient(host="localhost", port=54051)
    analyse_video(client, video_path, out_path)

if "__main__" == __name__:
    main()