import os
import sys
import re
import argparse
import json

from pympi.Elan import Eaf, to_eaf
from analyser.data import DataManager


def parse_args():
    parser = argparse.ArgumentParser(description="Export prediction to elan")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--data_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-p", "--prediction_path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_data = {}
    with open(args.prediction_path, "r") as f:
        for line in f:
            d = json.loads(line)

            if d["video_id"] not in video_data:
                video_data[d["video_id"]] = {}

            video_data[d["video_id"]][d["type"]] = d

    os.makedirs(args.output_path, exist_ok=True)

    data_manager = DataManager(data_dir=args.data_path)
    for k, v in video_data.items():
        basename = "".join(os.path.splitext(os.path.basename(k))[:-1])
        print(basename)
        eaf = Eaf(author="")
        eaf.remove_tier("default")
        eaf.add_linked_file(file_path=os.path.basename(k), mimetype="video/mp4")

        if "shot" in v:
            data = data_manager.load(v["shot"]["shots_id"])
            tier_id = "shots"
            eaf.add_tier(tier_id=tier_id)
            for i, s in enumerate(data.shots):
                start_time = int(s.start * 1000)
                end_time = int(s.end * 1000)
                eaf.add_annotation(tier_id, start=start_time, end=end_time, value=str(i))

        LABEL_LUT = {
            "p_angry": "Angry",
            "p_disgust": "Disgust",
            "p_fear": "Fear",
            "p_happy": "Happy",
            "p_sad": "Sad",
            "p_surprise": "Surprise",
            "p_neutral": "Neutral",
        }
        if "face_emotions_annotaion" in v:
            data = data_manager.load(v["face_emotions_annotaion"]["face_emotions_annotation_id"])
            tier_id = "face emotion"
            eaf.add_tier(tier_id=tier_id)
            for i, s in enumerate(data.annotations):
                start_time = int(s.start * 1000)
                end_time = int(s.end * 1000)
                eaf.add_annotation(
                    tier_id, start=start_time, end=end_time, value=",".join(LABEL_LUT[x] for x in s.labels)
                )

        if "places_annotaion" in v:
            data = data_manager.load(v["places_annotaion"]["places365_annotation_id"])
            tier_id = "places 365"
            eaf.add_tier(tier_id=tier_id)
            for i, s in enumerate(data.annotations):
                start_time = int(s.start * 1000)
                end_time = int(s.end * 1000)
                eaf.add_annotation(tier_id, start=start_time, end=end_time, value=",".join(s.labels))
            data = data_manager.load(v["places_annotaion"]["places16_annotation_id"])
            tier_id = "places 16"
            eaf.add_tier(tier_id=tier_id)
            for i, s in enumerate(data.annotations):
                start_time = int(s.start * 1000)
                end_time = int(s.end * 1000)
                eaf.add_annotation(tier_id, start=start_time, end=end_time, value=",".join(s.labels))
            data = data_manager.load(v["places_annotaion"]["places3_annotation_id"])
            tier_id = "places 3"
            eaf.add_tier(tier_id=tier_id)
            for i, s in enumerate(data.annotations):
                start_time = int(s.start * 1000)
                end_time = int(s.end * 1000)
                eaf.add_annotation(tier_id, start=start_time, end=end_time, value=",".join(s.labels))

        LABEL_LUT = {
            "p_ECU": "Extreme Close-Up",
            "p_CU": "Close-Up",
            "p_MS": "Medium Shot",
            "p_FS": "Full Shot",
            "p_LS": "Long Shot",
        }
        if "shot_size_annotaion" in v:
            data = data_manager.load(v["shot_size_annotaion"]["shot_size_annotation_id"])
            tier_id = "shot size"
            eaf.add_tier(tier_id=tier_id)
            for i, s in enumerate(data.annotations):
                start_time = int(s.start * 1000)
                end_time = int(s.end * 1000)
                eaf.add_annotation(
                    tier_id, start=start_time, end=end_time, value=",".join(LABEL_LUT[x] for x in s.labels)
                )
        to_eaf(file_path=os.path.join(args.output_path, f"{basename}.eaf"), eaf_obj=eaf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
