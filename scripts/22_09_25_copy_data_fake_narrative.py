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
    print(args)

    video_data = {}
    with open(args.prediction_path, "r") as f:
        for line in f:
            d = json.loads(line)

            if d["video_id"] not in video_data:
                video_data[d["video_id"]] = {}

            video_data[d["video_id"]][d["type"]] = d

    os.makedirs(args.output_path, exist_ok=True)

    src_data_manager = DataManager(data_dir=args.data_path)

    dst_data_manager = DataManager(data_dir=args.output_path)
    for k, v in video_data.items():
        print(k)
        if "shot" in v:
            data = src_data_manager.load(v["shot"]["shots_id"])
            print(dst_data_manager.save(data))

        if "face_emotions" in v:
            data = src_data_manager.load(v["face_emotions"]["face_emotions_id"])
            print(dst_data_manager.save(data))
        if "shot_size" in v:
            data = src_data_manager.load(v["shot_size"]["shot_size_id"])
            print(dst_data_manager.save(data))
        if "places" in v:
            data = src_data_manager.load(v["places"]["places365_id"])
            print(dst_data_manager.save(data))
            data = src_data_manager.load(v["places"]["places16_id"])
            print(dst_data_manager.save(data))
            data = src_data_manager.load(v["places"]["places3_id"])
            print(dst_data_manager.save(data))

        if "face_emotions_annotaion" in v:
            data = src_data_manager.load(v["face_emotions_annotaion"]["face_emotions_annotation_id"])
            print(dst_data_manager.save(data))

        if "places_annotaion" in v:
            data = src_data_manager.load(v["places_annotaion"]["places365_annotation_id"])
            print(dst_data_manager.save(data))
            data = src_data_manager.load(v["places_annotaion"]["places16_annotation_id"])
            print(dst_data_manager.save(data))
            data = src_data_manager.load(v["places_annotaion"]["places3_annotation_id"])
            print(dst_data_manager.save(data))

        if "shot_size_annotaion" in v:
            data = src_data_manager.load(v["shot_size_annotaion"]["shot_size_annotation_id"])
            print(dst_data_manager.save(data))

    return 0


if __name__ == "__main__":
    sys.exit(main())
