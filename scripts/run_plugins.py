from __future__ import annotations
from pathlib import Path
import pandas as pd
import pickle
import json
import os
import sys
import argparse

sys.path.append('.')

from pickle_plugins import analyse_video
from analyser.client import AnalyserClient


def parse_args():
    parser = argparse.ArgumentParser(description="Convert predictions from .msg files into pickled dictionaries.")

    parser.add_argument("-o", "--output_path", help="Path for the pickle files.")
    parser.add_argument("-p", "--prediction_path", help="Path of the prediction.jsonl file")
    parser.add_argument("-N", type=int, default=10, help="Number of videos")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    out_path = Path(args.output_path)
    try:
        os.makedirs(out_path)
    except:
        pass

    client = AnalyserClient(host="localhost", port=54051) # on devbox2  

    print("Reading predictions from", args.prediction_path)
    preds_df = pd.read_json(args.prediction_path, lines=True)

    video_ids = preds_df["video_id"].sample(args.N, random_state=42).tolist()

    for video in video_ids:
        analyse_video(client, Path(video), out_path=out_path)
    

if "__main__" == __name__:
    main()