import csv
import json
import logging
import os


def export_to_csv(video_id, input_data, media_folder, task, keys=None):
    fname = f"{video_id}_{task}.csv"

    entry = input_data[0]
    if not keys:  # use all keys in entry of input data
        keys = entry.keys()

    for key in keys:  # check if all keys exist
        if key not in entry:
            logging.error(f"Key {key} not in input data")
            return None

    with open(os.path.join(media_folder, fname), "w") as f:
        writer = csv.writer(f, delimiter=",")

        # write header
        writer.writerow(keys)

        # write content
        for entry in input_data:
            writer.writerow([entry[key] for key in keys])

    return os.path.join(media_folder, fname)


def export_to_jsonl(video_id, input_data, media_folder, task):

    fname = f"{video_id}_{task}.jsonl"
    with open(os.path.join(media_folder, fname), "w") as f:

        for entry in input_data:
            f.write(json.dumps(entry) + "\n")

    return os.path.join(media_folder, fname)


def export_to_shoebox(video_id, input_data, media_folder, task, ELANBegin_key, ELANEnd_key):

    fname = f"{video_id}_{task}.sht"
    with open(os.path.join(media_folder, fname), "w") as f:

        f.write("\\_sh v3.0  400  ElanExport" + "\n")
        f.write("\\_DateStampHasFourDigitYear" + "\n")
        f.write("\n")
        f.write("\\ELANExport" + "\n")

        for cnt, entry in enumerate(input_data):
            f.write("\n")
            # TODO proper formatting
            f.write(f"\\block {(cnt + 1):04d}" + "\n")
            f.write(f"\\{task}" + "\n")
            f.write(f"\\ELANBegin {entry[ELANBegin_key]}" + "\n")
            f.write(f"\\ELANEnd {entry[ELANEnd_key]}" + "\n")

    return os.path.join(media_folder, fname)