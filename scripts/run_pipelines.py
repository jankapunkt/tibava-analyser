import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import yaml

from analyser.data import DataManager
from analyser.client import AnalyserClient

import copy


def flat_dict(data_dict, parse_json=False):
    result_map = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            embedded = flat_dict(v)
            for s_k, s_v in embedded.items():
                s_k = f"{k}.{s_k}"
                if s_k in result_map:
                    logging.error(f"flat_dict: {s_k} alread exist in output dict")

                result_map[s_k] = s_v
            continue

        if k not in result_map:
            result_map[k] = []
        result_map[k] = v
    return result_map


def get_hash_for_plugin(plugin, parameters=[], inputs=[]):
    return hashlib.sha256(
        json.dumps(
            flat_dict(
                {
                    "plugin": plugin,
                    "parameters": parameters,
                    "inputs": inputs,
                }
            )
        ).encode()
    ).hexdigest()


def analyse_video(client: AnalyserClient, video_path: Path, output_path: Path, pipelines: dict, available_plugins: set):
    """Runs all plugins and computes their preconditions if necessary for a given video and an established client.
    Results will be saved as a pickled dictionary.

    Args:
        client (AnalyserClient): A client that can communicate with the backend.
        video_path (Path): The path of the video to be analysed.
        output_path (Path): The destination for results.
        pipelines (dict): Pipeline dictionary containing the plugins to run and their parameters.
        available_plugins (set): Set of available analyser plugins.
    """
    # Upload Video
    logging.info("Uploading file...")
    video = client.upload_file(video_path)  # NOTE must match the input of the pipeline
    video_file = os.path.basename(video_path)
    video_fname = os.path.splitext(video_file)[0]
    logging.info(f"Done! ID: {video}")

    cache = {}
    for pipeline in pipelines:
        # print(pipeline["outputs"])
        logging.info(f"{pipeline['pipeline']}: STARTED!")
        logging.debug(pipeline)

        data = {
            "video_file": video_file,
            "video_name": video_fname,
            "video_id": video,  # first requirement for all plugins
        }

        for entry in pipeline["inputs"]:
            if entry not in data:
                logging.error("input data missing!")

        for plugin in pipeline["plugins"]:
            # check if plugin is available
            if plugin["plugin"] not in available_plugins:
                logging.error(f"{pipeline['pipeline']}/{plugin['plugin']}: Unknown plugin")
                continue

            logging.info(f"{pipeline['pipeline']}/{plugin['plugin']}: RUNNING ...")
            logging.debug(plugin)

            # create parameters and inputs
            inputs = []
            if "requires" in plugin:
                try:
                    inputs = [{"name": key, "id": data[value]} for key, value in plugin["requires"].items()]
                except Exception as e:
                    logging.error(f"{pipeline['pipeline']}/{plugin['plugin']}: Missing requirement to run plugin! {e}")
                    continue

            parameters = []
            if "parameters" in plugin:
                parameters = [{"name": key, "value": value} for key, value in plugin["parameters"].items()]

            # create plugin hash and load results from cache if already computed
            plugin_hash = get_hash_for_plugin(plugin=plugin["plugin"], parameters=parameters, inputs=inputs)
            if plugin_hash in cache:
                data = {**data, **cache[plugin_hash]}
                logging.info(f"{pipeline['pipeline']}/{plugin['plugin']}: LOADED FROM CACHE!")
                continue

            # run plugin
            job_id = client.run_plugin(plugin["plugin"], inputs, parameters)
            result = client.get_plugin_results(job_id=job_id)
            if result is None:
                logging.info(f"{pipeline['pipeline']}/{plugin['plugin']}: No result!")
                continue

            # store plugin results
            plugin_results = {}
            for output in result.outputs:
                if output.name in plugin["provides"].keys():
                    plugin_results[plugin["provides"][output.name]] = output.id
                    client.download_data(output.id, output_path)

            cache[plugin_hash] = plugin_results
            data = {**data, **plugin_results}

            logging.info(f"{pipeline['pipeline']}/{plugin['plugin']}: DONE!")

        # store results
        filename = os.path.join(output_path, video_fname, f"{pipeline['pipeline']}.yml")
        store_output_ids(copy.deepcopy(pipeline), data, filename)
        logging.info(f"{pipeline['pipeline']}: DONE! Output IDs written to {filename}")

    return True


def store_output_ids(pipeline: dict, data: dict, filename: str):

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    outputs = {}

    for out in pipeline["outputs"]:

        # print(out)
        if out not in data:
            outputs[out] = {}
            continue

        outputs[out] = data[out]

    pipeline["outputs"] = [{key: val} for key, val in outputs.items()]
    logging.debug(f"{pipeline['pipeline']}: {pipeline}")

    with open(filename, "w") as f:
        pipeline["video_file"] = data["video_file"]
        pipeline["video_id"] = data["video_id"]
        yaml.dump(pipeline, f)


def get_git_revisions_hash():
    commits = ["HEAD", "HEAD^"]
    return [subprocess.check_output(["git", "rev-parse", "{}".format(x)]) for x in commits]


def parse_args():
    parser = argparse.ArgumentParser(description="Run analyser plugins and store output ids to the results.")

    parser.add_argument("-v", "--videos", type=str, nargs="+", help="Path to video file(s)")
    parser.add_argument("-p", "--pipelines", type=str, help="Path to .yml file defining the pipelines to run")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder")

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

    # read pipeline definitions
    with open(args.pipelines, "r") as f:
        pipelines = yaml.safe_load(f)

    # start client and get available plugins
    client = AnalyserClient(host=args.host, port=args.port, manager=DataManager(data_dir=args.output_path))
    available_plugins = set()
    for plugin in client.list_plugins().plugins:
        available_plugins.add(plugin.name)

    logging.debug(available_plugins)

    # analyse videos
    for video in args.videos:
        analyse_video(
            client=client,
            video_path=Path(video),
            output_path=Path(args.output_path),
            pipelines=pipelines,
            available_plugins=available_plugins,
        )


if "__main__" == __name__:
    main()
