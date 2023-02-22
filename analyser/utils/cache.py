import hashlib
import json
import logging
from typing import List, Dict


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


def get_hash_for_plugin(
    plugin: str, output: str, version: str = None, parameters: List = [], inputs: List = [], config: Dict = {}
):
    plugin_call_dict = {
        "plugin": plugin,
        "output": output,
        "parameters": parameters,
        "inputs": inputs,
        "config": config,
        "version": version,
    }
    # logging.info(f"[HASH] {plugin_call_dict}")

    # logging.info(f"[HASH] {flat_dict(plugin_call_dict)}")
    plugin_hash = hashlib.sha256(
        json.dumps(
            flat_dict(
                {
                    "plugin": plugin,
                    "output": output,
                    "parameters": parameters,
                    "inputs": inputs,
                    "config": config,
                    "version": version,
                }
            )
        ).encode()
    ).hexdigest()

    # logging.info(f"[HASH] {plugin_hash}")
    return plugin_hash
