import os
import sys
import re
import argparse
import logging
import ray
from ray import serve
from ray.serve.drivers import DefaultgRPCDriver
from ray.serve.handle import RayServeDeploymentHandle
from ray.serve.deployment_graph import InputNode
from typing import Dict
import struct
from ray.serve import Application

from analyser.data import DataManager
from analyser.inference.plugin import AnalyserPluginManager, AnalyserPlugin


@serve.deployment
class Deployment:
    def __init__(self, plugin: AnalyserPlugin, data_manager: DataManager) -> None:
        self.plugin = plugin
        self.data_manager = data_manager

    async def __call__(self, request) -> Dict[str, str]:
        data = await request.json()
        inputs = data.get("inputs")
        parameters = data.get("parameters")
        logging.error("###############")
        logging.error(inputs)
        logging.error(parameters)
        logging.error("###############")

        plugin_inputs = {}
        for name, id in inputs.items():
            data = self.data_manager.load(id)
            plugin_inputs[name] = data

        results = self.plugin(plugin_inputs, data_manager=self.data_manager, parameters=parameters)

        return {x: y.id for x, y in results.items()}


def app_builder(args) -> Application:
    logging.warning(args)
    data_manager = DataManager(args.get("data_path"))
    manager = AnalyserPluginManager()
    plugin = manager.build_plugin(args.get("model"), args.get("params", {}))

    return Deployment.bind(plugin, data_manager)
