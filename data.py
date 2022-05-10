from dataclasses import dataclass
from typing import Dict, List, Any, Type
import numpy.typing as npt


@dataclass
class PluginData:
    id: str


@dataclass
class VideoData(PluginData):
    path: str


@dataclass
class ImageData(PluginData):
    path: str
    time: float = None


@dataclass
class Shot:
    start: float
    end: float


@dataclass
class ShotsData(PluginData):
    shots: List[Shot]


@dataclass
class AudioData(PluginData):
    path: str


@dataclass
class ScalarData(PluginData):
    x: npt.NDArray
    time: List[float]


@dataclass
class HistData(PluginData):
    x: npt.NDArray
    time: List[float]
