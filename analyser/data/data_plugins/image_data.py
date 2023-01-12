import logging
import yaml
from typing import List, Union
from collections.abc import Iterator, Iterable

from dataclasses import dataclass, field, fields
import imageio.v3 as iio

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from analyser import analyser_pb2


@dataclass(kw_only=True)
class ImageData(Data):
    ref_id: str = None
    time: float = None
    delta_time: float = field(default=None)
    ext: str = field(default="jpg")


@DataManager.export("ImagesData", analyser_pb2.IMAGES_DATA)
@dataclass(kw_only=True)
class ImagesData(Data):
    images: List[ImageData] = field(default_factory=list)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("images_data.yml")
        self.images = [ImageData(**x) for x in data.get("images")]

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        self.save_dict("images_data.yml", {"images": [x.to_dict() for x in self.images]})

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "images": [x.to_dict() for x in self.images],
        }

    def __iter__(self) -> Iterator:
        yield from self.images

    def save_image(
        self, image: npt.ArrayLike, ref_id: str = None, time: float = None, delta_time: float = None
    ) -> None:
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"
        image_data = ImageData(ref_id=ref_id, time=time, delta_time=delta_time)
        try:

            encoded = iio.imwrite("<bytes>", image, extension=".jpg")
            with self.fs.open_file(f"{image_data.id}.jpg", "w") as f:
                f.write(encoded)
        except:
            logging.error("Could not add a new image")
            return None

        self.images.append(image_data)

    def load_image(self, image: Union[ImageData, str]) -> npt.ArrayLike:
        assert self.check_fs(), "No filesystem handler installed"

        image_id = image.id if isinstance(image, ImageData) else image
        try:
            with self.fs.open_file(f"{image_id.id}.jpg", "r") as f:
                return iio.imread(f.read())
        except:
            logging.error(f"Could not load a image with id {image_id}")
            return None
