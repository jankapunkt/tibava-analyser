import logging

from ..manager import DataManager
from ..data import Data
from analyser import analyser_pb2
from dataclasses import dataclass, field, fields
from collections.abc import Iterable


@DataManager.export("VideoData", analyser_pb2.VIDEO_DATA)
@dataclass(kw_only=True)
class VideoData(Data):
    type: str = field(default="VideoData")
    filename: str = None
    ext: str = None

    def load(self) -> None:
        super().load()
        data = self.load_dict("video_data.yml")
        self.filename = data.get("filename")
        self.ext = data.get("ext")

    def save(self) -> None:
        super().save()

        self.save_dict(
            "video_data.yml",
            {
                "filename": self.filename,
                "ext": self.ext,
            },
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "filename": self.filename,
            "ext": self.ext,
        }

    def open_video(self, mode="r"):
        assert self.check_fs(), "No fs register"
        return self.fs.open_file(f"video.{self.ext}", mode)

    def load_file_from_stream(self, data_stream: Iterable) -> None:

        assert self.check_fs(), "No fs register"
        assert self.fs.mode == "w", "Fs is not writeable"

        data_stream = iter(data_stream)
        first_pkg = next(data_stream)

        self.ext = first_pkg.ext
        self.filename = first_pkg.filename
        with self.open_video("w") as f:
            f.write(first_pkg.data_encoded)
            for x in data_stream:
                f.write(x.data_encoded)
                print(f"{self.ext} {self.filename}", flush=True)
