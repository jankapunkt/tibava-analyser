import imageio.v3 as iio


class VideoDecoder:
    # TODO: videos with sample aspect ratio (SAR) not equal to 1:1 are loaded with wrong shape
    def __init__(self, path, max_dimension=None, fps=None):
        self._path = path
        self._max_dimension = max_dimension
        self._fps = fps

        self._meta = iio.immeta(path)
        self._size = self._meta.get("size")

        if self._fps is None:
            self._fps = self._meta.get("fps")

    def __iter__(self):

        if self._max_dimension is not None:
            res = max(self._size[1], self._size[0])
            scale = min(self._max_dimension / res, 1)
            res = (round(self._size[0] * scale), round(self._size[1] * scale))
            video_reader = iio.imread(
                self._path,
                plugin="pyav",
                filter_sequence=[("fps", f"{self._fps}"), ("scale", {"width": f"{res[0]}", "height": f"{res[1]}"})],
            )
        else:
            video_reader = iio.imread(self._path, fps=self._fps, plugin="pyav")

        for i, frame in enumerate(video_reader):
            yield {"time": i / self._fps, "index": i, "frame": frame}

    def fps(self):
        return self._fps
