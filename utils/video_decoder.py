import imageio


class VideoDecoder:
    def __init__(self, path, max_dimension=None, fps=None):
        self._path = path
        self._max_dimension = max_dimension
        self._fps = fps
        reader = imageio.get_reader(path)

        self._meta = reader.get_meta_data()
        self._size = self._meta.get("size")

        if self._fps is None:
            self._fps = self._meta.get("fps")

    def __iter__(self):

        if self._max_dimension is not None:
            res = max(self._size[1], self._size[0])
            scale = min(self._max_dimension / res, 1)
            res = (round(self._size[0] * scale), round(self._size[1] * scale))
            video_reader = imageio.get_reader(self._path, fps=self._fps, size=res)
        else:
            video_reader = imageio.get_reader(self._path, fps=self._fps)

        for i, frame in enumerate(video_reader):
            yield {"time": i / self.fps, "index": i, "frame": frame}

    def fps(self):
        return self._fps
