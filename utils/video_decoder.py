import imageio


class VideoDecoder:
    def __init__(self, path, max_dimension=None, fps=None):
        self.path = path
        self.max_dimension = max_dimension
        self.fps = fps
        reader = imageio.get_reader(path)

        self.meta = reader.get_meta_data()
        self.size = self.meta.get("size")

        if self.fps is None:
            self.fps = self.meta.get("fps")

    def __iter__(self):

        if self.max_dimension is not None:
            res = max(self.size[1], self.size[0])
            scale = min(self.max_dimension / res, 1)
            res = (round(self.size[0] * scale), round(self.size[1] * scale))
            video_reader = imageio.get_reader(self.path, fps=self.fps, size=res)
        else:
            video_reader = imageio.get_reader(self.path, fps=self.fps)

        for i, frame in enumerate(video_reader):
            yield {"time": i / self.fps, "index": i, "frame": frame}
