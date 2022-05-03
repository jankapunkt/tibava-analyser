import imageio


class VideoDecoder:
    def __init__(self, path, max_dimension=None, fps=None):
        self.path = path
        self.max_dimension = max_dimension
        self.fps = fps
        reader = imageio.get_reader(path)

        self.meta = reader.get_meta_data()
        self.frame_count = reader.count_frames()
        print(self.meta)
        print(self.frame_count)

    def __iter__(self):

        fps = self.config.get("fps", 1)

        max_resolution = config.get("max_resolution")
        if max_resolution is not None:
            res = max(video.get("height"), video.get("width"))
            scale = min(max_resolution / res, 1)
            res = (round(video.get("width") * scale), round(video.get("height") * scale))
            video_reader = imageio.get_reader(video_file, fps=fps, size=res)
        else:
            video_reader = imageio.get_reader(video_file, fps=fps)

        os.makedirs(os.path.join(config.get("output_path"), hash_id), exist_ok=True)
        results = []
        for i, frame in enumerate(video_reader):
            thumbnail_output = os.path.join(config.get("output_path"), hash_id, f"{i}.jpg")
            imageio.imwrite(thumbnail_output, frame)
            results.append({"time": i / fps, "path": f"{i}.jpg"})
        pass
