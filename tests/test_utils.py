import tempfile
import os
import zipfile
import io
import imageio as iio
import librosa
import soundfile
import av
import imageio.v3 as iio

from analyser.data import DataManager
from analyser.utils import VideoDecoder

test_path = os.path.dirname(os.path.abspath(__file__))


def test_video_decoder():
    with open(os.path.join(test_path, "xg_mascara.mp4"), "rb") as f_in:
        fh = av.open(os.path.join(test_path, "xg_mascara.mp4"))
        fh = av.open(f_in)
        stream = fh.streams.video[0]
        frame = next(fh.decode(video=0))
        frame = frame.reformat(format="rgb24")
        print(dir(fh))
        print(frame)
        print(dir(frame))
        print("########")
        print(frame.interlaced_frame)
        print(frame.format)
        print(frame.planes)
        print(frame.pict_type)

        print(frame.width)
        print(frame.height)
        print(frame.to_ndarray().shape)
        iio.imwrite(os.path.join(test_path, "test_out.jpg"), frame.to_ndarray())
        print(stream.duration)
        print(stream.time_base)
        print(fh.size)
        print(float(stream.duration * stream.time_base))
        print(stream.average_rate)
        print(stream.guessed_rate)

        assert False

    # tmp_dir = tempfile.mkdtemp()
    # data_manager = DataManager(data_dir=tmp_dir)

    # data_path = None
    # with data_manager.create_data("AudioData") as data:
    #     data_id = data.id
    #     data_path = os.path.join(tmp_dir, data.id[0:2], data.id[2:4], f"{data.id}.zip")
    #     data.ext = "wav"
    #     with data.open_audio(mode="w") as f_out:
    #         with open(os.path.join(test_path, "xg_mascara.wav"), "rb") as f_in:
    #             while True:
    #                 data = f_in.read(128 * 1024)
    #                 if len(data) == 0:
    #                     break
    #                 f_out.write(data)

    # read_zipfile_dir(data_path)

    # with data_manager.load(data_id) as data:
    #     assert data.type == "AudioData"
    #     assert data.id == data_id
    #     assert data.ext == "wav"
    #     with data.open_audio() as f:
    #         # raw_data = f.read()
    #         tmp = io.BytesIO(f.read())
    #         # tmp.name = "a.mp3"
    #         y, sr = soundfile.read(tmp)
    #         assert y.shape[0] == 24001
