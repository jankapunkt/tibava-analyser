import tempfile
import os
import zipfile
import io
import imageio as iio
import librosa
import soundfile


from analyser.data import DataManager
from analyser.utils import VideoDecoder

test_path = os.path.dirname(os.path.abspath(__file__))


def read_zipfile_dir(path):
    z = zipfile.ZipFile(path, "r")
    print(z.namelist())


def test_video_data():
    tmp_dir = tempfile.mkdtemp()
    data_manager = DataManager(data_dir=tmp_dir)

    data_path = None
    with data_manager.create_data("VideoData") as data:
        data_id = data.id
        data_path = os.path.join(tmp_dir, data.id[0:2], data.id[2:4], f"{data.id}.zip")
        data.ext = "mp4"
        with data.open_video(mode="w") as f_out:
            with open(os.path.join(test_path, "test.mp4"), "rb") as f_in:
                while True:
                    data = f_in.read(128 * 1024)
                    if len(data) == 0:
                        break
                    f_out.write(data)

    read_zipfile_dir(data_path)

    with data_manager.load(data_id) as data:
        assert data.type == "VideoData"
        assert data.id == data_id
        assert data.ext == "mp4"
        with data.open_video() as f:
            # raw_data = f.read()

            count = 0
            frame = iio.v3.imiter(f, plugin="pyav", format="rgb24", extension=".mp4")
            for x in frame:
                print(x.shape)
                count += 1
            print(count)
            decoder = VideoDecoder(f, extension=".mp4")
            count = 0
            for x in decoder:
                print(x["frame"].shape)
                count += 1

            assert count == 50

    # assert False


def test_audio_data():
    tmp_dir = tempfile.mkdtemp()
    data_manager = DataManager(data_dir=tmp_dir)

    data_path = None
    with data_manager.create_data("AudioData") as data:
        data_id = data.id
        data_path = os.path.join(tmp_dir, data.id[0:2], data.id[2:4], f"{data.id}.zip")
        data.ext = "wav"
        with data.open_audio(mode="w") as f_out:
            with open(os.path.join(test_path, "test.wav"), "rb") as f_in:
                while True:
                    data = f_in.read(128 * 1024)
                    if len(data) == 0:
                        break
                    f_out.write(data)

    read_zipfile_dir(data_path)

    with data_manager.load(data_id) as data:
        assert data.type == "AudioData"
        assert data.id == data_id
        assert data.ext == "wav"
        with data.open_audio() as f:
            # raw_data = f.read()
            tmp = io.BytesIO(f.read())
            # tmp.name = "a.mp3"
            y, sr = soundfile.read(tmp)
            assert y.shape[0] == 24001


def test_audio_data():
    tmp_dir = tempfile.mkdtemp()
    data_manager = DataManager(data_dir=tmp_dir)

    data_path = None
    with data_manager.create_data("AudioData") as data:
        data_id = data.id
        data_path = os.path.join(tmp_dir, data.id[0:2], data.id[2:4], f"{data.id}.zip")
        data.ext = "wav"
        with data.open_audio(mode="w") as f_out:
            with open(os.path.join(test_path, "test.wav"), "rb") as f_in:
                while True:
                    data = f_in.read(128 * 1024)
                    if len(data) == 0:
                        break
                    f_out.write(data)

    read_zipfile_dir(data_path)

    with data_manager.load(data_id) as data:
        assert data.type == "AudioData"
        assert data.id == data_id
        assert data.ext == "wav"
        with data.open_audio() as f:
            # raw_data = f.read()
            tmp = io.BytesIO(f.read())
            # tmp.name = "a.mp3"
            y, sr = soundfile.read(tmp)
            assert y.shape[0] == 24001

    # assert False#
