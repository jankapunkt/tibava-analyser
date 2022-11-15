import pytest

from analyser.client import AnalyserClient
from analyser.data import ShotsData, Shot, ImageData, ImagesData, DataManager
from analyser.data import generate_id, create_data_path


def test_check_data():

    client = AnalyserClient("localhost", 50051)
    data_id = client.upload_data(ShotsData(shots=[Shot(start=0, end=10)]))
    assert data_id is not None
    assert client.check_data(data_id)
