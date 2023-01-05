from ..manager import DataManager
from ..data import ContainerDate
from analyser import analyser_pb2


@DataManager.export("ScalarData", analyser_pb2.SCALAR_DATA)
class ScalarData(ContainerDate):
    def to_dict(self) -> dict:
        meta = super().to_dict()
        return {**meta, "ref_id": self.ref_id, "y": self.y, "time": self.time, "delta_time": self.delta_time}
