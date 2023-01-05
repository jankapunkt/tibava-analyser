class AnalyserPluginCallback:
    def update(self, **kwargs):
        pass


class AnalyserProgressCallback(AnalyserPluginCallback):
    def __init__(self, shared_memory) -> None:
        self.shared_memory = shared_memory

    def update(self, progress=0.0, **kwargs):
        self.shared_memory["progress"] = progress
