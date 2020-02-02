import neal
import time
from dwave.system.composites import EmbeddingComposite
import circuits as c

class MockSampler:
    def __init__(self, return_vals):
        """ return_vals should be a list """
        self.return_vals = return_vals

    def sample(self, *args, **kwargs):
        return self.return_vals.pop(0)


class TimedEmbeddingComposite(EmbeddingComposite):

    def __init__(self, *args, **kwargs):
        self.timing = {'qpu_access_time': 0}
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        response = super().sample(*args, **kwargs)
        self.timing = c._merge_dicts_and_add(self.timing, response.info.get('timing', {}))

        return response

    def reset_timing(self):
        self.timing = {'qpu_access_time': 0}


class TimedSimulatedAnnealingSampler(neal.SimulatedAnnealingSampler):

    def __init__(self, *args, **kwargs):
        self.timing = {'qpu_access_time': 0}
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        start = time.time()
        response = super().sample(*args, **kwargs)
        end = time.time()
        self.timing['qpu_access_time'] += end-start

        return response

    def reset_timing(self):
        self.timing = {'qpu_access_time': 0}
