class MockSampler:
    def __init__(self, return_vals):
        """ return_vals should be a list """
        self.return_vals = return_vals

    def sample(self, *args, **kwargs):
        return self.return_vals.pop(0)
