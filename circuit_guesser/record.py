class Record:
    def __init__(self, strategy, layers, actual_weights, solutions, running_time, batch_size=0, embedding_time=0, timing={}, failure=False):
        self.strategy = strategy
        self.layers = layers
        self.actual_weights = actual_weights
        self.solutions = solutions
        self.running_time = running_time
        self.batch_size = batch_size
        self.embedding_time = embedding_time
        self.timing = timing
        self.failure = failure
