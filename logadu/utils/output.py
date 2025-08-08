class ModelOutput:
    def __init__(self, logits, probabilities, loss=None, embeddings=None):
        self.logits = logits
        self.probabilities = probabilities
        self.loss = loss
        self.embeddings = embeddings