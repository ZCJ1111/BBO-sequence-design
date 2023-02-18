import numpy as np

ensemble_rules = {
    "mean": lambda x: np.mean(x, axis=0),
    "lcb": lambda x: np.mean(x, axis=0) - np.std(x, axis=0),
    "ucb": lambda x: np.mean(x, axis=0) + np.std(x, axis=0),
}


class Ensemble:
    def __init__(self, models, ensemble_rule):
        self.models = models
        self.ensemble_func = ensemble_rules[ensemble_rule]
        self.cost = 0
        
    def train(self, sequences, labels):
        total_loss = 0.0
        for model in self.models:
            loss = model.train(sequences, labels)
            total_loss += loss
        return total_loss / len(self.models)

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]
        self.cost += len(sequences)
        ensemble = self.ensemble_func([model.get_fitness(sequences) for model in self.models])

        return ensemble
