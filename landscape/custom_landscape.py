import pandas as pd

from . import register_landscape


@register_landscape("custom")
class CustomLandscape:
    """Sample sequences from given fitness distributions."""

    def __init__(self, args):
        if not args.fitness_data:
            raise ValueError("--fitness-data not passed to custom landscape")

        # Read fitness dataset
        df = pd.read_csv(args.fitness_data)
        if args.fitness_col not in df.columns:
            raise ValueError(
                f"--fitness-column {args.fitness_col} not found in {args.fitness_data}"
            )

        # Make {sequence: fitness} dict
        self.fitness_data = (
            df.filter([args.seq_col, args.fitness_col])
            .set_index(args.seq_col)
            .to_dict()[args.fitness_col]
        )

        # If seed sequence is not given, use a sequence with median fitness
        self.median_fitness = df[args.fitness_col].median(axis=0)
        if args.starting_sequence and args.starting_sequence in self.fitness_data:
            self.starting_sequence = args.starting_sequence
        else:
            self.starting_sequence = df.loc[
                df[args.fitness_col] == self.median_fitness, args.seq_col
            ].to_numpy()[0]

    def get_fitness(self, sequences):
        # Input:  - sequences:      [query_batch_size, sequence_length]
        # Output: - fitness_scores: [query_batch_size]
        fitness_scores = [self.fitness_data.get(seq, self.median_fitness) for seq in sequences]

        return fitness_scores
