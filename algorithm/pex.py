import numpy as np
from tqdm import tqdm, trange

from utils.seq_utils import levenshtein_distance, sample_new_seqs

from . import register_algorithm


@register_algorithm("pex")
class ProximalExploration:
    """Proximal Exploration (PEX)"""

    def __init__(self, args, model, alphabet, starting_sequence):
        self.model = model
        self.alphabet = alphabet
        self.wt_sequence = starting_sequence
        self.num_queries_per_round = args.num_queries_per_round
        # self.num_model_queries_per_round = args.num_model_queries_per_round
        self.batch_size = args.batch_size
        # self.num_random_mutations = args.num_random_mutations
        # self.frontier_neighbor_size = args.frontier_neighbor_size
        self.dataset_range = args.datasetrange
        self.rng = np.random.default_rng(args.seed)

    def propose_sequences(self, measured_sequences, **kwargs):
        measured_sequence_set = set(measured_sequences["sequence"])
        # Generate random mutations in the first round.
        all_seqs = kwargs["all_seqs"]

        if len(measured_sequence_set) == 1:
            query_batch = sample_new_seqs(
                all_seqs, measured_sequence_set, self.num_queries_per_round, self.rng
            )
            return query_batch, [None] * len(query_batch)

        # Construct the candidate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        candidate_pool = list(set(all_seqs) - measured_sequence_set)

        # Arrange the candidate pool by the distance to the wild type.
        candidate_pool_dict = {}
        distances_to_wt = [
            levenshtein_distance(s1=self.wt_sequence, s2=candidate) for candidate in candidate_pool
        ]
        model_scores = []
        eval_batch_size = self.batch_size
        for i in trange(0, len(candidate_pool), eval_batch_size, desc="Model scores"):
            candidate_batch = candidate_pool[i : i + eval_batch_size]
            batch_model_scores = self.model.get_fitness(candidate_batch)
            model_scores.append(batch_model_scores)
        model_scores = np.concatenate(model_scores)

        for candidate, model_score, distance_to_wt in zip(
            candidate_pool, model_scores, distances_to_wt
        ):
            if distance_to_wt not in candidate_pool_dict.keys():
                candidate_pool_dict[distance_to_wt] = []
            candidate_pool_dict[distance_to_wt].append(
                dict(sequence=candidate, model_score=model_score)
            )
        for distance_to_wt in sorted(candidate_pool_dict.keys()):
            candidate_pool_dict[distance_to_wt].sort(reverse=True, key=lambda x: x["model_score"])
        # Construct the query batch by iteratively extracting the proximal frontier.
        query_batch = []
        model_scores = []
        print("Constructing query batch")
        while len(query_batch) < self.num_queries_per_round:
            # Compute the proximal frontier by Andrew's monotone chain convex hull algorithm. (line 5 of Algorithm 2 in the paper)
            # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
            stack = []
            for distance_to_wt in sorted(candidate_pool_dict.keys()):
                if len(candidate_pool_dict[distance_to_wt]) > 0:
                    data = candidate_pool_dict[distance_to_wt][0]
                    new_point = np.array([distance_to_wt, data["model_score"]])

                    def check_convex_hull(point_1, point_2, point_3):
                        return np.cross(point_2 - point_1, point_3 - point_1) <= 0

                    while len(stack) > 1 and not check_convex_hull(
                        stack[-2], stack[-1], new_point
                    ):
                        stack.pop(-1)
                    stack.append(new_point)
            while len(stack) >= 2 and stack[-1][1] < stack[-2][1]:
                stack.pop(-1)

            # Update query batch and candidate pool. (line 6 of Algorithm 2 in the paper)
            for distance_to_wt, model_score in stack:
                if len(query_batch) < self.num_queries_per_round:
                    new_seq = candidate_pool_dict[distance_to_wt][0]
                    query_batch.append(new_seq["sequence"])
                    model_scores.append(new_seq["model_score"])
                    candidate_pool_dict[distance_to_wt].pop(0)

        return query_batch, model_scores
