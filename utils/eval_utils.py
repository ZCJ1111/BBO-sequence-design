import time
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from utils.seq_utils import levenshtein_distance


class Runner:
    """The interface of landscape/model/explorer is compatible with FLEXS benchmark.

    - Fitness Landscape EXploration Sandbox (FLEXS)
      https://github.com/samsinai/FLEXS
    """

    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round
        self.alg = args.alg
        # self.cutom_data = args.custom_data_name

    def run(self, landscape, starting_sequence, model, explorer, name, runs, out_dir):
        # names = np.load("/home/tianyu/code/biodrug/unify-length/names.npy")
        names = list(landscape.fitness_data.keys())
        output_dir = Path(out_dir).expanduser().resolve()
        output_dir.mkdir(exist_ok=True, parents=True)

        np.random.seed(runs)
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        _, _, _, _, _ = self.update_results(0, [starting_sequence], [starting_fitness], 0)
        rounds_ = []
        score_maxs = []
        mutation: list[int] = []
        mutation_counts = []
        rts = []
        searched_seq_ = []
        loss_ = [None]
        round_min_seq = starting_sequence
        for round in range(1, self.num_rounds + 1):
            round_start_time = time.time()

            if len(self.sequence_buffer) > 1:
                print(f"Training model in round {round}")
                loss = model.train(self.sequence_buffer, self.fitness_buffer)
                print("loss", loss)
                loss_.append(loss)
            # np.save('loss100custom.npy',loss_)
            # inference all sequence?
            
            print(f"Proposing sequences in round {round}")
            
            if self.alg == "pexcons":
                sequences, _ = explorer.propose_sequences(self.results, round_min_seq)
                

            if self.alg == "antbo" or self.alg== "botorch":
                sequences, _ = explorer.propose_sequences(self.results, landscape=landscape, all_seqs=names)

            else:
                sequences, _ = explorer.propose_sequences(self.results, all_seqs=names)
                   
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)
            # print('len true_score',len(true_scores))
           
            round_mutation = []
            for seq in sequences:
                round_mutation.append(levenshtein_distance(s1=starting_sequence, s2=seq))
                print('distance is', levenshtein_distance(s1=starting_sequence, s2=seq))
            mutation += round_mutation
            
            print('round_mutation is', round_mutation)
            print('mutation is ', mutation)
            round_running_time = time.time() - round_start_time
            round_min_seq = sequences[np.argmin(round_mutation)]
            roundss, score_max, rt, mutcounts, searched_seq = self.update_results(
                round, sequences, true_scores, np.average(mutation), round_running_time
            )
            mutation_counts.append(mutcounts)
            rounds_.append(roundss)
            score_maxs.append(score_max)
            rts.append(rt)
            searched_seq_.append(searched_seq)
            result = pd.DataFrame(
                {
                    "round": rounds_,
                    "scoremax": score_maxs,
                    "run_time": rts,
                    "mutcounts": mutation_counts,
                    "loss": loss_,
                    "searched_seq": searched_seq_,
                }
            )
            result.to_csv(output_dir / f"trainlog_{name}_{runs}.csv", index=False)

    def update_results(
        self,
        round: int,
        sequences: List[str],
        true_scores: List[float],
        mutcounts: float,
        running_time: float = 0.0,
    ):
        self.results = pd.concat(
            (
                self.results,
                pd.DataFrame(
                    {
                        "round": round,
                        "sequence": sequences,
                        "true_score": true_scores,
                        "mutcounts": mutcounts,
                    }
                ),
            ),
            axis=0,
            ignore_index=True,
        )
        print(
            "round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec) mutation couts:{:.3f} searched sequence number {}".format(
                round, self.results["true_score"].max(), running_time, mutcounts, len(self.results)
            )
        )
        return round, self.results["true_score"].max(), running_time, mutcounts, len(self.results)

    @property
    def sequence_buffer(self):
        return self.results["sequence"].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results["true_score"].to_numpy()
