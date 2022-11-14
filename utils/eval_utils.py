import time
import numpy as np
import pandas as pd
from utils.seq_utils import hamming_distance
class Runner:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round

    def run(self, landscape, starting_sequence, model, explorer,name,runs):
        np.random.seed(runs)
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        _, _, _,_= self.update_results(0, [starting_sequence], [starting_fitness],0)
        rounds_=[]
        score_maxs=[]
        mutation=[]
        rts=[]
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()
            
            model.train(self.sequence_buffer, self.fitness_buffer)
            sequences, model_scores = explorer.propose_sequences(self.results)
            
            # print('len seq',len(sequences))
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)
            for i in range(len(sequences)):   
                mutation.append(hamming_distance(starting_sequence,sequences[i]))
            round_running_time = time.time()-round_start_time
            roundss, score_max,rt, mutcounts = self.update_results(round, sequences, true_scores, round_running_time,np.average(mutation))
            rounds_.append(roundss)
            score_maxs.append(score_max)
            rts.append(rt)
            result=pd.DataFrame({
                "round":rounds_,
                "scoremax":score_maxs,
                "run_time":rts,
                'mutcounts':mutcounts,
            })
            result.to_csv(f"expresult_avGFP/trainlog_{name}_{runs}.csv",index=False)
            
            
    def update_results(self, round, sequences, true_scores, mutcounts, running_time=0.0):
        self.results = self.results.append(
            pd.DataFrame({
                "round": round,
                "sequence": sequences,
                "true_score": true_scores,
                'mutcounts':mutcounts,
            })
        )
        print('round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec) mutation couts:{:.3f}'.format(round, self.results['true_score'].max(), running_time,mutcounts))
        return round, self.results['true_score'].max(), running_time,mutcounts
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['true_score'].to_numpy()
