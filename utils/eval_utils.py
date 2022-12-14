import time
import numpy as np
import pandas as pd
from utils.seq_utils import hamming_distance
from utils.seq_utils import levenshteinDistance, convert_str
import json
class Runner:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round

    def run(self, landscape, starting_sequence, model, explorer,name,runs,task):
        names=np.load('/home/tianyu/code/biodrug/unify-length/names.npy')

        np.random.seed(runs)
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        _, _, _,_,_= self.update_results(0, [starting_sequence], [starting_fitness],0)
        rounds_=[]
        score_maxs=[]
        mutation=[]
        mutation_counts=[]
        rts=[]
        searched_seq_=[]
        loss_=[]
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()
            
            loss= model.train(self.sequence_buffer, self.fitness_buffer)
            print('loss',loss)
            loss_.append(loss)
            # np.save('loss100custom.npy',loss_)
            ## inference all sequence?
            # print('result',self.results)
            sequences, model_scores = explorer.propose_sequences(self.results)
            # sequences=['CARVPRAYYYDSSGPNNDYW','CARVPRAYYYDSSGPNNDYW']
            # print('seq',sequences)
            # print('start seq',starting_sequence)
            # print('len seq',len(sequences),'seq eg',len(sequences[0]),'len start seq',len(starting_sequence))
            assert len(sequences) <= self.num_queries_per_round

            true_scores = landscape.get_fitness(sequences)
            # print('len true_score',len(true_scores))
            for i in range(len(sequences)): 
                # print('starting seq',convert_str(starting_sequence,names))
                # print('seq',convert_str(sequences[i],names))  
                mutation.append(hamming_distance(convert_str(starting_sequence,names),convert_str(sequences[i],names)))
                # edit_dist=levenshteinDistance(starting_sequence,sequences[i],names)
                # mutation.append(edit_dist)

            round_running_time = time.time()-round_start_time
            roundss, score_max,rt, mutcounts,searched_seq = self.update_results(round, sequences, true_scores, round_running_time,np.average(mutation))
            mutation_counts.append(mutcounts)
            rounds_.append(roundss)
            score_maxs.append(score_max)
            rts.append(rt)
            searched_seq_.append(searched_seq)
            result=pd.DataFrame({
                "round":rounds_,
                "scoremax":score_maxs,
                "run_time":rts,
                'mutcounts':mutation_counts,
                "searched_seq":searched_seq_,

            })
            result.to_csv(f"expresult_{task}/trainlog_{name}_{runs}.csv",index=False)
            
            
    def update_results(self, round, sequences, true_scores, mutcounts, running_time=0.0):
        self.results = self.results.append(
            pd.DataFrame({
                "round": round,
                "sequence": sequences,
                "true_score": true_scores,
                'mutcounts':mutcounts,
            })
        )
        print('round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec) mutation couts:{:.3f} searched sequence number {}'.format(round, self.results['true_score'].max(), running_time,mutcounts, len(self.results)))
        return round, self.results['true_score'].max(), running_time,mutcounts,len(self.results)
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['true_score'].to_numpy()
