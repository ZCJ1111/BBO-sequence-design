"""BO explorer."""
import random
from bisect import bisect_left
from typing import Optional, Tuple
from utils.seq_utils import  sample_new_seqs,levenshtein_distance,check_cdr_constraints

import flexs
import numpy as np
import pandas as pd
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    generate_random_sequences,
    one_hot_to_string,
    string_to_one_hot,
)

from . import register_algorithm


@register_algorithm("antbo2")
class BO(flexs.Explorer):
    """Evolutionary Bayesian Optimization (Evo_BO) explorer.

    Algorithm works as follows:     for N experiment rounds         recombine samples from previous
    batch if it exists and measure them,             otherwise skip         Thompson sample
    starting sequence for new batch         while less than B samples in batch             Generate
    `model_queries_per_batch/sequences_batch_size` samples             If variance of ensemble
    models is above twice that of the starting                 sequence             Thompson sample
    another starting sequence
    """

    def __init__(
        self,
        args,
        model,
        alphabet: str,
        starting_sequence: str,
    ):
        """
        Args:
            method (equal to EI or UCB): The improvement method used in BO,
                default EI.
            recomb_rate: The recombination rate on the previous batch before
                BO proposes samples, default 0.

        """
        method = "EI"
        name = f"BO_method={method}"
        self.name = name
        self.starting_sequence = starting_sequence
        self.sequences_batch_size = args.num_queries_per_round
        self.rounds = args.num_rounds
        self.model_queries_per_batch = args.num_model_queries_per_round
        self.model = model
        self.alphabet = alphabet
        self.method = "UCB"
        self.recomb_rate = 0.2
        self.batch_size = args.batch_size
        self.best_fitness = 0
        self.num_actions = 0
        self.state = None
        self.seq_len = None
        self.memory = None
        self.initial_uncertainty = None
        self.rng = np.random.default_rng(args.seed)

    def initialize_data_structures(self):
        """Initialize."""
        self.state = string_to_one_hot(self.starting_sequence, self.alphabet)
        self.seq_len = len(self.starting_sequence)
        # use PER buffer, same as in DQN
        self.memory = PrioritizedReplayBuffer(
            len(self.alphabet) * self.seq_len, 100000, self.sequences_batch_size, 0.6
        )

    def train_models(self): ## change reward
        """Train the model."""
        if len(self.memory) >= self.sequences_batch_size:
            batch = self.memory.sample_batch()
        else:
            self.memory.batch_size = len(self.memory)
            batch = self.memory.sample_batch()
            self.memory.batch_size = self.sequences_batch_size
        states = batch["next_obs"]
        state_seqs = [
            one_hot_to_string(state.reshape((-1, len(self.alphabet))), self.alphabet)
            for state in states
        ]
        rewards = batch["rews"]
        self.model.train(state_seqs, rewards)

    def _recombine_population(self, gen):
        np.random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if np.random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def EI(self, vals):
        """Compute expected improvement."""
        # print('vals',vals)
        # return np.mean([max(val - self.best_fitness, 0) for val in vals])
        return np.mean([max(vals - self.best_fitness, 0)])

    @staticmethod
    def UCB(vals, mean_pre, std_pre):
        """Upper confidence bound."""
        discount = 0.01

        return np.mean(vals) + mean_pre - discount * np.std(std_pre)


    def pick_action(self, all_measured_seqs,x_central_local,landscape,all_seqs, threshold=15):
        """Pick action."""
        states_to_screen = []
        states_to_screen=[]
        ## local search for all satisfied seq candidate pool
        candidate_pool=[]
        candidate_pool_ = list(set(all_seqs) - set(all_measured_seqs))

        ## put it outside
        for i in range(len(candidate_pool_)):
            dist=levenshtein_distance(x_central_local,candidate_pool_[i])
            if dist<threshold:
                candidate_pool.append(candidate_pool_[i])
                
        ## not enough do global search
        if len(candidate_pool)<(self.sequences_batch_size):
            states_to_screen_=sample_new_seqs(
                        all_seqs, all_measured_seqs, 1000, self.rng
                    )
            candidate_pool.extend(states_to_screen_)
            states_to_screen=candidate_pool

        ## enough then we sample from satisfied pool
        else:
            states_to_screen=candidate_pool

        ensemble_preds = []
        eval_batch_size = self.batch_size
        print('eval batch size',eval_batch_size)
        for i in range(0, len(states_to_screen), eval_batch_size):
            candidate_batch = candidate_pool[i : i + eval_batch_size]
            batch_model_scores = self.model.get_fitness(candidate_batch)
            ensemble_preds.append(batch_model_scores)
        ensemble_preds = np.concatenate(ensemble_preds)

        mean_pred = np.mean(ensemble_preds)
        std_pre = np.std(ensemble_preds)
        method_pred = (
            [self.EI(vals) for vals in ensemble_preds]
            if self.method == "EI"
            else [self.UCB(vals, mean_pred, std_pre) for vals in ensemble_preds]
        )

        action_ind = np.argpartition(method_pred, self.sequences_batch_size)[-self.sequences_batch_size:]
        uncertainty = np.std(method_pred)
        new_state_string = states_to_screen[action_ind]

        return uncertainty, new_state_string,action_ind

    @staticmethod
    def Thompson_sample(measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum(
            [np.exp(1 * x[0]) for x in measured_batch]
        )  # make it small inorder to avoid inf, previously it was 10*x[0]
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]

    def propose_sequences(
        self, measured_sequences: pd.DataFrame, landscape, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()
            x_central_local=self.starting_sequence
        else:
            # set state to best measured sequence from prior batch
            last_round_num = measured_sequences["round"].max()
            last_batch = measured_sequences[measured_sequences["round"] == last_round_num]
            _last_batch_seqs = last_batch["sequence"].tolist()
            _last_batch_true_scores = last_batch["true_score"].tolist()
            last_batch_seqs = _last_batch_seqs
            if self.recomb_rate > 0 and len(last_batch) > 1:
                last_batch_seqs = self._recombine_population(last_batch_seqs)
            measured_batch = []
            for seq in last_batch_seqs:
                if seq in _last_batch_seqs:
                    measured_batch.append(
                        (_last_batch_true_scores[_last_batch_seqs.index(seq)], seq)
                    )
                else:
                    measured_batch.append((np.mean(self.model.get_fitness([seq])), seq))
            measured_batch = sorted(measured_batch)
            sampled_seq = self.Thompson_sample(measured_batch)
            self.state = string_to_one_hot(sampled_seq, self.alphabet)
            max_score_id=np.argmax(_last_batch_true_scores)
            x_central_local = last_batch_seqs[max_score_id]
        # generate next batch by picking actions
        self.initial_uncertainty = None
        samples = set()
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        
        uncertainty, new_state_strings,action_ind = self.pick_action(all_measured_seqs,x_central_local,landscape,kwargs["all_seqs"]) 
        samples.add(new_state_strings)
        if self.initial_uncertainty is None:
            self.initial_uncertainty = uncertainty
        if uncertainty > 2 * self.initial_uncertainty:
            # reset sequence to starting sequence if we're in territory that's too
            # uncharted
            sampled_seq = self.Thompson_sample(measured_batch)
            self.state = string_to_one_hot(sampled_seq, self.alphabet)
            self.initial_uncertainty = None
    
        all_measured_seqs.add(samples)

        if len(samples) < self.sequences_batch_size:
            random_sequences = generate_random_sequences(
                self.seq_len, self.sequences_batch_size - len(samples), self.alphabet
            )
            samples.update(random_sequences)
        # get predicted fitnesses of samples
        samples = list(samples)
        preds = np.mean(self.model.get_fitness(samples))
        rewards=[]
        for i in range(len(samples)):
            rewards.append(landscape.get_fitness(samples[i]))

        self.memory.store(new_state_strings.ravel(), action_ind.ravel(), np.mean(rewards), samples.ravel()) 
        # train ensemble model before returning samples
        self.train_models()

        samples = random.sample(
            samples, self.sequences_batch_size
        ) 
        return samples, preds
