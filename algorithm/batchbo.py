"""BO explorer."""
# modified based on https://github.com/pytorch/botorch/blob/main/botorch/acquisition/monte_carlo.py
from bisect import bisect_left
from typing import Optional, Tuple
import scipy.stats as stats
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
import random
import os
from utils.seq_utils import check_cdr_constraints, levenshtein_distance, sample_new_seqs,sequence_to_one_hot
from scipy.stats import norm

from . import register_algorithm
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
from botorch.acquisition import qKnowledgeGradient

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
NUM_FANTASIES = 128 if not SMOKE_TEST else 4



@register_algorithm("batchbo")
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
            method (equal to EI or UCB or KG): The improvement method used in BO,
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
        self.method = method
        self.recomb_rate = 0
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

    def train_models(self):  # change reward
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
        self.model.train_model(state_seqs, rewards)

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
    
    @staticmethod
    def EI(mu, std, best):
        """Compute expected improvement."""
        # print('vals',vals)
        # return np.mean([max(val - self.best_fitness, 0) for val in vals])
        return norm.cdf((mu - best) / (std+1E-9))

    @staticmethod
    def calculate_knowledge_gradient(mean, std, current_best, num_fantasies):
        # Sample fantasized functions
        f = np.random.normal(mean, std, size=(num_fantasies,1))
        f_best = np.max(f, axis=1)
        
        # Compute mean and std of maximum value from fantasized functions
        f_best_mean = np.mean(f_best)
        f_best_std = np.std(f_best, ddof=1)
        
        # Compute knowledge gradient
        kg = (f_best_mean - current_best) * norm.cdf((f_best_mean - current_best) / f_best_std) + \
            f_best_std * norm.pdf((f_best_mean - current_best) / f_best_std)
        
        return kg


    @staticmethod
    def UCB(vals,std_pre):
        """Upper confidence bound."""
        discount = 0.5
        return vals + discount * std_pre

    def pick_action(self, all_measured_seqs,score_max, all_seqs):
        """Pick action."""
        states_to_screen = []
        states_to_screen = []
        method_pred = []
        # local search for all satisfied seq candidate pool
        candidate_pool = list(set(all_seqs) - set(all_measured_seqs))
        # not enough do global search
        if len(candidate_pool) < (self.model_queries_per_batch // self.sequences_batch_size):
            states_to_screen_ = sample_new_seqs(
                all_seqs,
                all_measured_seqs,
                (self.model_queries_per_batch // self.sequences_batch_size) - len(candidate_pool),
                self.rng,
            )
            candidate_pool.extend(states_to_screen_)
            states_to_screen = candidate_pool

        # enough then we sample from satisfied pool
        else:
            states_to_screen = candidate_pool
        ensemble_preds = self.model.get_fitness(states_to_screen) 
        uncertainty_pred=self.model.get_uncertainty(states_to_screen)
        max_pred = max(ensemble_preds)
        mean_pred = np.mean(self.model.get_fitness(candidate_pool))
        std_pre = np.std(self.model.get_fitness(candidate_pool))
        best_fitness_obs = score_max ## this is the best fitness observed from last round
        best_fitness = best_fitness_obs
        if self.method == "EI":
            method_pred = self.EI(ensemble_preds,uncertainty_pred,best_fitness)##https://machinelearningmastery.com/what-is-bayesian-optimization/ 
        if self.method == "KG":
            for i in range(len(ensemble_preds)):
                kg = self.calculate_knowledge_gradient(ensemble_preds[i], uncertainty_pred[i], best_fitness, num_fantasies=128)
                method_pred.append(kg)
        if self.method == "UCB":
            method_pred = self.UCB(ensemble_preds, uncertainty_pred)
        action_ind = np.argpartition(method_pred, -self.sequences_batch_size)[-self.sequences_batch_size:]
        action_ind = action_ind.tolist()
        new_state_string = np.asarray(states_to_screen)[action_ind]
        # self.state = string_to_one_hot(new_state_string, self.alphabet)
        # new_state = self.state
        reward = np.mean(ensemble_preds[action_ind])
        # if new_state_string not in all_measured_seqs:
        #     self.best_fitness = max(self.best_fitness, reward)
        #     self.memory.store(state.ravel(), action, reward, new_state.ravel())
        self.num_actions += 1
        return  new_state_string, reward

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
        self, measured_sequences: pd.DataFrame, score_max, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()
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
            max_score_id = np.argmax(_last_batch_true_scores)
        # generate next batch by picking actions
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        new_state_string, _ = self.pick_action(
            all_measured_seqs, score_max, kwargs["all_seqs"]
        )  
        samples=new_state_string
        # get predicted fitnesses of samples
        samples = list(samples)
        preds = np.mean(self.model.get_fitness(samples))
        # train ensemble model before returning samples

        return samples, preds
