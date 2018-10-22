import torch
import itertools as itools
from multiprocessing import Pool
from functools import partial
from torch.distributions import Normal
import numpy as np
from train import train
from test import Tester
import time

from environment import create_env
from model import A3C_MLP, A3C_CONV
from shared_optim import SharedRMSprop, SharedAdam

ELITE_PROP = 0.04

class EA:
    def _init_model(self, model_type, env, stack_frames=0, load=False, load_file="./model.bin"):
        if model_type == 'MLP':
            model = A3C_MLP(
                env.observation_space.shape[0], env.action_space, stack_frames)
        if model_type == 'CONV':
            model = A3C_CONV(stack_frames, env.action_space)
        if load:
            saved_state = torch.load(load_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(saved_state)
        return model

    def _compute_ranks(self, x):
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def _compute_centered_ranks(self, fitnesses):
        x = np.array(fitnesses)
        y = self._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y.tolist()


    def __init__(self, args, model_type, env, pop_size, learning_rate=0.0001,
                    stack_frames=0, load=False, load_file="./model.bin"):
        if pop_size < 1:
            raise ValueError("Population size has to be one or greater, otherwise this doesn't make sense")
        self.pop_size = pop_size
        self.population = [] # a list of lists/generators of model parameters
        self.selected = [] # a buffer for the selected individuals
        self.to_select = int(self.pop_size * ELITE_PROP)
        self.fitnesses = []
        self.args = args
        
        self.sigma = 0.1
        self.sigma_decay = 0.999
        self.min_sigma = 0.01
        for n in range(pop_size + self.to_select):
            start_model = self._init_model(model_type, env, stack_frames, load, load_file)

            for p in start_model.parameters():
                p.data.copy_(torch.randn_like(p.data))
            if n < self.pop_size:
                self.population.append(start_model)
                self.fitnesses.append(0)
            else:
                self.selected.append(start_model)
    
    def ask(self):
        return self.population

    def tell(self, fitnesses):
        if len(fitnesses) != len(self.fitnesses):
            raise ValueError("Fitness array mismatch")

        self.fitnesses = list(fitnesses)
    
    def step(self, baseline, stable_fitness_f=None):
        """One step of the evolution"""
        # Sort the population by fitness and select the top
        sorted_fit_idxs = list(reversed(sorted(zip(self.fitnesses, itools.count()))))
        sorted_pop = [self.population[ix] for _, ix in sorted_fit_idxs]
        print("best in the population ----> ", sorted_fit_idxs[0][0])
        print("worst in the population ----> ", sorted_fit_idxs[-1][0])
        print("worst parent --------------->", sorted_fit_idxs[self.to_select][0])
        print("average fitness ------> ", sum(self.fitnesses)/len(self.fitnesses))
        selected_buffer = sorted_pop[:self.to_select]
        
        # Run few episodes on the elite part of the population to get a stable
        # metric on the fitness
        stable_fitnesses_ix = []
        if stable_fitness_f != None:
            # parallelize the recalculation of fitness
            with Pool() as pool:
                stable_selected_fitnesses_ = pool.map(stable_fitness_f, selected_buffer)
            stable_fitnesses_ix = [(ssf, ix) for ssf, (_, ix) in zip(stable_selected_fitnesses_, sorted_fit_idxs)]
            stable_fitnesses_ix = list(reversed(sorted(stable_fitnesses_ix)))
            selected_buffer = [self.population[ix] for _, ix in stable_fitnesses_ix]
            _, max_idx = stable_fitnesses_ix[0]
        else:
            _, max_idx = sorted_fit_idxs[0]
            
        
        # Sort the population again after the update
        # make a copy of the selected individuals
        for from_m, to_m in zip(selected_buffer, self.selected):
            to_m.load_state_dict(from_m.state_dict())

        # next generation
        for i in range(self.pop_size):
            if i == max_idx:
                # save the best model
                state_to_save = self.population[i].state_dict()
                torch.save(state_to_save, r'{0}_{1}_{2}_{3}.dat'.format(self.args.save_model_dir, self.args.env, self.args.scale_legs, time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
                continue

            dart = int(torch.rand(1) * self.to_select)
            parent = self.selected[dart]
            indiv = self.population[i]
            indiv.load_state_dict(parent.state_dict())

            for p in indiv.parameters():
                mask = torch.tensor(torch.rand_like(p.data) > 0.0, dtype=torch.float)
                mutation = torch.randn_like(p.data) * self.sigma
                mutation *= mask
                p.data += mutation
        
        if self.sigma > self.min_sigma:
            self.sigma *= self.sigma_decay
        elif self.sigma < self.min_sigma:
            self.sigma = self.min_sigma
        



    def result(self):
        """Report the fittest individual and its fitness"""
        max_fitness = max(self.fitnesses)
        max_idx = self.fitnesses.index(max_fitness)
        return (self.population[max_idx], max_fitness)

def stable_fitness_calculation(model, args, num_evals=3):
    fitness = 0.0
    for i in range(num_evals):
        fitness += train(1, args, model, max_iter=200)

    fitness /= float(num_evals)
    return fitness


def rollout(args, pop_size=500):
    torch.manual_seed(args.seed)
    env = create_env(args)
    solver = EA(args, args.model, env, pop_size, stack_frames=args.stack_frames, load=False)
    fitness_list = [0 for _ in range(pop_size)] 
    while True:
        solutions = solver.ask()
        baseline = sum(fitness_list) / float(len(fitness_list))
        with Pool() as pool:
            fitness_list = list(pool.map(partial(stable_fitness_calculation, args=args, num_evals=1), solutions))
        stabilizer = partial(stable_fitness_calculation, args=args, num_evals=10)
        solver.tell(fitness_list)
        solver.step(baseline, stable_fitness_f=None)
        result = solver.result()
        tester = Tester(args, result[0])
        tester.test(0, show="none", save_max=True)



    
