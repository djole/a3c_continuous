import torch
from torch.distributions import Normal
import numpy as np
from train import train
from test import Tester

from environment import create_env
from model import A3C_MLP, A3C_CONV
from shared_optim import SharedRMSprop, SharedAdam

class NaturalES:
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


    def __init__(self, model_type, env, pop_size, learning_rate=0.0001,
                    stack_frames=0, load=False, load_file="./model.bin"):
        if pop_size < 1:
            raise ValueError("Population size has to be one or greater, otherwise this doesn't make sense")
        self.pop_size = pop_size
        self.population = [] # a list of lists/generators of model parameters
        self.fitnesses = []
        self.learning_rate = learning_rate
        for n in range(pop_size):
            start_model = self._init_model(model_type, env, stack_frames, load, load_file)

            for p in start_model.parameters():
                p.data.copy_(torch.randn_like(p.data)*0.1)
            self.population.append(start_model)
            self.fitnesses.append(0)
    
    def ask(self):
        return self.population

    def tell(self, fitnesses):
        if len(fitnesses) != len(self.fitnesses):
            raise ValueError("Fitness array mismatch")

        self.fitnesses = list(fitnesses)
    
    def step(self, baseline):
        """One step of the evolution"""

        # Calculate the mean of the population
        mean_list = []
        var_list = []
        mean_gradient_list = []
        var_gradient_list = []
        for p in self.population[0].parameters():
            mean_list.append(torch.zeros_like(p))
            mean_gradient_list.append(torch.zeros_like(p))
            var_list.append(torch.zeros_like(p))
            var_gradient_list.append(torch.zeros_like(p))
        
        for i, ind in enumerate(self.population):
            for j, par in enumerate(ind.parameters()):
                mean_list[j] *= i
                mean_list[j] += par.data
                mean_list[j] /= (i+1)
        
        for i, ind in enumerate(self.population):
            for j, par in enumerate(ind.parameters()):
                var_list[j] += (par.data - mean_list[j]) ** 2
        for v in var_list:
            v /= self.pop_size
        
        # Check the consistency of the data
        #for v in var_list:
        #    if (var_list[j] == 0.0).any():
        #        print("var ------------------>>", v)
        #        raise ValueError("The variance cannot have 0 elements")

        # Calculate the gradient approximations
        cent_ranks = self._compute_centered_ranks(self.fitnesses)
        print("centered ranks ======================", cent_ranks)
        for i, ind in enumerate(self.population):
            for j, par in enumerate(ind.parameters()):
                #mean_gradient_list[j] += cent_ranks[i] * ((par.data - mean_list[j]) / (var_list[j]**2))
                #var_gradient_list[j] += cent_ranks[i] * (((par.data - mean_list[j])**2 - var_list[j]**2) / (var_list[j]**3))
                mean_gradient_list[j] += (self.fitnesses[i] - baseline) * (par.data - mean_list[j])
                var_gradient_list[j] += (self.fitnesses[i] - baseline) * (((par.data - mean_list[j])**2 - var_list[j]**2) / (var_list[j]))
        
        
        for dm, ds in zip(mean_gradient_list, var_gradient_list):
            dm /= self.pop_size
            ds /= self.pop_size

        # Update the mean and the variance of the population
        for j, mean in enumerate(mean_list):
            ex = mean * 0.1
            mean += (self.learning_rate * mean_gradient_list[j]).min(-ex).max(ex)
        for j, sigma in enumerate(var_list):
            ex = sigma * 0.1
            sigma += (self.learning_rate * var_gradient_list[j]).min(-ex).max(ex)

        # Update the population by sampling new members from the updated distribution
        distributions = []
        for mean, sigma in zip(mean_list, var_list):
            #mean = torch.clamp(mean, -100, 100)
            sigma = torch.zeros_like(sigma) + 0.1
            if torch.isnan(mean).any() or torch.isnan(sigma).any():
                raise ValueError("The mean or the variance tensors contain NaN elements")
            distributions.append(Normal(mean, sigma))
            print("new mean", mean)
            print("new sigma", sigma)
        
        for ind in self.population:
            for dist, par in zip(distributions, ind.parameters()):
                par.data.copy_(dist.sample())
        
        print("======= end generation =======")


    def result(self):
        """Report the fittest individual and its fitness"""
        max_fitness = max(self.fitnesses)
        max_idx = self.fitnesses.index(max_fitness)
        return (self.population[max_idx], max_fitness)




def rollout(args, pop_size=50):
    torch.manual_seed(args.seed)
    env = create_env(args)
    solver = NaturalES(args.model, env, pop_size, stack_frames=args.stack_frames, load=False)
    fitness_list = [0 for _ in range(pop_size)] 
    while True:
        solutions = solver.ask()
        baseline = sum(fitness_list) / float(len(fitness_list))
        fitness_list = list(map(lambda m : train(1, args, m, max_iter=0), solutions))
        solver.tell(fitness_list)
        solver.step(baseline)
        result = solver.result()
        tester = Tester(args, result[0])
        tester.test(0, show="once")



    