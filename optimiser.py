#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:25:07 2020

@author: Nieves Montes

@description: Implementation of a Genetic Optimizer to find the family of
parameters that results in the most well-aligned model with respect to some
values.
"""

import copy
import portion
import pickle
import numpy as np

from tax_model import Society
from alignment import compute_alignment


class GeneticOptimizer:
  """
  Class that implements a Genetic Algorithm optimizer.
  Args:
    - model_cls: class of the MAS model to be optimized.
    - params_optimize: a dictionary of the model parameters to be optimized,
    with format {key: allowed interval}.
    - params_fixed: a dictionary of the model parameters that stay fixed
    throughout the search. It has the format: {key: fixed value}.
    - value: target value. Options are 'equality', 'fairness' and
    'aggregation'.
    - pop_size: population size for the genetic search. Default 50.
    - p: p parameter for intermediate recombination. Default 0.25.
    - keep_best: from one iteration of the genetic search to the next, keep
    this number of the best models from the previous generation. Default 5.
    - max_total_iter: maximum number of total iteration of the genetic search
    to perform before halting. Default 500.
    - max_partial_iter: maximum number of iterations to perform in the genetic
    search without an update on the candidate solution before halting.
    Default 20.
  """

  def __init__(self, model_cls, params_optimize, params_fixed, value,
              pop_size=100, p=0.25, keep_best=5, max_total_iter=500,
              max_partial_iter=50, fitness_threshold=0.90):
    self.model_cls = model_cls
    self.params_optimize = params_optimize
    self.params_fixed = params_fixed
    self.value = value
    self.pop_size = pop_size
    self.p = p
    self.keep_best = keep_best
    self.max_total_iter = max_total_iter
    self.max_partial_iter = max_partial_iter
    self.fitness_threshold = fitness_threshold
    self.population = []

  def make_random_candidate(self):
    """
    Build a random candidate and evaluate its fitness.
    """
    init_params = self.params_fixed
    for key, interval in self.params_optimize.items():
      # for optimizable parameters that are single values
      if not isinstance(interval, list):
        init_params[key] = np.random.uniform(low=interval.lower,
                                             high=interval.upper)
      # for optimizable parameters that are actually lists
      else:
        init_params[key] = [np.random.uniform(low=i.lower, high=i.upper)
                            for i in interval]
      # for redistribution_rates: normalize to respect constraint
      if key == 'redistribution_rates':
        init_params[key] = [i / sum(init_params[key])
                            for i in init_params[key]]
    model = self.model_cls(**init_params)
    model.fitness = compute_alignment(model, self.value)
    return model


  def build_initial_population(self):
    """
    Build an initial random population.
    """
    self.population = [self.make_random_candidate()
                       for _ in range(self.pop_size)]


  def tournament_selection(self):
    """
    Return a fit Candidate by performing 1 vs 1 tournament selection.
    """
    i = np.random.randint(0, self.pop_size)
    j = np.random.randint(0, self.pop_size)
    # participants must be different candidates
    while j == i:
      j = np.random.randint(0, self.pop_size)
    if self.population[i].fitness > self.population[j].fitness:
      return self.population[i]
    return self.population[j]


  def find_fittest(self):
    """
    Find the fittest individual in the population. If there is a tie, return
    the first best model in the population.
    """
    get_fitness = lambda m: m.fitness
    fitness_population = [get_fitness(model) for model in self.population]
    max_fitness_index = np.argmax(fitness_population)
    return self.population[max_fitness_index]


  def intermediate_recombination(self, parent1, parent2):
    """
    Perform intermediate recombination of the optimizable parameters.
    Args:
      - parent1, parent2: two models selected for breeding.
    Return:
      - A tuple if the two offspring models.
    """
    child1_params = copy.deepcopy(self.params_fixed)
    child2_params = copy.deepcopy(self.params_fixed)

    # cross-over
    for key, interval in self.params_optimize.items():

      # for optimizable parameters that are single values
      if not isinstance(interval, list):
        alpha = np.random.uniform(-self.p, 1 + self.p)
        beta = np.random.uniform(-self.p, 1 + self.p)
        child1_params[key] = alpha * getattr(parent1, key) + \
          (1 - alpha) * getattr(parent2, key)
        child2_params[key] = beta * getattr(parent1, key) + \
          (1 - beta) * getattr(parent2, key)
        while (child1_params[key] not in interval) or \
          (child2_params[key] not in interval):
          alpha = np.random.uniform(-self.p, 1 + self.p)
          beta = np.random.uniform(-self.p, 1 + self.p)
          child1_params[key] = alpha * getattr(parent1, key) + \
            (1 - alpha) * getattr(parent2, key)
          child2_params[key] = beta * getattr(parent1, key) + \
            (1 - beta) * getattr(parent2, key)

      # for parameters that are lists
      else:
        child1_params[key] = []
        child2_params[key] = []
        for i in range(parent1.num_segments):
          alpha = np.random.uniform(-self.p, 1 + self.p)
          beta = np.random.uniform(-self.p, 1 + self.p)
          child1_params[key].append(alpha * getattr(parent1, key)[i] + \
                                    (1 - alpha) * getattr(parent2, key)[i])
          child2_params[key].append(beta * getattr(parent1, key)[i] + \
                                    (1 - beta) * getattr(parent2, key)[i])
          while (child1_params[key][-1] not in interval[i]) or \
            (child2_params[key][-1] not in interval[i]):
            alpha = np.random.uniform(-self.p, 1 + self.p)
            beta = np.random.uniform(-self.p, 1 + self.p)
            child1_params[key][-1] = alpha * getattr(parent1, key)[i] + \
              (1 - alpha) * getattr(parent2, key)[i]
            child2_params[key][-1] = beta * getattr(parent1, key)[i] + \
              (1 - beta) * getattr(parent2, key)[i]

      # normalize redistribution rates
      if key == 'redistribution_rates':
        child1_params[key] = [i / sum(child1_params[key])
                              for i in child1_params[key]]
        child2_params[key] = [i / sum(child2_params[key])
                              for i in child2_params[key]]

    child1 = self.model_cls(**child1_params)
    child2 = self.model_cls(**child2_params)
    child1.fitness = compute_alignment(child1, self.value)
    child2.fitness = compute_alignment(child2, self.value)
    return child1, child2


  def genetic_search(self):
    """
    Perform the genetic search until any of the stopping criteria is met.
    Return:
      - Solution model.
    """
    print("...Building initial population...")
    self.build_initial_population()
    # find fittest so far
    fittest_so_far = copy.deepcopy(self.find_fittest())
    print("Fittest model: {:.4f}\n".format(fittest_so_far.fitness))
    if fittest_so_far.fitness > self.fitness_threshold:
      return fittest_so_far
    
    # save fittest so far
    filename = "solution_" + self.value + ".model"
    with open(filename, "wb") as file:
      pickle.dump(fittest_so_far, file)

    # GENETIC SEARCH LOOP
    partial_iter = 0
    get_fitness = lambda m: m.fitness
    for i in range(self.max_total_iter):
      print("...Iteration {}...".format(i))
      next_gen = []
      for _ in range(self.pop_size // 2):
        parent1 = self.tournament_selection()
        parent2 = self.tournament_selection()
        child1, child2 = self.intermediate_recombination(parent1, parent2)
        next_gen.append(child1)
        next_gen.append(child2)

      # drop the worst child models from the next generation and replace them
      # with the best current generation models
      next_gen_fitness = [get_fitness(child) for child in next_gen]
      population_fitness = [get_fitness(parent) for parent in self.population]
      worst_children_indices = np.array(next_gen_fitness).argsort().\
        tolist()[:self.keep_best]
      best_parent_indices = np.array(population_fitness).argsort().tolist()
      best_parent_indices.reverse()
      best_parent_indices = best_parent_indices[:self.keep_best]
      for bad_child_index, good_parent_index in zip(worst_children_indices,
                                                    best_parent_indices):
        _ = next_gen.pop(bad_child_index)
        next_gen.append(self.population[good_parent_index])

      # replace population with new generation and get new fittest in
      # population
      self.population = next_gen
      fittest_in_population = self.find_fittest()
      if fittest_in_population.fitness > fittest_so_far.fitness:
        fittest_so_far = copy.deepcopy(fittest_in_population)
        with open(filename, "wb") as file:
            pickle.dump(fittest_so_far, file)
        partial_iter = 0

      # check termination conditions
      if partial_iter >= self.max_partial_iter:
        print("Maximum partial iterations exceeded.")
        print("Solution alignment: {:.4f}".format(fittest_so_far.fitness))
        return fittest_so_far

      if fittest_so_far.fitness >= self.fitness_threshold:
        print("Current solution fitness exceeding threshold.")
        print("Solution alignment: {:.4f}".format(fittest_so_far.fitness))
        return fittest_so_far

      print("Fittest model: " + "{:.4f}".format(fittest_so_far.fitness) + "\n")

      partial_iter += 1

    print("Maximum total iterations exceeded.")
    print("Solution alignment: {:.4f}".format(fittest_so_far.fitness))
    return fittest_so_far


segments = 5

params_optimize = {
  'collecting_rates': [portion.closed(0, 1) for _ in range(segments)],
  'redistribution_rates': [portion.closed(0, 1) for _ in range(segments)],
  'catch': portion.closed(0, 0.5),
  'fine_rate': portion.closed(0, 1)
}

params_fixed = {
  'num_agents': 200,
  'num_evaders': 10,
  'invest_rate': 0.05
}

if __name__ == '__main__':

  v = 'aggregation'
  
  print("*** Optimization with respect to value {} ***\n".format(v.upper()))
  optimizer = GeneticOptimizer(
    model_cls=Society,
    params_optimize=params_optimize,
    params_fixed=params_fixed,
    value=v
  )
  
  optimal_model = optimizer.genetic_search()
