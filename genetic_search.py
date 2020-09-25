# -*- coding: utf-8 -*-
"""
@date: 25/04/2020

@name: genetic_search.py

@author: Nieves Montes Gómez

@description: Perform the genetic search with a genetic optimizer.
"""

import pickle

import portion
from fitness_functions import evaluate_equality, evaluate_justice, aggregate_equality_justice
from genetic_optimizer import GeneticOptimizer

from tax_model import Society

segments = 5

params_optimize = dict(
	collecting_rates=[portion.closed(0, 1) for _ in range(segments)],
	redistribution_rates=[portion.closed(0, 1) for _ in range(segments)],
	catch=portion.closed(0, 0.5),
	fine_rate=portion.closed(0, 1)
)

params_fixed = dict(
	num_agents=200,
	num_evaders=10,
	invest_rate=0.05
)

optimizer = GeneticOptimizer(
	model_cls=Society,
	params_optimize=params_optimize,
	params_fixed=params_fixed,
	fitness_function=aggregate_equality_justice  # set here the alignment function of choice
)

if __name__ == '__main__':
	optimal_model = optimizer.genetic_search()

	filename = "solution_" + optimizer.fitness_function.__name__ + ".model"
	with open(filename, "wb") as file:
		pickle.dump(optimal_model, file)
