# -*- coding: utf-8 -*-
"""
@date: 25/04/2020

@name: shapley.py

@author: Nieves Montes Gómez

@description: Compute Shapley value of individual norms in an optimal normative system.
"""

import math
import copy
import pickle
import json
from itertools import product

import numpy as np

from fitness_functions import evaluate_equality, evaluate_justice, aggregate_equality_justice
from genetic_search import segments, params_fixed
from tax_model import Society

length = 10  # length of the paths for evaluation of fitness/alignment
paths = 100  # number of paths for evaluation of fitness

# baseline parameters
baseline_params = {'num_agents': params_fixed['num_agents'],
                   'num_evaders': params_fixed['num_evaders'],
                   'collecting_rates': [0. for _ in range(segments)],
                   'redistribution_rates': [1 / segments for _ in range(segments)],
                   'invest_rate': params_fixed['invest_rate'],
                   'catch': 0.,
                   'fine_rate': 0.}

# coalition of norms
coalition = ['collecting_rates', 'redistribution_rates', 'catch', 'fine_rate']


def shapley_value(model_cls, individual_norm, baseline_parameters, optimal_parameters, norm_coalition, alignment_function):
	"""
	Compute the Shapley value with respect to a value for a specified norm.
	Args:
		- model_cls: the model class under consideration.
		- individual_norm: string with the norm to compute Shapley value of.
		- baseline_parameters: dict of parameters of baseline model.
		- optimal_parameters: dict of parameters of the optimal model.
		- coalition: a list of strings with the name of all norms in the normative system.
		- alignment_function: the alignment function to be used.
	"""
	# generate all coalitions
	N = len(norm_coalition)
	variable_norms = copy.deepcopy(norm_coalition)
	variable_norms.remove(individual_norm)
	all_coalitions = []
	for comb in product(('baseline_parameters', 'optimal_parameters'), repeat=N-1):
		all_coalitions.append({})
		for norm, origin in zip(variable_norms, comb):
			all_coalitions[-1][norm] = origin

	# compute the  contribution of each coalition
	shapley = 0
	for coalition in all_coalitions:
		N_union_n_norms = copy.deepcopy(baseline_parameters)
		N_norms = copy.deepcopy(baseline_parameters)
		for norm, origin in coalition.items():
			N_union_n_norms[norm] = locals()[origin][norm]
			N_norms[norm] = locals()[origin][norm]
		N_union_n_norms[individual_norm] = optimal_parameters[individual_norm]
		N_norms[individual_norm] = baseline_parameters[individual_norm]
		model_N_union_n = model_cls(**N_union_n_norms)
		model_N = model_cls(**N_norms)
		algn_N_union_n = alignment_function(model_N_union_n)
		algn_N = alignment_function(model_N)
		arr = np.array(list(coalition.values()))
		N_prime = int(np.where(arr == 'optimal_parameters', True, False).sum())
		shapley += math.factorial(N_prime) * math.factorial(N-N_prime-1) / math.factorial(N) * (algn_N_union_n - algn_N)
	return shapley



if __name__ == '__main__':
	# baseline model
	baseline_model = Society(**baseline_params)
	baseline_algn_equality = evaluate_equality(baseline_model)
	baseline_algn_justice = evaluate_justice(baseline_model)
	baseline_algn_aggregated = aggregate_equality_justice(baseline_model)
	print("Baseline model alignment with respect to equality: {:.4f}".format(baseline_algn_equality))
	print("Baseline model alignment with respect to justice: {:.4f}".format(baseline_algn_justice))
	print("Baseline model alignment with respect to aggregated values: {:.4f}".format(baseline_algn_aggregated))

	# Shapley values for optimal model with respect to equality
	with open("optimal_models/solution_evaluate_equality.model", "rb") as file:
		sol_equality = pickle.load(file)

	optimal_params_equality = {
		'num_agents': sol_equality.num_agents,
		'num_evaders': sol_equality.num_evaders,
		'collecting_rates': sol_equality.collecting_rates,
		'redistribution_rates': sol_equality.redistribution_rates,
		'invest_rate': sol_equality.invest_rate,
		'catch': sol_equality.catch,
		'fine_rate': sol_equality.fine_rate}

	shapley_values_equality = {}
	for norm in coalition:
		shapley_values_equality[norm] = shapley_value(Society, norm, baseline_params, optimal_params_equality, coalition, evaluate_equality)

	with open('shapley_values/shapley_values_equality.json', 'w') as file:
		json.dump(shapley_values_equality, file)

	# Shapley values for optimal model with respect to justice
	with open("optimal_models/solution_evaluate_justice.model", "rb") as file:
		sol_justice = pickle.load(file)

	optimal_params_justice = {
		'num_agents': sol_justice.num_agents,
		'num_evaders': sol_justice.num_evaders,
		'collecting_rates': sol_justice.collecting_rates,
		'redistribution_rates': sol_justice.redistribution_rates,
		'invest_rate': sol_justice.invest_rate,
		'catch': sol_justice.catch,
		'fine_rate': sol_justice.fine_rate}

	shapley_values_justice = {}
	for norm in coalition:
		shapley_values_justice[norm] = shapley_value(Society, norm, baseline_params, optimal_params_justice, coalition, evaluate_justice)

	with open('shapley_values/shapley_values_justice.json', 'w') as file:
		json.dump(shapley_values_justice, file)

	# Shapley values for optimal model with respect to aggregated equality and justice
	with open("optimal_models/solution_aggregate_equality_justice.model", "rb") as file:
		sol_aggregate = pickle.load(file)

		optimal_params_aggregate = {
			'num_agents': sol_aggregate.num_agents,
			'num_evaders': sol_aggregate.num_evaders,
			'collecting_rates': sol_aggregate.collecting_rates,
			'redistribution_rates': sol_aggregate.redistribution_rates,
			'invest_rate': sol_aggregate.invest_rate,
			'catch': sol_aggregate.catch,
			'fine_rate': sol_aggregate.fine_rate}

		shapley_values_aggregate = {}
		for norm in coalition:
			shapley_values_aggregate[norm] = shapley_value(Society, norm, baseline_params, optimal_params_aggregate, coalition, aggregate_equality_justice)

		with open('shapley_values/shapley_values_aggregate.json', 'w') as file:
			json.dump(shapley_values_aggregate, file)
