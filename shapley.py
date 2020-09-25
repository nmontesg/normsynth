#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:45:32 2020

@author: nmontes

@description: Compute Shapley value of individual norms in an optimal normative
system, and the cross-alignment.
"""

import math
import copy
import pickle
import json
from itertools import product

import numpy as np

from alignment import compute_alignment
from optimiser import params_fixed, segments
from tax_model import Society


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


def get_society_params(model):
  """
  Get the initialization parameters for an optimal model.
  """
  params = {
    'num_agents': model.num_agents,
    'num_evaders': model.num_evaders,
    'collecting_rates': model.collecting_rates,
    'redistribution_rates': model.redistribution_rates,
    'invest_rate': model.invest_rate,
    'catch': model.catch,
    'fine_rate': model.fine_rate
  }
  return params


def shapley_value(model_cls, individual_norm, baseline_parameters, optimal_parameters,
                  norm_coalition, value):
  """
  Compute the Shapley value with respect to a value for a specified norm.
  Args:
    - model_cls: the model class under consideration.
    - individual_norm: string with the norm to compute Shapley value of.
    - baseline_parameters: dict of parameters of baseline model.
    - optimal_parameters: dict of parameters of the optimal model.
    - coalition: a list of strings with the name of all norms in the normative system.
    - value: the value for which the model has been optimised.
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
    algn_N_union_n = compute_alignment(model_N_union_n, value)
    algn_N = compute_alignment(model_N, value)
    arr = np.array(list(coalition.values()))
    N_prime = int(np.where(arr == 'optimal_parameters', True, False).sum())
    shapley += math.factorial(N_prime) * math.factorial(N-N_prime-1) / math.factorial(N) * (algn_N_union_n - algn_N)
  return shapley



if __name__ == '__main__':
  # baseline model
  baseline_model = Society(**baseline_params)
  baseline_algn_equality = compute_alignment(baseline_model, 'equality')
  baseline_algn_fairness = compute_alignment(baseline_model, 'fairness')
  baseline_algn_aggregation = compute_alignment(baseline_model, 'aggregation')
  print("Baseline alignment with respect to equality: {:.4f}".format(baseline_algn_equality))
  print("Baseline alignment with respect to justice: {:.4f}".format(baseline_algn_fairness))
  print("Baseline alignment with respect to aggregated values: {:.4f}".format(baseline_algn_aggregation))
  
  values = ['equality', 'fairness', 'aggregation']

  # compute and save shapley values
  for v in values:
    filename = "optimal_models/solution_" + v + ".model"
    with open(filename, "rb") as file:
      model = pickle.load(file)
    optimal_params = get_society_params(model)
    shapley_values = {}
    
    for norm in coalition:
      shapley_values[norm] = shapley_value(Society, norm, baseline_params,
                                           optimal_params, coalition, v)

    with open('shapley_values_{}.json'.format(v), 'w') as file:
      json.dump(shapley_values, file)
