import math
import copy
import pickle
import json
from itertools import product, combinations

import numpy as np

from alignment import compute_alignment, length, paths
from optimiser import params_fixed, segments
from tax_model import Society


# baseline parameters
baseline_params = {'num_agents': params_fixed['num_agents'],
                   'num_evaders': params_fixed['num_evaders'],
                   'collecting_rates': [0. for _ in range(segments)],
                   'redistribution_rates': [1/segments for _ in \
                                            range(segments)],
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


def shapley_value(model_cls, individual_norm, baseline_parameters,
                  optimal_parameters, norm_coalition, value):
  """
  Compute the Shapley value with respect to a value for a specified norm.
  Args:
    - model_cls: the model class under consideration.
    - individual_norm: string with the norm to compute Shapley value of.
    - baseline_parameters: dict of parameters of baseline model.
    - optimal_parameters: dict of parameters of the optimal model.
    - coalition: a list of strings with the name of all norms in the normative
    system.
    - value: the value for which the model has been optimised.
  """
  # generate all coalitions
  N = len(norm_coalition)
  variable_norms = copy.deepcopy(norm_coalition)
  variable_norms.remove(individual_norm)
  all_coalitions = []
  for comb in product(('baseline_parameters', 'optimal_parameters'),
                      repeat=N-1):
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
    shapley += math.factorial(N_prime) * math.factorial(N-N_prime-1) / \
      math.factorial(N) * (algn_N_union_n - algn_N)
  return shapley



def check_monotone(value):
  """
  Check that the optimal normative system with respect to a value is monotone
  for the alignment with respect to it.
  """
  filename = "optimal_models/solution_" + value + ".model"
  with open(filename, "rb") as file:
    model = pickle.load(file)
  model_params = get_society_params(model)
  all_results = []
  for i in range(len(coalition)):
    all_N1 = combinations(coalition, i)
    for N1 in all_N1:
      model_N1_params = baseline_params
      for p in N1:
        model_N1_params[p] = model_params[p]
      model_N1 = Society(**model_N1_params)
      algn_N1 = compute_alignment(model_N1, value)

      # only add one extra individual norm to N1
      to_be_added = list(set(coalition)-set(N1))
      model_N2_params = model_N1_params
      for n in to_be_added:
        model_N2_params[n] = model_params[n]
        model_N2 = Society(**model_N2_params)
        algn_N2 = compute_alignment(model_N2, value)
        all_results.append({'N1': N1, 'n': n, 'algn_N1': algn_N1,
                            'algn_N2': algn_N2})
  return all_results


if __name__ == '__main__':
  
  # # baseline model: check that it leaves the society unchanged
  # baseline_evolution = []
  # for _ in range(paths):
  #   baseline_model = Society(**baseline_params)
  #   initial_global_state = [(a.wealth, a.position) for a in
  #                           baseline_model.agents]
  #   for _ in range(length):
  #     baseline_model.step()
  #   final_global_state = [(a.wealth, a.position) for a in
  #                         baseline_model.agents]
  #   has_baseline_evolved = not initial_global_state == final_global_state
  #   baseline_evolution.append(has_baseline_evolved)
  # has_baseline_evolved = bool(sum(baseline_evolution))
  # print("Is the baseline normative system causing model evolution? {}\n"\
  #       .format(has_baseline_evolved))
  
  # compute and save shapley values
  values = ['equality', 'fairness', 'aggregation']
  # v = values[2]
  
  # print("Shapley values for value {}:\n".format(v.upper()))

  # filename = "optimal_models/solution_" + v + ".model"
  # with open(filename, "rb") as file:
  #   model = pickle.load(file)
  # optimal_params = get_society_params(model)
  # shapley_values = {}

  # for norm in coalition:
  #   shapley_values[norm] = shapley_value(Society, norm, baseline_params,
  #                                         optimal_params, coalition, v)

  # with open('shapley_values_{}.json'.format(v), 'w') as file:
  #   json.dump(shapley_values, file)
    
  # print("Shapley values for the optimal model with respect to {}"\
  #       .format(v.upper()))
  # for norm in coalition:
  #   print("\t{}: {:.2f}".format(norm, shapley_values[norm]))
  # print("\n")
  
  # # check efficiency with respect to baseline normative system
  # sum_shapley_values = sum(shapley_values.values())
  # alignment_norms = compute_alignment(model, v)
  # model_baseline = Society(**baseline_params)
  # alignment_baseline = compute_alignment(model_baseline, v)
  
  # print("Sum of the Shapley values: {:.3f}".format(sum_shapley_values))
  # print("Alignment of the norms: {:.3f}".format(alignment_norms))
  # print("Alignment of the baseline: {:.3f}".format(alignment_baseline))
  # print("Predicted: {:.3f} = {:.3f} - {:.3f}"\
  #       .format(alignment_norms-alignment_baseline, alignment_norms,
  #               alignment_baseline))
  # print("Found: {:.3f} = {:.3f} - {:.3f}"\
  #       .format(sum_shapley_values, alignment_norms, alignment_baseline))
  
  tol = 0.025
  for v in values:
    print("*** Checking monotonicity for value {} ***".format(v))
    monotone_results = check_monotone(v)
    
    for r in monotone_results:
      if r['algn_N1'] > r['algn_N2'] + tol:
        print("Non-monotonicity found:")
        print("N1 = {}".format(list(r['N1'])))
        print("algn_N1 = {:.2f}".format(r['algn_N1']))
        print("N2 = {}".format(list(r['N1'])+[r['n']]))
        print("algn_N2 = {:.2f}".format(r['algn_N2']))
        print()
    
    print()
    
    