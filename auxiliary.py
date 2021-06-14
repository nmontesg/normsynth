#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 07:29:17 2020

@author: Nieves Montes

@description: Auxiliary script to inspect the optimal models and produce the
plots.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 36})

from tax_model import Society
from alignment import compute_alignment, compute_compatibility_sample, length
from optimiser import params_fixed, segments


values = ['equality', 'fairness', 'aggregation']

def print_model_data(model_filename):
  with open(model_filename, "rb") as file:
    model = pickle.load(file)
  collect = [str(round(c*100)) + "% " for c in model.collecting_rates]
  redistribute = [str(round(r*100)) + "% " for r in model.redistribution_rates]  
  catch = str(round(model.catch*100)) + "%"
  fine = str(round(model.fine_rate*100)) + "%"
  print("Model parameters:")
  print("\tCollect:", *collect)
  print("\tRedistribute:", *redistribute)
  print("\tCatch:", catch)
  print("\tFine:", fine)
  print("\tOptimal alignment: {:.2f}".format(round(model.fitness, 2)))
  print("\n")


def plot_initial_distribution(model):
  agent_wealth = [ag.wealth for ag in model.agents]
  fig, ax = plt.subplots(figsize=(12, 10))
  for ag in model.agents:
    if ag.is_evader:
      color = 'red'
      size = 3500
      linewidth = 2.5
    else:
      color = "black"
      size = 750
      linewidth = 1.5
    ax.scatter(ag.wealth, 3, color=color, s=size, marker="|",
               linewidth=linewidth, zorder=10)
  binwidth = 12.5
  bins = np.arange(0, max(agent_wealth) + binwidth, binwidth)
  ax.hist(agent_wealth, bins=bins, edgecolor="black", color="cyan", zorder=0,
          linewidth=2.5, alpha=0.25)
  ax.set_xlabel('Wealth')
  ax.set_ylabel('Number of agents')
  ax.set_xticks(np.arange(0, max(agent_wealth) + binwidth*2, binwidth*2))
  ax.set_ylim(0, 200)
  ax.set_xlim(0, 100)
  fig.tight_layout()
  plt.show()
  
  
def plot_final(model, hist_color):
  model_params = {
    'num_agents': model.num_agents,
    'num_evaders': model.num_evaders,
    'collecting_rates': model.collecting_rates,
    'redistribution_rates': model.redistribution_rates,
    'invest_rate': model.invest_rate,
    'catch': model.catch,
    'fine_rate': model.fine_rate
  }
  reset_model = Society(**model_params)
  for _ in range(length):
    reset_model.step()
  agent_wealth = [ag.wealth for ag in reset_model.agents]
  fig, ax = plt.subplots(figsize=(12, 10))
  for ag in reset_model.agents:
    if ag.is_evader:
      color = 'red'
      size = 3500
      linewidth = 2.5
    else:
      color = "black"
      size = 1000
      linewidth = 1.5
    ax.scatter(ag.wealth, 15, color=color, s=size, marker="|",
               linewidth=linewidth, zorder=10)
  binwidth = 12.5
  bins = np.arange(0, max(agent_wealth) + binwidth, binwidth)
  ax.hist(agent_wealth, bins=bins, edgecolor="black", color=hist_color,
          zorder=0, linewidth=2.5, alpha=0.25)
  ax.set_xlabel('Wealth')
  ax.set_ylabel('Number of agents')
  ax.set_xticks(np.arange(0, max(agent_wealth) + binwidth*2, binwidth*2))
  ax.set_ylim(0, 200)
  ax.set_xlim(0, 130)
  fig.tight_layout()
  plt.show()

  
def plot_state(model, hist_color, n):
  agent_wealth = [ag.wealth for ag in model.agents]
  fig, ax = plt.subplots(figsize=(12, 10))
  for ag in model.agents:
    if ag.is_evader:
      color = 'red'
      size = 3500
      linewidth = 2.5
    else:
      color = "black"
      size = 1000
      linewidth = 1.5
    ax.scatter(ag.wealth, 3, color=color, s=size, marker="|",
                linewidth=linewidth, zorder=10)
  binwidth = 12.5
  bins = np.arange(0, max(agent_wealth) + binwidth, binwidth)
  ax.hist(agent_wealth, bins=bins, edgecolor="black", color=hist_color,
          zorder=0, linewidth=2.5, alpha=0.3)
  ax.set_xlabel('Wealth')
  ax.set_ylabel('Number of agents')
  ax.set_xticks(np.arange(0, max(agent_wealth) + binwidth*2, binwidth*2))
  ax.set_xlim(0, 100)
  ax.set_ylim(0, 200)
  ax.text(2.5, 175, "State "+str(n))
  fig.tight_layout()
  return fig


def make_giff(filename, value, hist_color, length=10):
  with open(filename, "rb") as file:
    model = pickle.load(file)
  model_params = {
    'num_agents': model.num_agents,
    'num_evaders': model.num_evaders,
    'collecting_rates': model.collecting_rates,
    'redistribution_rates': model.redistribution_rates,
    'invest_rate': model.invest_rate,
    'catch': model.catch,
    'fine_rate': model.fine_rate
  }
  folder = value
  reset_model = Society(**model_params)
  fig = plot_state(reset_model, hist_color, 0)
  fig.savefig(folder + "/" + value + "-0.png", dpi=400)
  for i in range(1, length+1):
    reset_model.step()
    fig = plot_state(reset_model, hist_color, i)
    fig.savefig(folder + "/" + value + "-" + str(i) + ".png", dpi=400)
    
    
def shapley_values_efficiency(value):
  baseline_params = {'num_agents': params_fixed['num_agents'],
                     'num_evaders': params_fixed['num_evaders'],
                     'collecting_rates': [0. for _ in range(segments)],
                     'redistribution_rates': [1/segments for _ in \
                                              range(segments)],
                     'invest_rate': params_fixed['invest_rate'],
                     'catch': 0.,
                     'fine_rate': 0.}
  with open("shapley_values/shapley_values_{}.json".format(value)) as file:
    shapley_values = json.load(file)
  print(shapley_values)
  filename = "optimal_models/solution_{}.model".format(value)
  with open(filename, "rb") as file:
    model = pickle.load(file)
  
  sum_shapley_values = sum(shapley_values.values())
  alignment_norms = compute_alignment(model, value)
  model_baseline = Society(**baseline_params)
  alignment_baseline = compute_alignment(model_baseline, value)
  
  print("Sum of the Shapley values: {:.3f}".format(sum_shapley_values))
  print("Alignment of the norms: {:.3f}".format(alignment_norms))
  print("Alignment of the baseline: {:.3f}".format(alignment_baseline))
  print("Predicted: {:.3f} = {:.3f} - {:.3f}"\
        .format(alignment_norms-alignment_baseline, alignment_norms,
                alignment_baseline))
  print("Found: {:.3f} = {:.3f} - {:.3f}\n"\
        .format(sum_shapley_values, alignment_norms, alignment_baseline))



if __name__ == '__main__':
  
  # inspect model parameters
  # for v in values:
  #   print("Optimal model with respect to " + v + ":")
  #   filename = "optimal_models/solution_" + v + ".model"
  #   print_model_data(filename)
    
  # # compute cross alignments
  # for v_i, v_j in itertools.product(values, repeat=2):
  #   filename = "optimal_models/solution_" + v_i + ".model"
  #   with open(filename, "rb") as file:
  #     model = pickle.load(file)
  #   algn = compute_alignment(model, v_j)
  #   print("v_i {:12} -- v_j {:12} -- Algn {:.4f}".format(v_i, v_j,
  #                                                         round(algn, 4)))
  
  # # plots
  # filename = "optimal_models/solution_equality.model"
  # with open(filename, "rb") as file:
  #   model = pickle.load(file)
  # plot_initial_distribution(model)
  # plot_final(model, 'lawngreen')
  
  # filename = "optimal_models/solution_fairness.model"
  # with open(filename, "rb") as file:
  #   model = pickle.load(file)
  # plot_final(model, 'orangered')
  
  # filename = "optimal_models/solution_aggregation.model"
  # with open(filename, "rb") as file:
  #   model = pickle.load(file)
  # plot_final(model, 'blueviolet')

  # # build animations
  # make_giff(filename="optimal_models/solution_equality.model",
  #           value="equality",
  #           hist_color="lawngreen")
  
  # make_giff(filename="optimal_models/solution_fairness.model",
  #           value="fairness",
  #           hist_color="orangered")
  
  # make_giff(filename="optimal_models/solution_aggregation.model",
  #           value="aggregation",
  #           hist_color="blueviolet")
  
  # check efficiency of the Shapley values with respect to the relative
  # for v in values:
  #   shapley_values_efficiency(v)
  
  
  # maximum compatibility normative system
  print_model_data('optimal_models/solution_max_compatibility.model')
  
  with open('optimal_models/solution_max_compatibility.model', "rb") as file:
    model = pickle.load(file)
      
  # d = compute_compatibility_sample(model)  
  # print("Compatibility: {:.2f}".format(d))
  
  # for v in values:
  #   algn = compute_alignment(model, v)
  #   print("Alignment w.r.t. {}: {:.2f}".format(v, algn))
  
  # plots
  filename = "optimal_models/solution_max_compatibility.model"
  with open(filename, "rb") as file:
    model = pickle.load(file)
  plot_final(model, 'mediumspringgreen')
  
  # make_giff(filename="optimal_models/solution_max_compatibility.model",
  #           value="max-compatibility",
  #           hist_color="mediumspringgreen")