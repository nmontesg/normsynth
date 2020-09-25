# -*- coding: utf-8 -*-
"""
@date: 25/04/2020

@name: alignment_functions.py

@author: Nieves Montes Gómez

@description: Fitness functions to evaluate the alignment of a model with respect to some values.
Use the BatchRunner from the mesa library with support for multiprocessing.
"""

import numpy as np
from mesa.batchrunner import BatchRunner

from tax_model import Society, compute_gini_wealth

length = 10  # length of the paths for evaluation of fitness/alignment
paths = 100  # number of paths for evaluation of fitness


def evaluate_equality(model):
	"""
	Evaluate a model by its alignment with respect to equality.
	"""
	params = {
		'num_agents': model.num_agents,
		'collecting_rates': model.collecting_rates,
		'redistribution_rates': model.redistribution_rates,
		'invest_rate': model.invest_rate,
		'num_evaders': model.num_evaders,
		'catch': model.catch,
		'fine_rate': model.fine_rate}
	batch_run = BatchRunner(
		model_cls=Society,
		fixed_parameters=params,
		iterations=paths,
		max_steps=length,
		model_reporters={"Gini_wealth": compute_gini_wealth},
		display_progress=False)
	batch_run.run_all()
	run_data = batch_run.get_model_vars_dataframe()
	algn = 1 - 2 * run_data["Gini_wealth"].mean()
	return algn


def evaluate_justice(model):
	"""
	Evaluate a model by its alignment with respect to justice.
	"""
	params = {
		'num_agents': model.num_agents,
		'collecting_rates': model.collecting_rates,
		'redistribution_rates': model.redistribution_rates,
		'invest_rate': model.invest_rate,
		'num_evaders': model.num_evaders,
		'catch': model.catch,
		'fine_rate': model.fine_rate}
	batch_run = BatchRunner(
		model_cls=Society,
		fixed_parameters=params,
		iterations=paths,
		max_steps=length,
		agent_reporters={"Position": "position", "Evader": "is_evader"},
		display_progress=False)
	batch_run.run_all()
	info = batch_run.get_agent_vars_dataframe()
	evaders_info = info[info['Evader']]
	algn = - 1 + 2 * evaders_info["Position"].mean() / (params["num_agents"] - 1)
	return algn


def aggregate_equality_justice(model):
	"""
	Evaluate a model by its alignment aggregated over equality and justice.
	"""
	params = {
		'num_agents': model.num_agents,
		'collecting_rates': model.collecting_rates,
		'redistribution_rates': model.redistribution_rates,
		'invest_rate': model.invest_rate,
		'num_evaders': model.num_evaders,
		'catch': model.catch,
		'fine_rate': model.fine_rate}
	batch_run = BatchRunner(
		model_cls=Society,
		fixed_parameters=params,
		iterations=paths,
		max_steps=length,
		model_reporters={"Gini_wealth": compute_gini_wealth},
		agent_reporters={"Position": "position", "Evader": "is_evader"},
		display_progress=False)
	batch_run.run_all()
	# get gini index info
	model_data = batch_run.get_model_vars_dataframe()
	f = (1 - 2 * model_data["Gini_wealth"]).values
	# get justice-related info
	agent_data = batch_run.get_agent_vars_dataframe()
	evaders_data = agent_data[agent_data["Evader"]]
	g = np.array([])
	for run in evaders_data["Run"].unique():
		evaders_info_run = evaders_data[evaders_data["Run"] == run]
		g = np.append(g, [- 1 + 2 * evaders_info_run["Position"].mean() / (model.num_agents - 1)])
	# get F function
	algn = 0
	for x, y in zip(f, g):
		if x < 0 and y < 0:
			algn -= x * y
		else:
			algn += x * y
	return algn / paths
