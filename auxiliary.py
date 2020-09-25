# -*- coding: utf-8 -*-
"""
@date: 25/04/2020

@name: auxiliary.py

@author: Nieves Montes Gómez

@description: Auxiliary code to generate results and plots from the optimal models returned by the genetic search.
"""

import pickle

import matplotlib.pyplot as plt
from tax_model import Society
from fitness_functions import evaluate_equality, evaluate_justice, aggregate_equality_justice

plt.rcParams.update({'figure.autolayout': True, 'font.size': 30})

length = 10


if __name__ == '__main__':

	# OPTIMAL MODEL WITH RESPECT TO EQUALITY
	with open("optimal_models/solution_evaluate_equality.model", "rb") as file:
		sol_equality = pickle.load(file)

	# cross alignment
	optimal_equality_justice = evaluate_justice(sol_equality)
	optimal_equality_aggregate = aggregate_equality_justice(sol_equality)
	print("Optimal model with respect to equality:")
	print("Alignment with respect to justice: {:.4f}".format(optimal_equality_justice))
	print("Alignment with respect to aggregated values: {:.4f}".format(optimal_equality_aggregate))

	# plot evolution of Gini index
	plt.figure(figsize=(10, 6))
	for _ in range(10):
		for _ in range(length):
			sol_equality.step()
		df = sol_equality.data_collector.get_model_vars_dataframe()
		df['Gini_wealth'].plot(marker='o', markersize=7, linewidth=2)
		sol_equality = Society(
			sol_equality.num_agents, sol_equality.num_evaders, sol_equality.collecting_rates, sol_equality.redistribution_rates,
			sol_equality.invest_rate, sol_equality.catch, sol_equality.fine_rate)
	plt.xlabel("Step")
	plt.ylabel("Gini Index")
	plt.savefig("plots/Gini_index_solution_equality.eps", format='eps', bbox_inches='tight')

	# plot distribution of initial wealth
	for _ in range(length):
		sol_equality.step()
	df = sol_equality.data_collector.get_agent_vars_dataframe()
	initial = df.xs(0, level="Step")
	plt.figure(figsize=(10, 6))
	initial["Wealth"].hist(grid=False, edgecolor="black", color="powderblue")
	plt.xlabel("Wealth")
	plt.ylabel("Count")
	plt.savefig("plots/initial_wealth_solution_equality.eps", format='eps', bbox_inches='tight')

	# plot distribution of final wealth
	final = df.xs(10, level="Step")
	plt.figure(figsize=(10, 6))
	final["Wealth"].hist(grid=False, edgecolor="black", color="limegreen")
	plt.xlabel("Wealth")
	plt.ylabel("Count")
	plt.savefig("plots/final_wealth_solution_equality.eps", format='eps', bbox_inches='tight')

	# OPTIMAL MODEL WITH RESPECT TO JUSTICE
	with open("optimal_models/solution_evaluate_justice.model", "rb") as file:
		sol_justice = pickle.load(file)

	# cross alignment
	optimal_justice_equality = evaluate_equality(sol_justice)
	optimal_justice_aggregate = aggregate_equality_justice(sol_justice)
	print("Optimal model with respect to justice:")
	print("Alignment with respect to equality: {:.4f}".format(optimal_justice_equality))
	print("Alignment with respect to aggregated values: {:.4f}".format(optimal_justice_aggregate))

	# distribution for initial wealth with rug plot
	for _ in range(length):
		sol_justice.step()
	df = sol_justice.data_collector.get_agent_vars_dataframe()
	initial = df.xs(0, level="Step")
	plt.figure(figsize=(10, 6))
	for index, data in initial.iterrows():
		if data["Evader"]:
			color = "red"
			size = 2500
		else:
			color = "black"
			size = 400
		plt.scatter(data["Wealth"], 3, color=color, s=size, marker="|", zorder=10)
	initial["Wealth"].hist(grid=False, edgecolor="black", color="powderblue", zorder=0)
	plt.xlabel("Wealth")
	plt.ylabel("Count")
	plt.savefig("plots/initial_wealth_solution_justice.eps", format='eps', bbox_inches='tight')

	# distribution of final wealth with rug plot
	final = df.xs(10, level="Step")
	plt.figure(figsize=(10, 6))
	for index, data in final.iterrows():
		if data["Evader"]:
			color = "red"
			size = 2500
		else:
			color = "black"
			size = 400
		plt.scatter(data["Wealth"], 3, color=color, s=size, marker="|", zorder=10)
	final["Wealth"].hist(grid=False, edgecolor="black", color="limegreen", zorder=0)
	plt.xlabel("Wealth")
	plt.ylabel("Count")
	plt.savefig("plots/final_wealth_solution_justice.eps", format='eps', bbox_inches='tight')

	# OPTIMAL MODEL WITH RESPECT TO THE AGGREGATED VALUES
	with open("optimal_models/solution_aggregate_equality_justice.model", "rb") as file:
		sol_aggregate = pickle.load(file)

	# cross alignment
	optimal_aggregate_equality = evaluate_equality(sol_aggregate)
	optimal_aggregate_justice = evaluate_justice(sol_aggregate)
	print("Optimal model with respect to aggregated values:")
	print("Alignment with respect to equality: {:.4f}".format(optimal_aggregate_equality))
	print("Alignment with respect to justice: {:.4f}".format(optimal_aggregate_justice))

	# plot evolution of Gini index
	plt.figure(figsize=(10, 6))
	for _ in range(10):
		for _ in range(length):
			sol_aggregate.step()
		df = sol_aggregate.data_collector.get_model_vars_dataframe()
		df['Gini_wealth'].plot(marker='o', markersize=10, linewidth=3)
		sol_aggregate = Society(
			sol_aggregate.num_agents, sol_aggregate.num_evaders, sol_aggregate.collecting_rates, sol_aggregate.redistribution_rates,
			sol_aggregate.invest_rate, sol_aggregate.catch, sol_aggregate.fine_rate)
	plt.xlabel("Step")
	plt.ylabel("Gini Index")
	plt.savefig("plots/Gini_index_solution_aggregated.eps", format='eps', bbox_inches='tight')

	# distribution for initial wealth with rug plot
	for _ in range(length):
		sol_aggregate.step()
	df = sol_aggregate.data_collector.get_agent_vars_dataframe()
	initial = df.xs(0, level="Step")
	plt.figure(figsize=(10, 6))
	for index, data in initial.iterrows():
		if data["Evader"]:
			color = "red"
			size = 2500
		else:
			color = "black"
			size = 400
		plt.scatter(data["Wealth"], 3, color=color, s=size, marker="|", zorder=10)
	initial["Wealth"].hist(grid=False, edgecolor="black", color="powderblue", zorder=0)
	plt.xlabel("Wealth")
	plt.ylabel("Count")
	plt.savefig("plots/initial_wealth_solution_aggregated.eps", format='eps', bbox_inches='tight')

	# distribution of final wealth with rug plot
	final = df.xs(10, level="Step")
	plt.figure(figsize=(10, 6))
	for index, data in final.iterrows():
		if data["Evader"]:
			color = "red"
			size = 2500
		else:
			color = "black"
			size = 400
		plt.scatter(data["Wealth"], 3, color=color, s=size, marker="|", zorder=10)
	final["Wealth"].hist(grid=False, edgecolor="black", color="limegreen", zorder=0)
	plt.xlabel("Wealth")
	plt.ylabel("Count")
	plt.savefig("plots/final_wealth_solution_aggregated.eps", format='eps', bbox_inches='tight')
