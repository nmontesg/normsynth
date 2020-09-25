# -*- coding: utf-8 -*-
"""
@date: 11/02/2020

@name: tax_model.py

@author: Nieves Montes Gómez

@description: A simple multi-agent model with two types of agents:
  - Individual: law-abiding citizens that always contribute to taxes if they can afford to.
  - Evaders: never contribute to taxes.
  Taxes are collected at every step, invested and redistributes. At every step, there is a probability that any evader
  is caught and fined.
Built with library mesa: https://mesa.readthedocs.io/en/master/index.html
"""

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from numpy import mean
from numpy.random import uniform, choice


def compute_gini_wealth(model):
	"""
	Compute the Gini index for the distribution of wealth in the model.
	"""
	agent_wealths = [agent.wealth for agent in model.schedule.agents]
	num = sum([sum([abs(x_i - x_j) for x_j in agent_wealths]) for x_i in agent_wealths])
	return num / (2 * model.num_agents ** 2 * mean(agent_wealths))


class Individual(Agent):
	"""
	An individual with some initial wealth.
	"""

	def __init__(self, model, unique_id):
		super().__init__(unique_id, model)
		self.wealth = uniform(0, 100)
		self.segment = 0
		self.position = 0
		self.is_evader = False

	def step(self):
		"""
		Individual contributes to common fund according to his/her tax rate.
		"""
		# law-abiding agents
		if not self.is_evader:
			tax = self.wealth * self.model.collecting_rates[self.segment]
			self.wealth -= tax
			self.model.common_fund += tax
		# evader agents:
		else:
			if uniform(0, 1) < self.model.catch:
				fine = self.wealth * self.model.collecting_rates[self.segment] * (1 + self.model.fine_rate)
				if fine >= self.wealth:
					self.model.common_fund += self.wealth
					self.wealth = 0
				else:
					self.wealth -= fine
					self.model.common_fund += fine


class Society(Model):
	"""
	A very simple of a society where taxes are collected and redistributed.
	"""

	def __init__(self, num_agents, num_evaders, collecting_rates, redistribution_rates, invest_rate, catch, fine_rate):
		assert len(collecting_rates) == len(
			redistribution_rates), "different number of collecting and redistributing segments."
		self.num_agents = num_agents  # number of agents
		self.common_fund = 0.  # collected taxes for each transition
		self.num_segments = len(collecting_rates)  # number of segments
		self.collecting_rates = collecting_rates  # collecting rates by group
		self.redistribution_rates = redistribution_rates  # redistribution rates by group
		self.invest_rate = invest_rate  # interest return to the investment of the common fund
		assert num_evaders <= self.num_agents, "more evaders than agents"
		self.num_evaders = num_evaders  # number of evader agents
		self.catch = catch  # probability of catching an evader
		self.fine_rate = fine_rate  # fine to be imposed if an evader in caught
		self.schedule = RandomActivation(self)
		self.running = True

		# create agents
		for i in range(self.num_agents):
			a = Individual(self, i)
			self.schedule.add(a)
		# assign some of the agents as evaders randomly
		for _ in range(self.num_evaders):
			random_agent = choice(self.schedule.agents)
			while random_agent.is_evader:
				random_agent = choice(self.schedule.agents)
			random_agent.is_evader = True

		# assign agents to their wealth group
		self.assign_agents_to_segments()

		# data collector on wealth of agents and Gini index of society
		self.data_collector = DataCollector(model_reporters={"Gini_wealth": compute_gini_wealth},
																				agent_reporters=dict(Wealth="wealth", Segment="segment", Position="position",
																														 Evader="is_evader"))
		# collect initial data
		self.data_collector.collect(self)

	def assign_agents_to_segments(self):
		"""
		Assign the agents in a model to their wealth segment and overall position.
		"""
		# assign agents to their wealth segment
		sorted_agents = sorted(self.schedule.agents, key=lambda a: a.wealth)
		# assign agents to their position in ranking
		for i in range(len(sorted_agents)):
			setattr(sorted_agents[i], 'position', self.num_agents - i - 1)
		# assign agents to their segment
		cut_index = int(self.num_agents / self.num_segments)
		for n in range(self.num_segments):
			for ag in sorted_agents[n * cut_index: (n + 1) * cut_index]:
				setattr(ag, 'segment', n)
			try:
				[setattr(ag, 'segment', n) for ag in sorted_agents[(n + 1) * cut_index:]]
			except:
				pass

	def step(self):
		"""
		Taxes and (if any) fines are collected into the common fund, and redistributed with interest.
		"""
		self.common_fund = 0.
		# collect taxes from all agents
		self.schedule.step()
		# redistribute common fund
		for ind in self.schedule.agents:
			ind.wealth += self.common_fund * (1 + self.invest_rate) * self.redistribution_rates[
				ind.segment] * self.num_segments / self.num_agents
		# recompute segments
		self.assign_agents_to_segments()
		# collect data
		self.data_collector.collect(self)
