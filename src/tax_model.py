import numpy as np

def compute_gini_wealth(model):
  """
  Compute the Gini index for the distribution of wealth in the model.
  """
  agent_wealths = [agent.wealth for agent in model.agents]
  numerator = sum([sum([abs(x_i - x_j) for x_j in agent_wealths])
                  for x_i in agent_wealths])
  gini_index = numerator / (2 * model.num_agents ** 2 * np.mean(agent_wealths))
  return gini_index


class Individual:
  """
  An individual with some initial wealth.
  """

  def __init__(self, model):
    self.model = model
    self.wealth = np.random.uniform(0, 100)
    self.segment = 0
    self.position = 0
    self.is_evader = False

  def step(self):
    """
    Individual contributes to common fund according to his/her tax rate.
    """
    payment = 0.
    # law-abiding agents
    if not self.is_evader:
      payment = self.wealth * self.model.collecting_rates[self.segment]
      self.wealth -= payment
      self.model.common_fund += payment
    # evader agents:
    else:
      if np.random.uniform(0, 1) < self.model.catch:
        payment = self.wealth * self.model.collecting_rates[self.segment] * \
          (1 + self.model.fine_rate)
        if payment >= self.wealth:
          payment = self.wealth
          self.model.common_fund += payment
          self.wealth = 0
        else:
          self.wealth -= payment
          self.model.common_fund += payment
    self.payment = payment
          

class Society:
  """
  A very simple of a society where taxes are collected and redistributed.
  """

  def __init__(self, num_agents, num_evaders, collecting_rates, 
               redistribution_rates, invest_rate, catch, fine_rate):
    assert len(collecting_rates) == len(
      redistribution_rates), "different number of collecting and \
      redistributing segments."
    self.num_segments = len(collecting_rates)  # number of segments
    self.num_agents = num_agents  # number of agents
    assert num_evaders <= self.num_agents, "more evaders than agents"
    self.num_evaders = num_evaders  # number of evader agents
    
    self.collecting_rates = collecting_rates  # collecting rates by group
    self.redistribution_rates = redistribution_rates  # redist. rates by group
    self.catch = catch  # probability of catching an evader
    self.fine_rate = fine_rate  # fine to be imposed if an evader in caught
    
    self.invest_rate = invest_rate  # interest return to the investment
    self.common_fund = 0.  # collected taxes for each transition

    # create agents
    self.agents = []
    for i in range(self.num_agents):
      a = Individual(self)
      self.agents.append(a)
      
    # assign some of the agents as evaders randomly
    evaders = np.random.choice(self.agents, size=self.num_evaders,
                               replace=False)
    for ev in evaders:
      ev.is_evader = True

    # assign agents to their wealth group
    self.assign_agents_to_segments()
    
    self.time_step = 0

  def assign_agents_to_segments(self):
    """
    Assign the agents in a model to their wealth segment and overall position.
    """
    # assign agents to their wealth segment
    sorted_agents = sorted(self.agents, key=lambda a: a.wealth)
    # assign agents to their position in ranking
    for i in range(len(sorted_agents)):
      setattr(sorted_agents[i], 'position', self.num_agents-i)
    # assign agents to their segment
    cut_index = int(self.num_agents / self.num_segments)
    for n in range(self.num_segments):
      for ag in sorted_agents[n * cut_index: (n + 1) * cut_index]:
        setattr(ag, 'segment', n)
      try:
        [setattr(ag, 'segment', n) for ag in sorted_agents[(n + 1)*cut_index:]]
      except:
        pass

  def step(self):
    """
    Taxes and (if any) fines are collected into the common fund, and 
    redistributed with interest.
    """
    self.common_fund = 0.
    # collect taxes from all agents
    for ind in self.agents:
      _ = ind.step()
    # redistribute common fund
    for ind in self.agents:
      payback = self.common_fund * (1 + self.invest_rate) * \
        self.redistribution_rates[ind.segment] * self.num_segments \
          / self.num_agents
      ind.wealth += payback
      ind.payback = payback
    # recompute segments
    self.assign_agents_to_segments()
    self.time_step += 1
