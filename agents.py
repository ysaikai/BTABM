import numpy as np
import random
from mesa import Agent

"""
Farmer
"""
class Farmer(Agent):
  def __init__(self, fid, pos, model):
    super().__init__(fid, model)
    self.pos = pos
    self.vision = random.randrange(0,3) # max size = 2
    # self.vision = np.random.choice([0,1,2],p=[.25,.5,.25])
    self.BT = False
    self.banned = False
    self.sprayed = False
    self._nextBT = False # next choice
    self.profit = None
    self.pi_diff = 0 # profit difference btw alternatives
    self.neighbors = list()
    self.prob = None # switching probability


  def step(self):
    """Skip the prior periods (pp)
    model.schedule.steps starts with 0
    """
    if self.model.schedule.steps < self.model.pp:
      return None

    """BT ban"""
    if self.banned:
      self._nextBT = False
      return None

    """Decision
    If no neighbors OR by some chance, decide rationally; otherwise, mimic.
    """
    choice, self.pi_diff = self.rational_choice()
    if not self.neighbors or (random.random() < self.per): # rational
      if self.BT == choice: # if the alternative is no better
        self.prob = 0
      else:
        self.prob = 1 - np.exp(-self.model.beta*self.pi_diff)

      if random.random() < self.prob:
        self._nextBT = not self.BT
    else: # mimicking
      self._nextBT = (np.mean([a.BT for a in self.neighbors]) >= .5)

    if self._nextBT == False:
      self.pi_diff = 0


  def rational_choice(self):
    pi_BT = self.model.get_profit(True, self)
    pi_Non = self.model.get_profit(False, self)
    """Return whether BT is a better choice and the difference"""
    return ((pi_BT > pi_Non), abs(pi_BT - pi_Non))


  def advance(self):
    self.BT = self._nextBT
    self.profit = self.model.get_profit(self.BT, self)


"""
Pest
"""
class Pest(Agent):
  ini_prop_RR = .001
  ini_prop_SS = .999

  def __init__(self, iid, pos, model):
    super().__init__(iid, model)
    self.iid = iid
    self.pos = pos
    self.host = None

    prop_RR = self.ini_prop_RR
    prop_SS = self.ini_prop_SS
    prop_RS = 1 - prop_RR - prop_SS
    self.genotypes = np.array((prop_RR, prop_SS, prop_RS))
    self.destinations = list()

    """0: Previous, 1: Current to use a logistic growth"""
    self.population = np.zeros(2)
    r = int(random.random() < .5)
    self.population = np.array((random.random()+.2, random.random()+.2))
    model.reproduce()
