
import numpy as np
import random
from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agents import *

class BTABM(Model):
  proportion_corn = .44 # Corn fields / Cropped land
  base_refuge = .05
  rs_survival = 0.18
  price = 129.91 # output price
  harvest = 10.92
  costs = 1202.51
  spray_cost = 33.51 # cost of spraying
  fees = [17.49, 17.49, 17.49, 17.49, 17.49, 17.49, 17.49, 17.49, 17.45, 17.04,
    15.78, 13.75, 11.41, 9.18, 8.29, 7.82, 7.39, 7.04]

  adoption_data = [.01, .08, .13, .10, .14, .12, .17, .23, .24, .28, .32, .41,
    .49, .50, .51, .59, .63, .66, .75, .73, .73, .73] # 22 points (1996-2017)

  def __init__(self, height=30, width=70, seed=None, pp=11, tax=0, refuge=.05,
    ban=(0,1), spray=(0,0), beta=.0036, per=.3):
    """Random seed"""
    if seed is not None:
      super().__init__(seed)

    self.width = width
    self.height = height
    self.beta = beta
    self.size_grid = int(width * height)
    self.pp = pp
    self.dispersion = height // 10 # How far pest reaches
    self.threshold = 0.5 # for banning BT
    self.efficacy = 0.8 # sparying efficacy
    self.num_data = len(type(self).adoption_data)
    self.tax = tax
    r = self.base_refuge
    """Fees (0 added for pp and discounted due to 5% default refuge)"""
    fees = [0]*pp + [_*(1-r) for _ in type(self).fees]
    self.fees = fees
    self.fee = 0 # init
    self.rate_survival = [1, r, r+(1-r)*self.rs_survival]

    self.schedule = SimultaneousActivation(self)
    self.grid = MultiGrid(width, height, torus=True)

    """Intervention"""
    self.flag_refuge = False # whether refuge has already been updated
    self.flag_tax = False # whether tax has already been updated
    self.timing = pp + self.num_data
    self.refuge = refuge
    self.ban = ban[0]
    self.ban_radius = ban[1]
    self.spray = spray[0]
    self.spray_radius = spray[1]

    """Create agents (both farmer and pest)"""
    iid = -1 # insect id
    self.farmers = dict()
    self.pests = dict()
    for contents, x, y in self.grid.coord_iter():
      if random.random() < self.proportion_corn:
        iid += 1
        pest = Pest(iid, (x,y), self)
        fid = iid # farmer id
        farmer = Farmer(fid, (x,y), self)
        farmer.per = per
        farmer.pest = pest
        farmer.profit = self.get_profit(farmer.BT, farmer)
        self.farmers[fid] = farmer
        self.pests[iid] = pest
        pest.host = farmer
        self.schedule.add(farmer)
        self.grid.place_agent(farmer, (x,y))
        self.grid.place_agent(pest, (x,y))

    self.N = len(self.schedule.agents) # number of agents
    self.running = True

    """Pest sources and Neighbors for each farmer
    Pest is assumed to detect corn, instead of randomly disperse and
    die if landing on an empty cell.
    """
    for iid, pest in self.pests.items():
      """Neighbor farmers (include himself)"""
      farmer = pest.host
      neighbors = self.grid.get_neighbors(farmer.pos, moore=True,
        include_center=False, radius=farmer.vision)
      farmer.neighbors = [a for a in neighbors if type(a) is Farmer]

      """Pest destinations"""
      destinations = self.grid.get_neighbors(pest.pos, moore=True,
        include_center=True, radius=self.dispersion)
      destinations = [a for a in destinations if type(a) is Pest]
      pest.destinations = destinations
      # the set is defined to be twice as large
      destinations2 = self.grid.get_neighbors(pest.pos, moore=True,
        include_center=True, radius=self.dispersion*2)
      destinations2 = [a for a in destinations2 if type(a) is Pest]
      pest.destinations2 = destinations2

    """DataCollector"""
    # For simplicity, $values (prof, rev, and spray cost) are the landscape
    # averages. This is fine as long as being consistent.
    # Although spray costs are included in profits, collect them seprate for
    # the purpose of presentation.
    self.dc = DataCollector(
      model_reporters = {
        "seed": lambda m: m.seed,
        "Pest": lambda m: np.mean([a.population[1] for a in m.pests.values()]),
        "R": lambda m: np.mean([a.genotypes[0]+a.genotypes[2]/2 for a in m.pests.values()]),
        "S": lambda m: np.mean([a.genotypes[1]+a.genotypes[2]/2 for a in m.pests.values()]),
        "BT": lambda m: sum(a.BT for a in m.farmers.values()) / self.N,
        "Profit": lambda m: np.mean([a.profit for a in m.farmers.values()]),
        "Benefit": lambda m: np.mean([a.pi_diff for a in m.farmers.values()]),
        "Spray": lambda m: sum(a.sprayed for a in m.farmers.values())*self.spray_cost/self.N,
        "Revenue": lambda m: sum(a.BT for a in m.farmers.values())*self.fee/self.N},
      agent_reporters = {
        "Profit": lambda a: a.profit,
        "BT": lambda a: a.BT,
        "Benefit": lambda a: a.pi_diff,
        "Prob": lambda a: a.prob,
        "Pest": lambda a: a.pest.population[1],
        "R": lambda a: a.pest.genotypes[0]+a.pest.genotypes[2]/2}
    )


  def step(self):
    self.update_fee()
    self.schedule.step()
    self.apply_BT()
    """
    Intervention affecting the following periods.
    Development is judged after BT toxin effect and before reproduction.
    """
    if self.schedule.steps >= self.timing:
      """Tax change should precedes refuge change"""
      if self.flag_tax == False and self.tax > 0:
        self.trigger_tax()
      if self.flag_refuge == False and self.refuge > self.base_refuge:
        self.trigger_refuge()
      if self.spray:
        self.spray_adults()
      if self.ban:
        self.ban_BT()
    """Reproduction"""
    self.mate()
    self.reproduce()
    self.disperse()

    self.dc.collect(self)


  def update_fee(self):
    if self.schedule.steps < len(self.fees):
      self.fee = self.fees[self.schedule.steps]
    else:
      self.fee = self.fees[-1] # keep using the last available fee


  def check_dev(self, f):
    return f.pest.genotypes[0]+f.pest.genotypes[2]/2 >= self.threshold


  def trigger_tax(self):
    for f in self.farmers.values():
      if self.check_dev(f):
        """
        n.b. beyond the number of the fee data, the last one is assumed to be
        used. So, append a taxed one to the last.
        """
        self.fees.append( self.fees[-1]*(1+self.tax) )
        self.flag_tax = True
        break


  def trigger_refuge(self):
    for f in self.farmers.values():
      if self.check_dev(f):
        """Effective survival rates and tech fees given the refuge policy"""
        r = self.refuge
        self.rate_survival = [1, r, r+(1-r)*self.rs_survival]
        self.fees[-1] = self.fees[-1]/(1-self.base_refuge)*(1-r)
        self.flag_refuge = True
        break


  def spray_adults(self):
    for f in self.farmers.values():
      f.sprayed = False
      """radius is either 0 (only its own cell) or the one used for dispersal"""
      if self.spray_radius == 0:
        f.sprayed = self.check_dev(f)
      else:
        for p in f.pest.destinations:
          if self.check_dev(p.host):
            f.sprayed = True
            break
      """n.b. ok to change pop in the same iter as only genotypes are used"""
      if f.sprayed:
        f.pest.population[1] *= (1-self.efficacy) # insect reduction
        f.profit -= self.spray_cost


  def ban_BT(self):
    for f in self.farmers.values():
      f.banned = False # init
      """radius is either the dispersal or twice of it"""
      if self.ban_radius==1:
        destinations = f.pest.destinations
      else:
        destinations = f.pest.destinations2
      """banned if any development in the neighborhood"""
      for p in destinations:
        if self.check_dev(p.host):
        # if p.genotypes[0]+p.genotypes[2]/2 >= self.threshold:
          f.banned = True
          break


  def apply_BT(self):
    for a in [a for a in self.farmers.values() if a.BT]:
      a.pest.genotypes = a.pest.genotypes * self.rate_survival
      tmp = sum(a.pest.genotypes)
      a.pest.population[1] *= sum(a.pest.genotypes)
      a.pest.genotypes = a.pest.genotypes / tmp


  def reproduce(self):
    capacity = 1.4
    ω = 2.15;

    for pest in self.pests.values():
      """
      A logistic growth model to generate a generic oscilation
      with wavelength=7 and apmplitude=0.4 around 0.6.
      """
      x = pest.population
      _next = max(ω*x[1]*(1 - x[0]/capacity), 0)
      pest.population[0] = pest.population[1]
      pest.population[1] = _next


  """Update the genotype proportions"""
  def mate(self):
    for pest in self.pests.values():
      """p[0]: %RR, p[1]: %SS, p[2]: %RS"""
      p = pest.genotypes
      p[0] = p[0]**2 + p[0]*p[2] + .25*p[2]**2
      p[1] = p[1]**2 + p[1]*p[2] + .25*p[2]**2
      p[2] = 1 - p[0] - p[1]
      pest.genotypes = p


  """
  Each cell calculates incoming pests (including her own) as a sum of
  the source cells' fraction of the outgoing pests. Each fraction is stored
  in each cell's destinations dictionary with key=fid as a target neighbor.
  """
  def disperse(self):
    _population = np.zeros(self.N)
    _genotypes = np.zeros((self.N, 3))

    """Caluculate"""
    for p in self.pests.values():
      num_dest = len(p.destinations)
      for dest in p.destinations:
        inc = p.population[1]/num_dest
        _population[dest.iid] += inc
        _genotypes[dest.iid] += inc * p.genotypes

    """Update"""
    for p in self.pests.values():
      p.population[1] = _population[p.iid]
      if sum(_genotypes[p.iid]) > 0:
        p.genotypes = _genotypes[p.iid] / sum(_genotypes[p.iid])


  def get_profit(self, BT, farmer):
    """
    The loss function is based on the supplemental document of
    Hutchison et al (2010). n.b. It has an error in E(λ)!!
    """
    pest = farmer.pest
    x = pest.population[1]
    if BT:
      genotypes = pest.genotypes * self.rate_survival
      x = x * sum(genotypes)
    if x<0:
      x = 0

    harvest = self.harvest
    price = self.price
    costs = self.costs
    fee = BT*self.fee
    α = 1/.58
    m2 = (2.56*x + 5.65*x**.5)**2
    s2 = (3.4+1.73*x)**2
    loss = .021*m2**((2*α-1)/(2*α**2)) * (1/(s2+m2))**((α-1)/(2*α**2))
    return price*harvest*(1 - loss) - costs - fee
