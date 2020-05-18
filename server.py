from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

from agents import *
from model import BTABM

def portrayal(agent):
  inscribe_fid = False # Inscription of fid on each plot

  '''
  The argument, "agent", is obtained from get_cell_list_contents(). So, these
  are every types of agents placed on somewhere in the grid. As a result,
  "agent" may contain multiple agents, i.e. a list of agent objects.
  '''
  assert agent is not None
  # if type(agent) is Farmer:
  #   portrayal = {
  #     "Shape": "circle",
  #     "r": .6,
  #     "Filled": True,
  #     "Layer": 1,
  #     "Color": "black"
  #   }
  #   if agent.BT:
  #     portrayal['Filled'] = True
  #   else:
  #     portrayal['Filled'] = False
  #   return portrayal
  #
  # elif type(agent) is Pest and agent.genotypes[0]+agent.genotypes[2]/2 > 0.5:
  #   portrayal = {
  #     "Shape": "rect",
  #     "w": 1,
  #     "h": 1,
  #     "Filled": True,
  #     "Layer": 0,
  #     "Color": "gainsboro"
  #   }
  #   return portrayal
  if type(agent) is Farmer:
    portrayal = {
      "Shape": "circle",
      "r": .6,
      "Filled": True,
      "Layer": 1
    }
    if agent.BT:
      portrayal['Color'] = '#0066ff'
    else:
      portrayal['Color'] = '#339966'

    return portrayal

  elif type(agent) is Pest and agent.genotypes[0]+agent.genotypes[2]/2 > 0.4:
    portrayal = {
      "Shape": "rect",
      "w": 1,
      "h": 1,
      "Filled": True,
      "Layer": 0,
      "Color": "#ffe699"
    }
    return portrayal


def launch(w, h, scale, num_steps, seed=0):
  title = 'BTABM'
  grid = CanvasGrid(portrayal, w, h, w*scale, h*scale)
  chart = ChartModule([
    {'Label': 'BT', 'Color': 'blue'},
    {"Label": "R", "Color": "#cc3300"},
    {"Label": "Pest", "Color": "#ffd24d"}
  ], canvas_height=300, canvas_width=600, data_collector_name="dc")

  model_params = {"width": w, "height": h, "seed": seed}
  server = ModularServer(BTABM, [grid, chart], title, model_params)
  # server.max_steps = num_steps

  server.launch()
