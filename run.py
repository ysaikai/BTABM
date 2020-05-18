import datetime
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
  mode = int(sys.argv[1]) # mode: 1. Visualization, 2. Console for details
  if not(mode == 1 or mode == 2):
    print("Give me 1 or 2.")
    quit()
except IndexError:
  mode = 2
  print("A wrong argument! Processing as if '{}' were given...".format(mode))

if mode == 1:
  num_steps = 61
  h = 25
  w = 50
  scale = 750 // max(w, h)
  import server
  server.launch(w, h, scale, num_steps)

elif mode == 2:
  # start = "{:%Y-%m-%d, %H:%M}".format(datetime.datetime.now()) # for notification
  num_steps = 61

  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  from model import BTABM

  '''Seed''' # must be between 0 & 4294967295
  seed = int( '{:%f}'.format(datetime.datetime.now()) )
  print("seed =", seed)

  model = BTABM(seed=seed)
  for i in range(num_steps):
    step = model.schedule. steps
    model.step()
    print("{:3d}. BT = {:.2f}, Pest = {:.2f}, %R = {:.2f}, Benefit = {:.2f}, fee = {:.2f}".format(
      step,
      sum(a.BT for a in model.farmers.values()) / model.N,
      np.mean([a.population[1] for a in model.pests.values()]),
      np.mean([a.genotypes[0]+a.genotypes[2]/2 for a in model.pests.values()]),
      np.mean([a.pi_diff for a in model.farmers.values()]),
      model.fee
      ))


  '''Model data'''
  df_model = model.dc.get_model_vars_dataframe()
  df_model = df_model.round(2)
  cols = ["BT", "Pest", "R", "Profit", "Benefit", "Spray", "Revenue"]
  df_model = df_model[cols]
  # df_model.to_csv("output/{:%Y%m%d%H%M%S.csv}".format(datetime.datetime.now()))
  print(df_model)


  '''Plot'''
  x = range(num_steps)
  y1 = df_model['BT'].values*100
  y2 = df_model['Pest'].values
  c2 = "#ffe699"
  y3 = df_model['R'].values*100
  c3 = "#cc3300"

  fig, ax1 = plt.subplots(figsize=(20,4))
  ax2 = ax1.twinx()
  ax2.plot(x, y1, 'b', x, y3, c3, linewidth=2)
  ax1.plot(x, y2, c2, linewidth=2)
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
  ax2.set_ylabel('%Bt Adoption, %R allele')
  ax1.set_ylabel('Larvae / plant')
  font = {'size':14}
  plt.rc('font',**font)
  plt.show()
