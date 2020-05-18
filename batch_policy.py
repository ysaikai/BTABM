import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import datetime
import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunnerMP
import multiprocessing as mp
from model import BTABM

start = "{:%Y-%m-%d, %H:%M}".format(datetime.datetime.now()) # for notification

pp = 11 # prior_periods
nd = 22 # number of datapoints
num_steps = pp+nd+25 # pp + num_datapoints + policy effects windows
num_seeds = 1000

"""seeds"""
try:
  init_seed = int(sys.argv[1])
except IndexError:
  init_seed = 0
print("init_seed =", init_seed)
print("num_seeds =", num_seeds)
seeds = range(init_seed, init_seed+num_seeds)

refuges = [.05,.2,.5] # .05 as the baseline
bans = [(0,1),(1,1),(1,2)]
sprays = [(0,0),(1,0),(1,1)]
taxes = [0,.25,.5]
variable_parameters = {
  "seed": seeds,
  "refuge": refuges,
  "ban": bans,
  "spray": sprays,
  "tax": taxes,
  "beta": [.0022],
  "per": [1.0]
}
model_reporters = {"seed": lambda m: m.seed}
# for this trick, see http://stupidpythonideas.blogspot.com/2016/01/for-each-loops-should-define-new.html
for t in range(1,26):
  model_reporters["Prof"+"{:02d}".format(t)] = (lambda s: (lambda m: sum(m.dc.model_vars["Profit"][pp+nd:pp+nd+s])))(t)
  model_reporters["Bene"+"{:02d}".format(t)] = (lambda s: (lambda m: sum(m.dc.model_vars["Benefit"][pp+nd:pp+nd+s])))(t)
  model_reporters["Rev"+"{:02d}".format(t)] = (lambda s: (lambda m: sum(m.dc.model_vars["Revenue"][pp+nd:pp+nd+s])))(t)
  model_reporters["Spray"+"{:02d}".format(t)] = (lambda s: (lambda m: sum(m.dc.model_vars["Spray"][pp+nd:pp+nd+s])))(t)

batch_run = BatchRunnerMP(
  model_cls=BTABM,
  nr_processes=mp.cpu_count(),
  variable_parameters=variable_parameters,
  max_steps=num_steps,
  model_reporters=model_reporters
)
batch_run.run_all()


"""Resutls"""
df = batch_run.get_model_vars_dataframe()
df["b_radius"] = df["ban"] # copy
df["s_radius"] = df["spray"] # copy
cols = ["Run","seed","refuge","ban","b_radius","spray","s_radius","tax"]
for t in range(1,26):
  cols.append("Prof"+"{:02d}".format(t))
for t in range(1,26):
  cols.append("Bene"+"{:02d}".format(t))
for t in range(1,26):
  cols.append("Rev"+"{:02d}".format(t))
for t in range(1,26):
  cols.append("Spray"+"{:02d}".format(t))
df = df[cols]
# split policy ON/OFF flag and radius in a tuple
for index, row in df.iterrows():
  df.at[index,"ban"] = row["ban"][0]
  df.at[index,"b_radius"] = row["ban"][1]
  df.at[index,"spray"] = row["spray"][0]
  df.at[index,"s_radius"] = row["spray"][1]
for t in range(1,26):
  df = df.round({"Prof"+"{:02d}".format(t):1, "Bene"+"{:02d}".format(t):1, "Rev"+"{:02d}".format(t):1, "Spray"+"{:02d}".format(t):1})
df = df.set_index("Run")
df = df.sort_values(by=["refuge","ban","b_radius","spray","s_radius","tax"])
df.to_csv("output/{:p%Y%m%d%H%M%S.csv}".format(datetime.datetime.now()))


# """Email notification"""
# import notification
# notification.main(start)
