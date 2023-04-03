#!/c/Users/Bardi/AppData/Local/Programs/Python/Python39/python -i
from data_fetch import data_fetch
from regression import regression
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mainstat import mainstat
from sys import argv

msts = {}

# class OutputWrapper:
#   def __init__(self, to_file):
#     self._to = to_file

#   def write(self, data):
#     # let's write to log file with some extra string.
#     # self._to.write("-----")
#     self._to.write(data)

def train(arg='monohulled'):
  mst = mainstat(arg)
  mst.initial()
  for i in range(6):
    mst.repeat()
  mst.prepare()
  mst.data_frame.to_csv(arg + '_full.csv')
  mst.data_frame.set_index('Variant')['vscore'].to_csv(arg + '_vscore.csv')
  # mst.data_frame['rscore'].reset_index().to_csv(arg + '_rscore.csv', index=False)
  mst.region_pref()
  # mst.region_preference.reset_index().to_csv(arg + '_rpref.csv', index=False)
  msts[arg] = mst

def task_train():
  if len(argv) > 1:
    train(argv[1])
  else:
    print("---monohulled")
    train('monohulled')
    print("---catamarans")
    train('catamarans')

def task_plot():
  sp = msts['monohulled'].data_frame.sort_values(by='Price')['Price']
  psp = msts['monohulled'].data_frame.sort_values(by='Price')['PredPrice']
  plt.plot(sp[sp < 600000], psp[sp < 600000], label='Monohulled Boats')
  sp = msts['catamarans'].data_frame.sort_values(by='Price')['Price']
  psp = msts['catamarans'].data_frame.sort_values(by='Price')['PredPrice']
  plt.plot(sp[sp < 600000], psp[sp < 600000], label='Catamarans')
  plt.legend()

def task_test():
  print("==testing hong kong==")
  print('--mono--')
  hk = pd.read_csv('hkmono.csv')
  msts['monohulled'].hk = 243026.933
  msts['monohulled'].test_regression(hk, 'linear')
  hk.to_csv('hk_m_full.csv', index=False)
  print('--cata--')
  hk = pd.read_csv('hkcata.csv')
  msts['catamarans'].hk = 387162.866
  msts['catamarans'].test_regression(hk, 'forest')
  hk.to_csv('hk_c_full.csv', index=True)
  

def task_pv():
  pass

task_train()

# # mst.region_evaluate()
# mst.normal_evaluate_all()
# mst.year_f()
# # mst.region_evaluate()
# mst.normal_evaluate_all()
# mst.variant_f()
# # mst.region_evaluate()
# # mst.region_f()
# mst.region_plot()
# mst.normal_evaluate_all()
# mst.score_regression()
