#!python -i
import matplotlib.pyplot as plt
import pandas as pd
import math
class data_fetch:
  data = pd.DataFrame()

  def __init__(self, fn = "dealedmonohulled.csv"):
    self.data = pd.read_csv(fn)
    self.data = self.data.loc[:, 'Make':]
    self.data = self.data.applymap(lambda x: str(x).strip())
    self.data.Length = self.data.Length.apply(lambda x : int(x))
    self.data.Year = self.data.Year.apply(lambda x : (2025 - int(x)))
    # self.data.Year = self.data.Year.apply(lambda x : int(x))
    self.data.Price = self.data.Price.apply(lambda x : math.log(int(x)))
  
