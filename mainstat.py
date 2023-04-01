from data_fetch import data_fetch
from regression import regression
import math
import matplotlib.pyplot as plt
import pandas as pd

class mainstat:
  data_frame = pd.DataFrame([[]],dtype='int64')
  lr = None
  regions = set()
  makes = set()
  def __init__(self):
    self.data_frame = data_fetch().data
    for make in data_frame.Make.values:
      self.makes.add(make)
    for region in data_frame.CRS.values:
      self.regions.add(region)

  def byregion(self, region, features=['Length', 'Year'], make=None):
    print("Select Region: {}".format(region))
    if make:
      print("Select make: {}".format(make))
    else:
      make = self.data_frame['Make'].mode()[0]
      print("Most frequent make: {}".format(make))
    dt = self.data_frame[self.data_frame['CRS'] == region]
    dt = dt[dt['Make'] == make][features + ['Price']]
    self.lr = regression(dt[features], dt['Price'], [features])

  def bymake(self, make, features=['Length', 'Year']):
    print("Select make: {}".format(make))
    dt = self.data_frame[self.data_frame['Make'] == make][features + ['Price']]
    self.lr = regression(dt[features], dt['Price'], [features])

  def plot(self):
    self.lr.evaluate()