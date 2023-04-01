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
  variants = set()
  coef_cept = dict()
  def __init__(self):
    self.data_frame = data_fetch().data
    for make in self.data_frame.Make.values:
      self.makes.add(make)
    for region in self.data_frame.CRS.values:
      self.regions.add(region)
    for index in self.data_frame.index:
      self.variants.add("{} {}".format(self.data_frame.loc[index].Make, self.data_frame.loc[index].Variant))

  def byregion(self, region, features=['Length', 'Year'], plot=True, make=None):
    print("Select Region: {}".format(region))
    if make:
      print("Select make: {}".format(make))
    else:
      make = self.data_frame[self.data_frame['CRS'] == region]['Make'].mode()[0]
      print("Most frequent make: {}".format(make))
    dt = self.data_frame[self.data_frame['CRS'] == region]
    dt = dt[dt['Make'] == make][features + ['Price']]
    self.lr = regression(dt[features], dt['Price'], [features])
    self.coef_cept[(region, make)] = self.lr.model.intercept_ + self.lr.model.coef_
    residal = self.lr.evaluate()
    if type(residal) == type(None):
      return
    residal.name = "{}.{}".format(region, make)
    if residal[residal > 100].any():
      print(residal)
    if plot:
      ax = residal.plot(legend=residal.name)
    


  def bymake(self, make, features=['Length', 'Year'], plot=True):
    print("Select make: {}".format(make))
    dt = self.data_frame[self.data_frame['Make'] == make][features + ['Price']]
    self.lr = regression(dt[features], dt['Price'], [features])
    self.coef_cept[make] = self.lr.model.intercept_ + self.lr.model.coef_
    residal = self.lr.evaluate()
    if type(residal) == type(None):
      return
    residal.name = make
    # print(residal)
    if plot:
      residal.plot()
    