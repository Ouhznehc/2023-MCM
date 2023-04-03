from data_fetch import data_fetch
from regression import regression
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class mainstat:
  data_frame = pd.DataFrame([[]],dtype='int64')
  normal_frame = pd.DataFrame([[]],dtype='int64')

  model = None
  lr = None
  size = 0
  regions = set()
  geos = set()
  reg2geo = dict()
  geo2reg = dict()
  makes = set()
  variants = set()
  coef_cept = None
  year_mean = None
  region_score = None
  region_preference = None
  variant_score = None
  X = None
  y = None
  hk_rscore = 0

  def __init__(self, name="monohulled"):
    self.data_frame = data_fetch(name+".csv").data
    self.normal_frame = self.data_frame
    self.size = len(self.data_frame.index)
    for make in self.data_frame.Make.values:
      self.makes.add(make)
    for region in self.data_frame.CRS.values:
      self.regions.add(region)
    for geo in self.data_frame.GeoRegion.values:
      self.geos.add(geo)
      self.geo2reg[geo] = set()
    for index in self.data_frame.index:
      self.data_frame.loc[index, 'Variant'] = "{} {}".format(self.data_frame.loc[index].Make, self.data_frame.loc[index].Variant)
      self.variants.add(self.data_frame.loc[index].Variant)
      self.reg2geo[self.data_frame.loc[index].CRS] = self.data_frame.loc[index].GeoRegion
      self.geo2reg[self.data_frame.loc[index].GeoRegion].add(self.data_frame.loc[index].CRS)

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
    
  def price_plot(self, frame, func=None, make=None, region=None, Year=None):
    # print(frame)
    price_v = frame.Price.sort_values()
    if func:
      price_v = price_v.apply(func)
    price_v.index = (range(len(price_v.index)))
    # price_log = price_v.apply(lambda x: math.log(x, 1.1))
    # bindex = np.arange(0, len(price_v.index))
    # poly = np.polyfit(price_log.to_numpy(), bindex, 1)
    # bindex_pred = np.polyval(poly, price_log.to_numpy())
    # print(poly)
    price_v.plot(legend="Original Price")
    plt.xlabel('Boat index')
    plt.ylabel('Price')
    # plt.title('The Original and Predicated Price by Polynominal Fitting')
    plt.title('Boat Price Distribution')
    # plt.plot(bindex_pred, price_v.to_numpy(), label="Pred Price")
    plt.legend()
    # plt.show()

  def year_normalize(self, plot=False, stdyear=2010):
    def normal(x):
      x['Price'] *= self.year_mean.loc[stdyear] / self.year_mean.loc[x['Year']]
      return x
    self.normal_frame = self.normal_frame.apply(axis='columns', func=normal)

  def year_disnormalize(self, plot=False, stdyear=2010):
    def disnormal(x):
      x['Price'] *= self.year_mean.loc[x['Year']] / self.year_mean.loc[stdyear]
      return x
    self.normal_frame = self.normal_frame.apply(axis='columns', func=disnormal)

  # def Variant_plot

  def year_plot(self, limit=0, plot=False, make=None, region=None, Year=None):
    variant_cnt_list = self.normal_frame['Variant'].value_counts()
    variant_cnt_list = variant_cnt_list[variant_cnt_list > limit]
    year_mean = self.normal_frame.set_index('Variant').loc[variant_cnt_list.index.to_numpy()].groupby('Year')['Price'].mean()
    # print(year_mean)
    if plot:
      year_mean.plot(legend="price")
    year_df = year_mean.reset_index()
    print("Corr (pearson): {:.3f}".format(year_df['Price'].corr(year_df['Year'], method='pearson')))
    print("Corr (spearman): {:.3f}".format(year_df['Price'].corr(year_df['Year'], method='spearman')))
    print("Corr (kendall): {:.3f}".format(year_df['Price'].corr(year_df['Year'], method='kendall')))
    _, p = pearsonr(year_mean.index, year_mean.values)
    print("p: {:.9f}, significant".format(p))
    poly = np.polyfit(year_mean.index, year_mean.values, 1)
    print(poly)
    year_mean_fit = np.polyval(poly, year_mean.index)
    self.year_mean = pd.Series(year_mean, index=range(2005, 2020))
    self.data_frame['yscore'] = self.data_frame['Year'].map(self.year_mean)
    if plot:
      plt.plot(year_mean.index, year_mean_fit, label="Price, linear fitted")
      plt.legend()
      plt.xlabel('Year')
      plt.ylabel('Price')
      plt.title('Average Price, By Year')

  def region_plot(self, plot=False):
    self.region_score = self.normal_frame.groupby('CRS')['Price'].mean()
    self.data_frame['rscore'] = self.data_frame['CRS'].map(self.region_score)
    if plot:
      print(self.region_score)
      self.region_score.plot(kind="bar")

  def region_normalize(self, plot=False, stdregion='Cyprus'):
    def normal(x):
      x['Price'] *= self.region_score.loc[stdregion] / self.region_score.loc[x['CRS']]
      return x
    self.normal_frame = self.normal_frame.apply(axis='columns', func=normal)

  def region_pref(self, limit=5):
    region_cnt_list = self.normal_frame['CRS'].value_counts()
    region_cnt_list = region_cnt_list[region_cnt_list > limit]
    self.variant_disnormalize(stdvariant=self.data_frame.Variant.mode()[0])
    self.region_preference = self.normal_frame.set_index('CRS').loc[region_cnt_list.index.to_numpy()].groupby('CRS')['Price'].mean()

  def region_disnormalize(self, plot=False, stdregion='Cyprus'):
    def disnormal(x):
      x['Price'] *= self.region_score.loc[x['CRS']] / self.region_score.loc[stdregion]
      return x
    self.normal_frame = self.normal_frame.apply(axis='columns', func=disnormal)

  def variant_plot(self, plot=False):
    self.variant_score = self.normal_frame.groupby('Variant')['Price'].mean()
    self.data_frame['vscore'] = self.data_frame['Variant'].map(self.variant_score)
    if plot:
      print(self.variant_score)
      self.variant_score.plot(kind="bar")

  def variant_normalize(self, plot=False, stdvariant=None):
    def normal(x):
      x['Price'] *= self.variant_score.loc[stdvariant] / self.variant_score.loc[x['Variant']]
      return x
    self.normal_frame = self.normal_frame.apply(axis='columns', func=normal)

  def variant_disnormalize(self, plot=False, stdvariant=None):
    def disnormal(x):
      x['Price'] *=  self.variant_score.loc[x['Variant']] / self.variant_score.loc[stdvariant]
      return x
    self.normal_frame = self.normal_frame.apply(axis='columns', func=disnormal)

  def price_distrib_plot(self):
    self.price_plot(self.data_frame)
    self.price_plot(self.normal_frame)
    # self.normal_frame.drop('Year', axis=1)
    plt.legend(["Price", "Price, normalized"])
    plt.title("Price Distribution Before and After Normalization")


  def year_f(self):
    print("Normalizing year")
    self.year_plot()
    self.year_normalize()

  def year_r(self):
    self.year_disnormalize()
    self.year_f()

  def region_f(self):
    print("Normalizing region")
    self.region_plot()
    self.region_normalize()

  def region_r(self):
    self.region_disnormalize()
    self.year_f()

  def variant_f(self):
    print("Normalizing variant")
    self.variant_plot()
    self.variant_normalize(stdvariant=self.data_frame.Variant.mode()[0])

  def variant_r(self):
    self.variant_disnormalize(stdvariant=self.data_frame.Variant.mode()[0])
    self.variant_f()

  def initial(self):
    self.year_f()
    self.variant_f()
    self.region_plot()
    self.score_regression()

  def repeat(self):
    self.year_r()
    self.variant_r()
    self.region_plot()
    self.score_regression()

  def prepare(self):
    self.score_regression('forest')

  def geo_plot(self, plot=False):
    geo_mean = self.normal_frame.groupby('GeoRegion')['Price'].mean()
    print(geo_mean)
    if plot:
      geo_mean.plot(kind="bar")

  def normal_evaluate(self, grouplist=['Variant']):
    var_base = self.normal_frame.groupby(grouplist)['Price'].agg(['var', (lambda x:len(x)), 'mean', (lambda x: "{:.2f}%".format(math.sqrt(x.var())/x.mean()*100))]).dropna()
    # print(var_base)
    var_mean = var_base.apply(func=lambda x:x['var'] * x['<lambda_0>'], axis=1).sum()
    var_max = var_base.sort_values(by='<lambda_0>', ascending=False).iloc[0]
    var_median = var_max['var']
    var_median_size = var_max['<lambda_0>']
    var_median_mean = var_max['mean']
    var_size = var_base['<lambda_0>'].sum()
    print("The average inpurity in {} of the sample after normalization is {:.2f}".format(grouplist, math.sqrt(var_mean/var_size)))
    print("The median inpurity in {} of the sample after normalization is {:.2f}".format(grouplist, math.sqrt(var_median)))
    total_mean = self.normal_frame.groupby(grouplist)['Price'].sum().sum()/var_size
    print("Inpurity rate (mean): {:.2f}%".format(math.sqrt(var_mean/var_size)/total_mean*100))
    print("Inpurity rate (max): {:.2f}%".format(math.sqrt(var_median)/var_median_mean*100))

  def normal_evaluate_all(self):
    self.normal_evaluate(['Variant', 'CRS'])
    self.normal_evaluate(['Year'])
    self.normal_evaluate(['CRS'])
    self.normal_evaluate(['Variant'])

  def train_init_l(self):
    self.X = self.data_frame[['yscore', 'rscore', 'vscore']].applymap(math.log)
    self.y = self.data_frame['Price'].apply(math.log)
    # self.model = CascadeForestRegressor(random_state=1)
    self.model = LinearRegression()
    self.model.fit(self.X, self.y)

  def train_init_d(self):
    self.X = self.data_frame[['yscore', 'rscore', 'vscore']].applymap(math.log)
    self.y = self.data_frame['Price'].apply(math.log)
    self.model = CascadeForestRegressor(random_state=1)
    # self.model = LinearRegression()
    self.model.fit(self.X, self.y)
    # self.coef_cept = [self.model.intercept_, self.model.coef_]
    return 

  def score_regression(self, method='linear'):
    if method == 'linear':
      self.train_init_l()
    else:
      self.train_init_d()
    y_pred = self.model.predict(self.X)
    mse = mean_squared_error(self.y.apply(math.exp), np.exp(y_pred))
    self.data_frame['PredPrice'] = np.exp(y_pred)
    self.data_frame
    print("Testing sqrt of MSE: {:.3f}, R^2: {:.3f}, MAPE: {:.2f}%".format(
      math.sqrt(mse),
      1 - (mse/self.data_frame['Price'].var()) * len(y_pred) / (len(y_pred) - 1),
      self.data_frame.apply(axis=1,func=lambda x:abs(x['PredPrice']-x['Price'])/x['Price']).mean()*100
    ))
    return 
  
  def test_regression(self, frame, method='linear'):
    if method == 'linear':
      self.train_init_l()
    else:
      self.train_init_d()
    y_pred = self.model.predict(frame[['yscore', 'rscore', 'vscore']].applymap(math.log))
    mse = mean_squared_error(frame['Money'], np.exp(y_pred))
    frame['PredPrice'] = np.exp(y_pred)
    print("Testing sqrt of MSE: {:.3f}, R^2: {:.3f}, MAPE: {:.2f}%".format(
      math.sqrt(mse),
      1 - (mse/frame['Money'].var()) * len(y_pred) / (len(y_pred) - 1),
      frame.apply(axis=1,func=lambda x:abs(x['PredPrice']-x['Money'])/x['Money']).mean()*100
    ))
    print(frame)
    print(frame.apply(axis=1,func=lambda x:abs(x['PredPrice']-x['Money'])/x['Money']))
    print(mse, frame['Money'].var(), len(y_pred))
    return 

  def region_score_evaluate(self, plot=True):
    # self.region_score = self.region_score.sort_
    rsdf = self.region_score.reset_index()
    rsdf['GeoRegion'] = rsdf['CRS'].map(self.reg2geo)
    rsdf = rsdf.sort_values(by='Price')
    self.region_score = rsdf.set_index('CRS')['Price']
    colors = [('green' if self.reg2geo[region] == 'USA' else 'orange' if self.reg2geo[region] == 'Europe' else 'purple') for region in self.region_score.index]
    print("unbiased standard deviation of region score is {:.2f}, rate: {:.2f}%".format(self.region_score.std(), self.region_score.std()/self.region_score.mean()*100))
    self.region_score.plot(kind='bar', color=colors)
    plt.ylabel('Price')
    plt.xlabel('Region')
    # plt.legend('Caribbean')
    custom_legend = [
      plt.Line2D([], [], color='orange', lw=2, label='Europe'),
      plt.Line2D([], [], color='green', lw=2, label='USA'),
      plt.Line2D([], [], color='purple', lw=2, label='Caribbean')
    ]
    plt.legend(handles=custom_legend)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)
    # plt.xticks(rotation=75)

  def region_length_evaluate(self, plot=True):
    # self.region_score = self.region_score.sort_
    rsdf = self.data_frame.groupby('CRS')['Length'].mean().sort_values().reset_index()
    rsdf['GeoRegion'] = rsdf['CRS'].map(self.reg2geo)
    rsdf = rsdf.sort_values(by='Length')
    tmp = rsdf.set_index('CRS')['Length']
    colors = [('green' if self.reg2geo[region] == 'USA' else 'orange' if self.reg2geo[region] == 'Europe' else 'purple') for region in tmp.index]
    tmp.plot(kind='bar', color=colors)
    plt.ylabel('Length')
    plt.xlabel('Region')
    custom_legend = [
      plt.Line2D([], [], color='orange', lw=2, label='Europe'),
      plt.Line2D([], [], color='green', lw=2, label='USA'),
      plt.Line2D([], [], color='purple', lw=2, label='Caribbean')
    ]
    plt.legend(handles=custom_legend)
    plt.tight_layout()

  def region_evaluate(self):
    self.region_plot()
    self.geo_plot()
    self.region_score_evaluate()
    plt.show()

  # def plot_region_score(self):
  #   plt.legend(["Price", "Price, normalized"])
  #   plt.title("Price Distribution Before and After Normalization")

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