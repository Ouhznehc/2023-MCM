import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class regression:
  X = pd.Series([], dtype='int64')
  y = pd.Series([], dtype='int64')
  model_type = None
  feature_names = np.ndarray([])

  def __init__(self, X, y, feature_names, model_type = LinearRegression):
    self.X = X
    self.y = y
    self.model = model_type()
    self.model_type = model_type
    self.feature_names = feature_names
    self.model.feature_names_ = feature_names
    self.model.fit(self.X, self.y)

  def mean_squared_error(self):
    y_pred = self.model.predict(self.X)
    return mean_squared_error(self.y, y_pred)

  def reset(self):
    self.model = self.model_type()

  def train(self):
    self.model.fit(self.X, self.y)

  def evaluate(self, plot=False):
    y_pred = pd.Series([], dtype='float64')
    for i in range(len(self.X)):
      self.reset()
      X_test = pd.DataFrame([self.X.iloc[i]], columns=['Length', 'Year'])
      tmp_X = self.X.drop(self.X.index[i])
      tmp_y = self.y.drop(self.X.index[i])
      self.model.feature_names_ = self.feature_names 
      self.model.fit(tmp_X, tmp_y)
      y_pred = pd.concat([y_pred, pd.Series(self.model.predict(X_test))])
    y_pred.index = self.y.index
    y_pred.name = "pred price"
    pred_frame = pd.concat([y_pred.apply(lambda x : math.exp(x)), self.y.apply(lambda x : math.exp(x))], axis='columns')
    residual = pred_frame.apply((lambda x : (x.iloc[0] - x.iloc[1]) * 100 / x.iloc[1]), axis="columns")
    residual.index = range(len(residual.index))
    residual.plot()
    print("Square root of Evaluated MSE:", math.sqrt(mean_squared_error(self.y.apply(lambda x : math.exp(x)), y_pred.apply(lambda x : math.exp(x)))))
    return residual