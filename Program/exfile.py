#!/c/Users/Bardi/AppData/Local/Programs/Python/Python39/python -i
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import matplotlib.pyplot as plt
from deepforest import CascadeForestRegressor

flag = "cata"
add =  pd.read_csv(flag + "_add.csv")
# add.set_index('Variant')
full = pd.read_csv(flag + "_add_full.csv")

# full = full.set_index('Variant')

# v2m = {}
# v2s = {}
# v2l = {}

# for i in full.index:
#   tmp = full.loc[i]
#   if type(tmp.Make) == str:
#     v2m[i] = tmp.Make
#   else:
#     v2m[i] = tmp.Make[0]
#   if type(tmp.vscore) == np.float64:
#     v2s[i] = tmp.vscore
#   else:
#     v2s[i] = tmp.vscore[0] 
#   if type(tmp.Length) == np.float64:
#     v2l[i] = tmp.Length
#   else:
#     v2l[i] = tmp.Length[0] 

# add['Make'] = add['Variant'].map(v2m)
# add['vscore'] = add['Variant'].map(v2s)
# add['Length'] = add['Variant'].map(v2l)

def plot(param, bparam, kind='line', tight=False):
  full.groupby(param)[bparam].mean().sort_index().plot(kind=kind)
  plt.xlabel(param)
  plt.ylabel(bparam)
  if tight:
    plt.tight_layout()
  plt.show()

model = CascadeForestRegressor(random_state=1)
X = full[['Length', 'Beam', 'Displacement', 'Fuel', 'Water', 'Cabins', 'Draft']]
y = full['vscore']
model.fit(X, y)
y_pred = model.predict(X)
full['Predscore'] = y_pred
mse = mean_squared_error(y, y_pred)
print("Testing sqrt of MSE: {:.3f}, R^2: {:.3f}, MAPE: {:.2f}%".format(
      math.sqrt(mse),
      1 - (mse/full['vscore'].var()) * len(y_pred) / (len(y_pred) - 1),
      full.apply(axis=1,func=lambda x:abs(x['Predscore']-x['vscore'])/x['vscore']).mean()*100
    ))
# add.sort_values(by='Length').interpolate().to_csv(flag + "_add_full.csv", index=False)
# add = add.set_index('Length')

hk = pd.read_csv('hk'+flag+'.csv')
hk['vscore'] = model.predict(hk[['Length', 'Beam', 'Displacement', 'Fuel', 'Water', 'Cabins', 'Draft']])