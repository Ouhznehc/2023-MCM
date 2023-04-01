#!python -i
from data_fetch import data_fetch
from regression import regression
import math
import matplotlib.pyplot as plt
from mainstat import mainstat

mst = mainstat()

for region in mst.regions:
  mst.byregion(region)

plt.show()