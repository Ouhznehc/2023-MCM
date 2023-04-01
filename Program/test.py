#!python -i
import csv
import numpy as np
import matplotlib as plt

class boat_data_analysis:
  
  boat_data = []
  catagories = dict()
  header = []
  body = []
  catagory_labels = dict()

  def generate_data_series(vary, condition):
    for boat_info in body:
      for catagory in condition:
        if boat_info[catagories[catagory]] != condition[catagory]:
          break
      else:
        continue
      
    return

  def __init__(self, boat_filename):
    with open(boat_filename, "r") as boat_file:
      self.boat_data = list(csv.reader(boat_file))

    self.header = self.boat_data[0]
    for index in range(len(self.header)):
      self.catagories[self.header[index]] = index 

    self.body = self.boat_data[1:]
    self.catagory_labels = {name : set() for name in self.header}

    for boat_info in self.body:
      boat_info[self.catagories['Variant']] = "{} {}".format(boat_info[self.catagories['Make']].strip(), boat_info[self.catagories['Variant']].strip())
      boat_info[self.catagories['Price']] = int("".join(boat_info[self.catagories['Price']].strip('$ ').split(',')))
      for name in self.header:
        if type(boat_info[self.catagories[name]]) == type(''):
          self.catagory_labels[name].add(boat_info[self.catagories[name]].strip()); 
        else:
          self.catagory_labels[name].add(boat_info[self.catagories[name]]); 

# for i in body:
  
if __name__ == '__main__':
  filename = input()
  bda = boat_data_analysis(filename)