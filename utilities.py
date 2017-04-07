
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import calendar
import time

# Helper function for reading Google trends data
def read_csv(input_filepath, delimiter):
  return pd.read_csv(input_filepath, delimiter=delimiter, header=0)

# Currently only works for two-dimensional embeddings
def plot_series(data, input_filepath, keyword):
  data['date'] = data['date'].astype('datetime64[ns]')

  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.set_ylabel(r"Relative Search Interest")
  ax.set_xlabel(r"Time")

  ax.plot_date(data['date'].values, data[keyword].values, color="k", ls="solid", ms=2)

  plt.ylim(0, 100)

  plt.savefig("{0}.png".format(input_filepath+".trend")) 

# Currently only works for two-dimensional embeddings
def plot_embedding(embedded, input_filepath, dimensions):
  x = [row[dimensions[0]] for row in embedded]
  y = [row[dimensions[1]] for row in embedded]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  if dimensions[0] == 0:
    ax.set_ylabel(r"$x(t)$")
  elif dimensions[0] == 1:
    ax.set_ylabel(r"$x(t+\tau)$")  
  else:  
    ax.set_ylabel(r"$x(t+{0}*\tau)$".format(dimensions[0]))

  if dimensions[1] == 0:
    ax.set_xlabel(r"$x(t)$")
  elif dimensions[1] == 1:
    ax.set_xlabel(r"$x(t+\tau)$")
  else:
    ax.set_xlabel(r"$x(t+{0}*\tau)$".format(dimensions[1]))
  
  plt.scatter(y, x, s=4, color="k")

  plt.xlim(0, 100)
  plt.ylim(0, 100)

  plt.savefig("{0}.png".format(input_filepath+".embed")) 
