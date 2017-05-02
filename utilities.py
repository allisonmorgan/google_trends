
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from matplotlib import rcParams

# Try to make plots look nice
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

rcParams['xtick.major.pad'] = '8'
rcParams['lines.solid_capstyle'] = 'butt'

# Dates have been removed from the data, so add them back in
dates = pd.read_csv("data/dates_hourly.csv", names=["date"], parse_dates=[0])["date"].tolist()

# Helper function for reading Google trends data
def read_csv(input_filepath, delimiter):
  return pd.read_csv(input_filepath, delimiter=delimiter, header=0)

def plot_series(data, input_filepath, keyword, save_output = True):
  #data['date'] = [date.astype('datetime64[ns]') for date in dates]
  #data['date'] = dates

  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.set_ylabel(r"Relative Search Interest")
  ax.set_xlabel(r"Time")

  ax.plot_date(data['date'].values, data[keyword].values, color="blue", ls="solid", ms=2)  
  plt.xticks(rotation=45)
  #ax.plot_date(range(len(data[keyword].values)), data[keyword].values, color="k", ls="solid", ms=2)

  plt.ylim(0, 100)
  fig.tight_layout()

  if save_output:
    plt.savefig("{0}.png".format(input_filepath+".trend")) 

  return [data['date'].values, data[keyword].values]

# Currently only works for two-dimensional embeddings
def plot_embedding(embedded, input_filepath, dimensions, save_output = True):
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
  
  plt.scatter(y, x, s=4, color="blue")

  plt.xlim(0, 100)
  plt.ylim(0, 100)

  if save_output:
    plt.savefig("{0}.png".format(input_filepath+".embed")) 

def plot_prediction(input_filepath, truth, prediction, test):
  fig = plt.figure(figsize=(12,6))
  ax = fig.add_subplot(111)
  
  ax.set_ylabel(r"Relative Search Interest")
  ax.set_xlabel(r"Time")

  ax.plot_date(dates[len(dates) - test:], truth, mec="blue", mfc="white", ms=4) 
  ax.plot_date(dates[len(dates) - test:], prediction, color="red", marker='+', ms=4)
  plt.xticks(rotation=45)
  plt.ylim(0, 50)

  fig.tight_layout()
  plt.savefig(input_filepath + "_prediction.png")

