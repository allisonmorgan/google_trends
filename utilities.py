
import matplotlib.pyplot as plt
import csv
import calendar
import time

# Helper function for reading Google trends data
def read_csv(input_filepath, delimiter, skip_rows, date_index, count_index):
  data = [];
  with open(input_filepath, "r") as f:
    reader = csv.reader(f, delimiter=delimiter)
    lines = [[x.strip() for x in row if len(x.strip()) > 0] for row in reader]

    for i, line in enumerate(lines):
      if i in skip_rows:
        continue

      try:
        datetime = calendar.timegm(time.strptime(line[date_index], '%Y-%m'))
      except:
        datetime = calendar.timegm(time.strptime(line[date_index], '%Y-%m-%d'))

      data.append([float(line[count_index]), datetime])

  return data

# Currently only works for two-dimensional embeddings
def plot_embedding(embedded, input_filepath, dimensions):
  x = [row[dimensions[0]] for row in embedded]
  y = [row[dimensions[1]] for row in embedded]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.set_ylabel(r"$x(t+{0}*\tau)$".format(dimensions[0]))
  ax.set_xlabel(r"$x(t+{0}*\tau)$".format(dimensions[1]))
  plt.scatter(y, x)

  plt.savefig("{0}.png".format(input_filepath+".embed")) 
