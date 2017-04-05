
import matplotlib.pyplot as plt
import os
import csv

# Find the best tau. Construct a plot of average mutual info as a 
# function of tau and identify the first minimum. 
#
# Currently, this is just a wrapper for the TISEAN package and will
# fail if TISEAN is not installed.

tisean_filepath = "/Users/allisonmorgan/Code/bin/tisean"

def mutual_information(data, input_filepath, delay):
  # Run the TISEAN mutual function
  output_filepath = input_filepath + ".mi"
  command = '{tisean}/mutual "{input}" -D {delay} -o "{output}"'.format(
    tisean=tisean_filepath, 
    input=input_filepath, 
    delay=int(delay),
    output=output_filepath)

  var = os.system(command)

  # Open the output file, and generate a plot of mutual information
  # versus index.
  index = []; mi = [];
  with open(output_filepath, "r") as f:
    lines = csv.reader(f, delimiter=' ')
    for i, line in enumerate(lines):
      if i == 0:
        continue

      index.append(int(line[0]))
      mi.append(float(line[1]))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylabel(r"Mutual Information")
  ax.set_xlabel(r"Index")
  plt.plot(index, mi)
  plt.savefig("{0}.png".format(output_filepath))


# Find the best embedding dimension. Construct a plot of percentage of
# false nearest neighbors as a function of m and identiy the m for 
# which the percentage drops below 10%

def false_nearest_neighbors(data, input_filepath, delay):
  # Run the TISEAN false nearest neighbors function
  output_filepath = input_filepath + ".fnn"
  command = '{tisean}/false_nearest "{input}" -d {delay} -o "{output}"'.format(
    tisean=tisean_filepath, 
    input=input_filepath, 
    delay=int(delay),
    output=output_filepath)

  var = os.system(command)

  # Open the output file, and generate a plot of percentage of false
  # nearest neighbors versus dimension.
  mi = []; fnn = [];
  with open(output_filepath, "r") as f:
    lines = csv.reader(f, delimiter=' ')
    for i, line in enumerate(lines):
      mi.append(int(line[0]))
      fnn.append(float(line[1]))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylabel(r"False Nearest Neighbors")
  ax.set_xlabel(r"Dimension")
  plt.plot(mi, fnn)
  plt.savefig("{0}.png".format(output_filepath))  

def recurrence(data, input_filepath, delay):
  # Run the TISEAN mutual function
  output_filepath = input_filepath + ".recurr"
  command = '{tisean}/recurr "{input}" -m 1,2 -d {delay} -o "{output}"'.format(
    tisean=tisean_filepath, 
    input=input_filepath, 
    delay=int(delay),
    output=output_filepath)

  var = os.system(command)

  # Open the output file, and generates a recurrence plot
  ti = []; tj = [];
  with open(output_filepath, "r") as f:
    lines = csv.reader(f, delimiter=' ')
    for i, line in enumerate(lines):
      ti.append(float(line[0]))
      tj.append(float(line[1]))

  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.set_ylabel(r"Time")
  ax.set_xlabel(r"Time")
  plt.plot(ti, tj)

  plt.savefig("{0}.png".format(output_filepath))  

# Takes data ([(x_1, t_1), (x_2, t_2)]), period tau (float, amount of 
# time between steps), and embedding dimension m (int) and returns the
# embedded data [(x_1, ..., x_m), ... ] 

def embedding(data, tau, m):
  # Determine the period tau in terms of indices
  n = len(data); points = [];
  #delta = int(round(tau/(data[1][1] - data[0][1])))
  delta = tau

  for i, _ in enumerate(data):
    point = [];
    # Find the points one, two, three, etc. indices away
    for j in range(m):
      if (i + j*delta < len(data)):
        point.append(data[i + j*delta][0])
    
    # If we were unable to collect m coordinates (as in, we reached 
    # the end of the list), don't add that entry to our data
    if len(point) == m:
      points.append(point)

  return points
