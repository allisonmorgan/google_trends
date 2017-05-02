
import matplotlib.pyplot as plt
import os
import csv

# Find the best tau. Construct a plot of average mutual info as a 
# function of tau and identify the first minimum. 
#
# Currently, this is just a wrapper for the TISEAN package and will
# fail if TISEAN is not installed.

tisean_filepath = "tisean"

def mutual_information(input_filepath, delay, save_output=True):
  # Run the TISEAN mutual function
  output_filepath = input_filepath + ".mi"
  command = '{tisean}/mutual "{input}" -c2 -D {delay} -o "{output}"'.format(
    tisean=tisean_filepath, 
    input=input_filepath, 
    delay=delay,
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
  plt.plot(index, mi, color='blue')
  plt.xlim(0, max(index))

  if save_output:
    plt.savefig("{0}.png".format(output_filepath))

  return [index, mi]


# Find the best embedding dimension. Construct a plot of percentage of
# false nearest neighbors as a function of m and identiy the m for 
# which the percentage drops below 10%

def false_nearest_neighbors(input_filepath, delay, theiler, min_dim, max_dim, ratio, save_output=True):
  # Run the TISEAN false nearest neighbors function
  output_filepath = input_filepath + ".fnn"
  command = '{tisean}/false_nearest "{input}" -c2 -d {delay} -t {theiler} -M {min_dim},{max_dim} -f {ratio} -o "{output}"'.format(
    tisean=tisean_filepath, 
    input=input_filepath, 
    delay=int(delay),
    theiler=int(theiler),
    min_dim=min_dim,
    max_dim=max_dim,
    ratio=float(ratio),
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
  plt.axhline(y=0.10, color ='k', linestyle='dashed')
  plt.plot(mi, fnn, color='blue')

  if save_output:
    plt.savefig("{0}.png".format(output_filepath)) 

  return [mi, fnn] 

def recurrence(input_filepath, delay, save_output=True):
  # Run the TISEAN mutual function
  output_filepath = input_filepath + ".recurr"
  command = '{tisean}/recurr "{input}" -m 1,4 -c2 -d {delay} -o "{output}"'.format(
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
  plt.plot(ti, tj, color='blue')

  if save_output:
    plt.savefig("{0}.png".format(output_filepath))  

  return [ti, tj]

# Takes data ([(x_1, t_1), (x_2, t_2)]), period tau (float, amount of 
# time between steps), and embedding dimension m (int) and returns the
# embedded data [(x_1, ..., x_m), ... ] 

def embedding(input_filepath, data, tau, m, column_name, save_output=True):
  # Determine the period tau in terms of indices
  n = len(data); points = [];
  #delta = int(round(tau/(data[1][1] - data[0][1])))
  delta = tau

  for i, row in enumerate(data[column_name]):
    point = [];
    # Find the points one, two, three, etc. indices away
    for j in range(m):
      if (i + j*delta < len(data[column_name])):
        point.append(data[column_name][i + j*delta])
    
    # If we were unable to collect m coordinates (as in, we reached 
    # the end of the list), don't add that entry to our data
    if len(point) == m:
      points.append(point)

  with open(input_filepath + ".embed", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(points)

  return points
