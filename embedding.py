
import os
import csv
import calendar
import time

# Find the best tau. Construct a plot of average mutual info as a 
# function of tau and identify the first minimum. 
#
# Currently, this is just a wrapper for the TISEAN package and will
# fail if TISEAN is not installed.

tisean_filepath = "/Users/allisonmorgan/Code/bin/tisean"

def mutual_information(data, input_filepath, output_filepath, delay):
  # Run the TISEAN mutual function
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

  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylabel(r"Mutual Information")
  ax.set_xlabel(r"Index")
  plt.plot(index, mi)
  plt.savefig("{0}.png".format(output_filepath))

# Takes data ([(x_1, t_1), (x_2, t_2)]), period tau (float, amount of 
# time between steps), and embedding dimension m (int) and returns the
# embedded data [(x_1, ..., x_m), ... ] 

def embedding(data, tau, m):
  # Determine the period tau in terms of indices
  n = len(data); points = [];
  delta = int(round(tau/(data[1][1] - data[0][1])))

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


if __name__ == "__main__":

  input_filepath = "data/influenza.csv"
  output_filepath = "data/influenza.csv.out"
  delay = 12

  #input_filepath = "data/baseball.csv"
  #output_filepath = "data/baseball.csv.out"
  #delay = 12

  #input_filepath = "data/supreme_court.csv"
  #output_filepath = "data/supreme_court.csv.out"
  #delay = 50

  #input_filepath = "data/beyonce.csv"
  #output_filepath = "data/beyonce.csv.out"
  #delay = 200

  data = read_csv(input_filepath, ",", [0, 1, 2], 0, 1)
  
  mutual_information(data, input_filepath, output_filepath, delay)

  #embedded = embedding(data, 0.15, 7)




