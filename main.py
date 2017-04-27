
import embedding
import utilities
import argparse
import copy

import knn

if __name__ == "__main__":
  input_filepath = "data/baseball_hourly.csv"
  #input_filepath = "data/influenza_hourly.csv"
  #input_filepath = "data/fullmoon_hourly.csv"
  keyword = 'Baseball'
  #keyword = 'Influenza'
  #keyword = 'Full moon'

  data = utilities.read_csv(input_filepath, "   ")
  print(data[:5])
  utilities.plot_series(data, input_filepath, keyword)
  
  embedding.mutual_information(input_filepath, len(data))

  delay = 5 # baseball
  #delay = 13 # influenza
  #delay = 14 # full moon
  theiler = 0
  min_dim = 1; max_dim = 10;
  ratio = 10.0;
  embedding.false_nearest_neighbors(input_filepath, delay, theiler, min_dim, max_dim, ratio)

  m = 4 # baseball
  #m = 5 # influenza
  #m = 5 # full moon

  embedded = embedding.embedding(input_filepath, data, delay, m, keyword)
  utilities.plot_embedding(embedded, input_filepath, [1, 2])

  #embedding.recurrence(input_filepath, delay)

  parser = argparse.ArgumentParser(description='KNN classifier options')
  parser.add_argument('--k', type=int, default=3, help="Number of nearest points to use")
  parser.add_argument('--multistep', type=bool, default=False, help="Perform a multi-step forecast (feed predictions back into our training set), or perform predictions on each point in our training set")
  args = parser.parse_args()

  # args.k = 5 # baseball (Error: 0.387174821025)
  # args.k = 5 # influenza (Error: 1.25175439578)
  # args.k = 5 # full mooon (Error: 0.876690272743)

  if args.multistep:
    print("Since multi-step forecast is {0}, number of nearest neighbors (currently {1}) must be set to 1".format(args.multistep, args.k))
    args.k = 1

  data = knn.Data(input_filepath + ".embed")

  k = knn.Knearest(data.train_x, data.train_y, args.k)
  print("Done loading data for KNN k = {0}".format(args.k))

  prediction = []; truth = [];
  test = copy.copy(data.train_x)
  for i in range(len(data.train_x)):
    xx, yy = test[i], data.train_y[i]
    prediction_embedded = k.classify(xx)

    prediction.append(prediction_embedded[0])
    truth.append(yy[0])

    if args.multistep and (i + 1) < len(data.test_x):
      test[i + 1] = prediction_embedded

  # Calculate the average error
  training = [row[0] for row in data.train_x]
  print("Error on training set: {0}".format(k.error(prediction, truth, training)))

  prediction = []; truth = [];
  test = copy.copy(data.test_x)
  for i in range(len(data.test_x)):
    xx, yy = test[i], data.test_y[i]
    prediction_embedded = k.classify(xx)

    prediction.append(prediction_embedded[0])
    truth.append(yy[0])

    if args.multistep and (i + 1) < len(data.test_x):
      test[i + 1] = prediction_embedded

  # Calculate the average error
  training = [row[0] for row in data.train_x]
  print("Error on test set: {0}".format(k.error(prediction, truth, training)))

  utilities.plot_prediction(input_filepath, truth, prediction, len(data.test_x))
