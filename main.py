
import embedding
import utilities
import argparse
import copy

import knn

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Prediction of Google search trends options')
  parser.add_argument('--keyword', type=str, default="baseball", help="Google search trend to use")
  parser.add_argument('--k', type=int, default=5, help="Number of nearest points to use for KNN")
  parser.add_argument('--multistep', type=bool, default=False, help="Perform a multi-step forecast (feed predictions back into our training set), or perform predictions a single-step forecast")
  args = parser.parse_args()

  trend_options = {
    "baseball": ["data/baseball_hourly.csv", "Baseball", 5, 4], 
    "influenza": ["data/influenza_hourly.csv", "Influenza", 13, 5],
    "full moon": ["data/fullmoon_hourly.csv", "Full moon", 14, 5]
  }

  if not trend_options.has_key(args.keyword):
    print("Unsupported keyword supplied")
    exit()

  input_filepath, keyword, delay, m = trend_options[args.keyword]

  data = utilities.read_csv(input_filepath, "   ")
  utilities.plot_series(data, input_filepath, keyword)
  
  embedding.mutual_information(input_filepath, len(data))

  theiler = 0
  min_dim = 1; max_dim = 10;
  ratio = 10.0;
  embedding.false_nearest_neighbors(input_filepath, delay, theiler, min_dim, max_dim, ratio)

  embedded = embedding.embedding(input_filepath, data, delay, m, keyword)
  utilities.plot_embedding(embedded, input_filepath, [1, 2])

  #embedding.recurrence(input_filepath, delay)

  # args.k = 5 # baseball (Error: 0.387174821025)
  # args.k = 5 # influenza (Error: 1.25175439578)
  # args.k = 5 # full mooon (Error: 0.907941254943)

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
