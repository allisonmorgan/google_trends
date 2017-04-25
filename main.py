
import embedding
import utilities

if __name__ == "__main__":

  input_filepath = "data/influenza_hourly.csv"
  keyword = 'Influenza'

  data = utilities.read_csv(input_filepath, "   ")
  print(data[:5])
  utilities.plot_series(data, input_filepath, keyword)
  
  embedding.mutual_information(input_filepath, len(data))

  delay = 100
  theiler = 5
  min_dim = 1; max_dim = 10;
  ratio = 10.0;
  embedding.false_nearest_neighbors(input_filepath, delay, theiler, min_dim, max_dim, ratio)

  m = 5
  embedded = embedding.embedding(input_filepath, data, delay, m, keyword)
  utilities.plot_embedding(embedded, input_filepath, [1, 2])

  embedding.recurrence(input_filepath, delay)