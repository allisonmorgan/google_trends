
import embedding
import utilities

if __name__ == "__main__":

  input_filepath = "data/baseball.csv"
  keyword = 'Baseball'

  data = utilities.read_csv(input_filepath, ",")
  #utilities.plot_series(data, input_filepath, keyword)
  
  #embedding.mutual_information(input_filepath, 10*365)
  delay = 200
  theiler = 0
  min_dim = 1; max_dim = 10;
  ratio = 2.5;
  embedding.false_nearest_neighbors(input_filepath, delay, theiler, min_dim, max_dim, ratio)
  #m = 2

  #embedded = embedding.embedding(data, delay, m, keyword)

  #utilities.plot_embedding(embedded, input_filepath, [0, 1])

  #embedding.recurrence(input_filepath, delay)