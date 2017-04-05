
import embedding
import utilities

if __name__ == "__main__":

  input_filepath = "data/influenza.csv"
  delay = 6; dimension = 2;

  #input_filepath = "data/baseball.csv"
  #delay = 6; dimension = 2;

  #input_filepath = "data/supreme_court.csv"
  #delay = 30; dimension = ;

  #input_filepath = "data/beyonce.csv"
  #delay = 27

  data = utilities.read_csv(input_filepath, ",", [0, 1, 2], 0, 1)
  
  embedding.mutual_information(data, input_filepath, delay*2)
  embedding.false_nearest_neighbors(data, input_filepath, delay)

  embedded = embedding.embedding(data, delay, dimension)

  utilities.plot_embedding(embedded, input_filepath, [0, 1])

  embedding.recurrence(data, input_filepath, delay)