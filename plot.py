import pandas as pd

import embedding
import utilities

if __name__ == "__main__":
  input_filepath = "~/Documents/chaotic_dynamics/fullmoon_pastweek.csv"
  keyword = 'Full moon'

  data = pd.read_csv(input_filepath, parse_dates=[0])
  print(data[:5])
  utilities.plot_series(data, "fullmoon_pastweek", keyword, save_output = True)