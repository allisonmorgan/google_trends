# Predicting Google Search Trends

Final Project for CSCI 5466 - Chaotic Dynamics

## Introduction

The goals of this project were to see whether of not some Google search trends are predictable. Class presentation can be found in [`slides.pdf`].

I relied heavily on Joshua Garland and Liz Bradley's ["Prediction in Projection"](https://arxiv.org/pdf/1503.01678.pdf), which described how forecasting methods can be very accurate under incomplete embeddings. This 

### Setup

This package requires `numpy`, `sklearn`, `matplotlib` and `pandas`.

The jupyter notebook, `parse_google_trends.ipynb`, contains code for downloading, and cleaning a given Google search trend. To learn more read Samantha Molnar's [blog post](http://samanthamolnar.me/personal/2017/05/02/hacking-google-trends.html).

To play with one of the trends already downloaded ("baseball", "influenza", and "full moon"), you may run:

```{bash}
python main.py "full moon"
``` 

This will generate a plot of the [time series](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.trend.png), [mutual information](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.mi.png), percentage of [false nearest neighbors](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.fnn.png), a [two-dimensional embedding](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.embed.png), and our [prediction](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv_prediction.png) of the trend.

#### Works Cited

I would highly recommend checking out 

The file `tigramite_preprocessing.py` comes from [Tigramite](https://github.com/jakobrunge/tigramite). 

```
J. Runge et al., Nature Communications, 6, 8502 (2015)
J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths, Phys. Rev. Lett. 108, 258701 (2012)
J. Runge, J. Heitzig, N. Marwan, and J. Kurths, Phys. Rev. E 86, 061121 (2012)
J. Runge, V. Petoukhov, and J. Kurths, Journal of Climate, 27.2 (2014)
```

The file `embedding.py` contains python wrappers for some functions from the time series analysis package, [TISEAN](https://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html). Binaries relevant to this project have been reproduced in the `tisean` folder.

```
R. Hegger, H. Kantz, and T. Schreiber, Practical implementation of nonlinear time series methods: The TISEAN package, CHAOS 9, 413 (1999)
```